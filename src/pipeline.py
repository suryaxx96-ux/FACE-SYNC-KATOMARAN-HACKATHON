"""End-to-end pipeline: capture -> detect (every N frames) -> track -> recognise -> log."""
import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.database import Database, face_label
from src.detector import FaceDetector
from src.event_logger import EventLogger, setup_logger
from src.recognizer import FaceRecognizer
from src.stream import LatestFrameCapture
from src.tracker import FaceTracker, Track


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        st = cfg["storage"]
        reset = st.get("reset_on_start", True)
        self.log = setup_logger(st["log_file"], reset=reset)
        self.db = Database(st["db_path"], reset=reset)
        source = cfg["input"]
        source_name = (
            "rtsp_stream"
            if source["mode"] == "rtsp"
            else os.path.splitext(os.path.basename(source["video_path"]))[0]
        )
        self.events = EventLogger(self.db, st["logs_dir"], self.log, source_name)
        self.detector = FaceDetector(cfg["detection"])
        self.recognizer = FaceRecognizer(cfg["recognition"])
        tracking_cfg = dict(cfg["tracking"])
        self.reentry_grace_frames = tracking_cfg.pop("reentry_grace_frames", 120)
        self.tracker = FaceTracker(**tracking_cfg)
        self.rcfg = cfg["recognition"]
        self.skip = cfg["detection"].get("detection_skip_frames", 0)
        self.scrfd_fallback_interval = max(
            1, int(cfg["detection"].get("scrfd_fallback_interval", 3))
        )
        self.scrfd_fallback_size = max(
            256, int(cfg["detection"].get("scrfd_fallback_size", 512))
        )
        self.detection_passes = 0
        self.last_scrfd_count = 0
        self.last_detection_count = 0
        self.gallery: Dict[int, np.ndarray] = self.db.load_faces()
        self.active: Dict[int, int] = {}  # face_id -> number of live tracks
        # Delay EXIT briefly so temporary detector/tracker gaps don't create
        # duplicate entry/exit photo pairs for the same continuous appearance.
        self.pending_exits: Dict[int, dict] = {}
        self._seen_tracks = set()          # track ids already announced in the log
        self.frame_idx = 0
        self._audit("pipeline_started", "Pipeline ready", metadata={
            "detection_skip_frames": self.skip,
            "gallery_size": len(self.gallery),
            "source_mode": source["mode"],
            "source_name": source_name,
        })

    def _audit(self, event_type: str, summary: str,
               face_id: Optional[int] = None, track_id: Optional[int] = None,
               metadata: Optional[dict] = None) -> Optional[int]:
        """Write critical metadata to SQLite and mirror it to events.log."""
        timestamp = datetime.now()
        details = dict(metadata or {})
        details["summary"] = summary
        try:
            audit_id = self.db.add_audit_event(
                event_type, timestamp, face_id=face_id, track_id=track_id,
                frame_idx=self.frame_idx, metadata=details,
            )
        except Exception as exc:
            # Keep the file log useful if SQLite is temporarily unavailable.
            self.log.exception(
                "CRITICAL EVENT DB WRITE FAILED | type=%s | face=%s | track=%s | frame=%d | metadata=%s | %s",
                event_type, face_label(face_id) if face_id is not None else "-",
                track_id if track_id is not None else "-", self.frame_idx,
                json.dumps(details, sort_keys=True, separators=(",", ":")), exc,
            )
            return None
        self._log_audit_record(audit_id, event_type, timestamp, face_id, track_id, details)
        return audit_id

    def _log_audit_record(self, audit_id: int, event_type: str, timestamp: datetime,
                          face_id: Optional[int], track_id: Optional[int], details: dict) -> None:
        """Mirror an already committed audit row to the mandatory text log."""
        self.log.info(
            "CRITICAL EVENT | audit_id=%d | type=%s | face=%s | timestamp=%s "
            "| track=%s frame=%d | metadata=%s",
            audit_id, event_type.upper(), face_label(face_id) if face_id is not None else "-",
            timestamp.isoformat(timespec="milliseconds"),
            track_id if track_id is not None else "-", self.frame_idx,
            json.dumps(details, sort_keys=True, separators=(",", ":")),
        )

    # ---------- input ----------
    def _open_capture(self):
        inp = self.cfg["input"]
        if inp["mode"] == "rtsp":
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            src = inp["rtsp_url"]
            cap = LatestFrameCapture(src)  # live: always process the newest frame
        else:
            src = inp["video_path"]
            cap = cv2.VideoCapture(src)    # file: process every frame in order
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")
        self.log.info("Opened %s source: %s", inp["mode"], src)
        return cap

    def _reconnect(self) -> Optional[cv2.VideoCapture]:
        inp = self.cfg["input"]
        for n in range(1, inp.get("reconnect_attempts", 10) + 1):
            self.log.warning("Stream lost - reconnect attempt %d", n)
            time.sleep(inp.get("reconnect_delay_sec", 3))
            try:
                return self._open_capture()
            except RuntimeError:
                continue
        return None

    # ---------- identity ----------
    def _match(self, emb: np.ndarray) -> Tuple[Optional[int], float]:
        if not self.gallery:
            return None, 0.0
        ids = list(self.gallery.keys())
        sims = np.stack([self.gallery[i] for i in ids]) @ emb
        k = int(np.argmax(sims))
        return (ids[k], float(sims[k])) if sims[k] >= self.rcfg["similarity_threshold"] else (None, float(sims[k]))

    def _identify(self, t: Track) -> None:
        emb = np.mean(t.embeddings, axis=0)
        emb = (emb / np.linalg.norm(emb)).astype(np.float32)
        fid, sim = self._match(emb)
        if fid is None:
            timestamp = datetime.now()
            details = {
                "best_similarity": sim,
                "unique_faces": len(self.gallery) + 1,
                "summary": "New face registered",
            }
            fid, audit_id = self.db.register_face(
                emb, timestamp, t.track_id, self.frame_idx, details,
            )
            self.gallery[fid] = emb
            self._log_audit_record(audit_id, "registration", timestamp, fid, t.track_id, details)
        else:
            self._audit(
                "recognition", "Existing face recognized", face_id=fid, track_id=t.track_id,
                metadata={"similarity": sim},
            )
            if sim >= self.rcfg["gallery_update_threshold"]:  # slow EMA keeps gallery fresh
                new = 0.9 * self.gallery[fid] + 0.1 * emb
                updated_embedding = (new / np.linalg.norm(new)).astype(np.float32)
                timestamp = datetime.now()
                details = {"similarity": sim, "summary": "Face gallery embedding updated"}
                audit_id = self.db.update_embedding_with_audit(
                    fid, updated_embedding, timestamp, t.track_id, self.frame_idx, details,
                )
                self.gallery[fid] = updated_embedding
                self._log_audit_record(
                    audit_id, "embedding_gallery_update", timestamp, fid, t.track_id, details,
                )
        t.face_id = fid
        pending_exit = self.pending_exits.pop(fid, None)
        self.active[fid] = self.active.get(fid, 0) + 1
        if self.active[fid] == 1:  # first live track for this face -> one ENTRY event
            if pending_exit is None:
                crop = t.best_crop if t.best_crop is not None else t.last_crop
                self.events.log_event("entry", fid, t.track_id, self.frame_idx, crop)
            else:
                self._audit(
                    "visit_resumed", "Face reacquired within the visit grace period",
                    face_id=fid, track_id=t.track_id,
                    metadata={"gap_frames": self.frame_idx - pending_exit["frame_idx"],
                              "grace_frames": self.reentry_grace_frames},
                )

    def _flush_pending_exits(self, force: bool = False) -> None:
        """Commit delayed exits after the re-entry grace period expires."""
        for face_id, pending in list(self.pending_exits.items()):
            gap = self.frame_idx - pending["frame_idx"]
            if not force and gap <= self.reentry_grace_frames:
                continue
            self.pending_exits.pop(face_id, None)
            self.events.log_event(
                "exit", face_id, pending["track_id"], pending["frame_idx"],
                pending["crop"], timestamp=pending["timestamp"],
            )

    def _finalize(self, t: Track) -> None:
        self._audit(
            "tracking_end", "Track ended", face_id=t.face_id, track_id=t.track_id,
            metadata={"hits": t.hits, "identified": t.face_id is not None},
        )
        if t.face_id is None:
            required_samples = self.rcfg["min_samples"]
            if len(t.embeddings) >= required_samples:
                # Normally identification happens as soon as min_samples is reached.
                # Keep this guard for tracks finalized after unusual control flow.
                self._identify(t)
            else:
                self._audit(
                    "track_dropped", "Track ended before sufficient valid embeddings",
                    track_id=t.track_id,
                    metadata={"valid_embeddings": len(t.embeddings),
                              "required_embeddings": required_samples, "hits": t.hits},
                )
                return
        fid = t.face_id
        self.active[fid] -= 1
        if self.active[fid] <= 0:  # last live track for this face -> one EXIT event
            self.active.pop(fid, None)
            # Hold EXIT briefly: if this identity is reacquired after a short
            # detector gap, the visit continues and no duplicate pair is logged.
            crop = t.last_crop if t.last_crop is not None else t.best_crop
            self.pending_exits[fid] = {
                "frame_idx": self.frame_idx,
                "track_id": t.track_id,
                "timestamp": datetime.now(),
                "crop": crop.copy() if crop is not None else None,
            }

    # ---------- per-track work ----------
    def _process_observed(self, t: Track, frame) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in t.box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        t.last_crop = crop.copy()
        if t.face_id is not None or t.rejected or t.hits < self.tracker.min_hits:
            return
        if min(x2 - x1, y2 - y1) < self.rcfg["min_face_size"]:
            return
        gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharp < self.rcfg["min_sharpness"]:
            return
        if sharp > t.best_sharp:
            t.best_sharp, t.best_crop = sharp, crop.copy()
        t.attempts += 1
        emb = self.recognizer.embed(frame, (x1, y1, x2, y2))
        if emb is not None:
            t.embeddings.append(emb)
            self._audit(
                "embedding_generated", "Face embedding generated", track_id=t.track_id,
                metadata={"sample_number": len(t.embeddings),
                          "required_samples": self.rcfg["min_samples"]},
            )
        if len(t.embeddings) >= self.rcfg["min_samples"]:
            self._identify(t)
        elif t.attempts >= self.rcfg["max_attempts"]:
            t.rejected = True
            self._audit(
                "track_rejected", "Track rejected because no reliable embedding was produced",
                track_id=t.track_id, metadata={"attempts": t.attempts},
            )

    # ---------- drawing ----------
    def _draw(self, frame) -> None:
        for t in self.tracker.tracks:
            # Keep rejected detections (commonly back-of-head false positives)
            # out of the annotated output after face-quality checks fail.
            if t.rejected:
                continue
            x1, y1, x2, y2 = [int(v) for v in t.box]
            ok = t.face_id is not None
            color = (0, 200, 0) if ok else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = face_label(t.face_id) if ok else "analyzing..."
            cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"Unique visitors: {self.db.unique_count()}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ---------- main loop ----------
    def run(self) -> int:
        out_cfg = self.cfg["output"]
        cap = None
        writer = None
        t0 = time.time()
        try:
            cap = self._open_capture()
            while True:
                stop_file = self.cfg.get("_stop_file")
                if stop_file and os.path.exists(stop_file):
                    self.log.info("Cooperative stop requested by dashboard")
                    break
                ok, frame = cap.read()
                if not ok:
                    if self.cfg["input"]["mode"] == "rtsp":
                        cap.release()
                        cap = self._reconnect()
                        if cap is None:
                            self.log.error("Could not reconnect - stopping")
                            break
                        continue
                    self.log.info("End of video")
                    break
                self.frame_idx += 1
                # Expire exits before processing this frame so a face returning
                # after the grace period starts a new visit as expected.
                self._flush_pending_exits()

                dets = self.detector.detect(frame) if (self.frame_idx - 1) % (self.skip + 1) == 0 else None
                if dets is not None:
                    self.detection_passes += 1
                    # A second, independent face detector periodically recovers
                    # small/dim/profile faces missed by YOLO. Its proposals use
                    # the same tracker and recognition quality gates.
                    if self.detection_passes % self.scrfd_fallback_interval == 0:
                        self.last_scrfd_count = 0
                        try:
                            scrfd_boxes = self.recognizer.detect_faces(
                                frame, self.detector.tile_fraction,
                                self.scrfd_fallback_size,
                            )
                            self.last_scrfd_count = len(scrfd_boxes)
                            if scrfd_boxes:
                                dets = self.detector._nms(dets + scrfd_boxes)
                        except Exception:
                            self.log.exception("SCRFD fallback failed; continuing with YOLO detections")
                    self.last_detection_count = len(dets)
                removed = self.tracker.step(dets)
                for t in self.tracker.tracks:
                    if t.track_id not in self._seen_tracks:
                        self._seen_tracks.add(t.track_id)
                        self._audit(
                            "tracking_start", "Track started", track_id=t.track_id,
                            metadata={"box": [float(value) for value in t.box]},
                        )
                for t in self.tracker.tracks:
                    if t.observed_now:
                        self._process_observed(t, frame)
                for t in removed:
                    self._finalize(t)

                if out_cfg.get("annotated_video_path") or out_cfg.get("show_window"):
                    self._draw(frame)
                if out_cfg.get("annotated_video_path"):
                    if writer is None:
                        os.makedirs(os.path.dirname(out_cfg["annotated_video_path"]) or ".", exist_ok=True)
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25
                        hh, ww = frame.shape[:2]
                        writer = cv2.VideoWriter(out_cfg["annotated_video_path"],
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
                    writer.write(frame)
                if out_cfg.get("show_window"):
                    cv2.imshow("Face Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                n = out_cfg.get("stats_every_n_frames", 100)
                if n and self.frame_idx % n == 0:
                    fps_now = self.frame_idx / max(1e-6, time.time() - t0)
                    dropped = getattr(cap, "dropped", 0)
                    self.log.info("STATS | frame=%d fps=%.1f detections=%d scrfd=%d active_tracks=%d unique=%d dropped=%d",
                                  self.frame_idx, fps_now, self.last_detection_count,
                                  self.last_scrfd_count,
                                  len(self.tracker.tracks),
                                  self.db.unique_count(), dropped)
        except KeyboardInterrupt:
            self.log.warning("Interrupted - flushing active tracks")
        finally:
            try:
                # One problematic track must not prevent the remaining tracks from
                # being flushed or the capture/database resources from closing.
                for t in list(self.tracker.tracks):
                    try:
                        self._finalize(t)
                    except Exception:
                        self.log.exception("Track finalization failed | track=%s", t.track_id)
                try:
                    self._flush_pending_exits(force=True)
                except Exception:
                    self.log.exception("Could not flush pending exit events")
            finally:
                if cap is not None:
                    cap.release()
                if writer:
                    writer.release()
                try:
                    cv2.destroyAllWindows()
                except cv2.error:  # headless OpenCV builds have no GUI
                    pass
                try:
                    total = self.db.unique_count()
                    self.log.info("DONE | frames=%d | UNIQUE VISITORS=%d | events=%s",
                                  self.frame_idx, total, self.db.event_counts())
                finally:
                    try:
                        self.db.close()
                    finally:
                        # Release the log file so IDLE reruns and Windows cleanup work cleanly.
                        for handler in self.log.handlers[:]:
                            self.log.removeHandler(handler)
                            handler.flush()
                            handler.close()
        return total
