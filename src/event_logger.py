"""Central logging: events.log, structured image folders, and DB events."""
import logging
import os
import re
import tempfile
from datetime import datetime

import cv2

from src.database import Database, face_label


def setup_logger(log_file: str, reset: bool = False) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = logging.getLogger("face_tracker")
    logger.setLevel(logging.INFO)
    # Close handlers before replacing them (important when rerunning in IDLE).
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    fh = logging.FileHandler(log_file, mode="w" if reset else "a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class EventLogger:
    """Write one lossless, full-resolution image and record per entry/exit event."""

    FOLDERS = {"entry": "entries", "exit": "exits"}

    def __init__(self, db: Database, logs_dir: str, logger: logging.Logger,
                 source_name: str = "video"):
        self.db, self.logs_dir, self.log = db, logs_dir, logger
        # Keep folder names readable and prevent paths/URL credentials entering them.
        source_name = os.path.splitext(os.path.basename(source_name.rstrip("\\/")))[0]
        self.source_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("._") or "video"
        self._recover_log()

    def _recover_log(self) -> None:
        """Restore committed DB rows missing from events.log after an abrupt stop."""
        log_path = next((h.baseFilename for h in self.log.handlers
                         if isinstance(h, logging.FileHandler)), None)
        logged = set()
        logged_audit = set()
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as stream:
                    for line in stream:
                        match = re.search(r"event_id=(\d+)", line)
                        if match:
                            logged.add(int(match.group(1)))
                        audit_match = re.search(r"audit_id=(\d+)", line)
                        if audit_match:
                            logged_audit.add(int(audit_match.group(1)))
            except OSError as exc:
                self.log.error("Could not scan event log for recovery: %s", exc)
        for row in self.db.events_for_recovery():
            event_id, face_id, kind, timestamp, image_path, track_id, frame_idx = row
            if event_id in logged:
                continue
            self.log.warning(
                "EVENT LOG RECOVERED | event_id=%d | %s | face=%s track=%s frame=%s timestamp=%s image=%s",
                event_id, kind.upper(), face_label(face_id), track_id, frame_idx, timestamp, image_path,
            )
        for row in self.db.audit_events_for_recovery():
            audit_id, kind, timestamp, face_id, track_id, frame_idx, image_path, metadata = row
            if audit_id in logged_audit:
                continue
            self.log.warning(
                "CRITICAL EVENT RECOVERED | audit_id=%d | type=%s | face=%s | timestamp=%s "
                "| track=%s frame=%s image=%s metadata=%s",
                audit_id, kind.upper(), face_label(face_id) if face_id is not None else "-",
                timestamp, track_id, frame_idx, image_path, metadata,
            )

    def _save_image(self, event_type: str, face_id: int, crop, ts: datetime):
        if crop is None or crop.size == 0:
            return None
        folder = os.path.join(
            self.logs_dir,
            self.FOLDERS[event_type],
            self.source_name,
            ts.strftime("%Y-%m-%d"),
        )
        os.makedirs(folder, exist_ok=True)
        # Windows filenames cannot contain '/', so use hyphens as date/time separators.
        # Add a numeric suffix only if the same face/type is saved in the same second.
        stem = f"{face_label(face_id)}_{event_type.upper()}_{ts.strftime('%d-%m-%Y_%H-%M-%S')}"
        name = f"{stem}.png"
        path = os.path.join(folder, name)
        suffix = 2
        while os.path.exists(path):
            path = os.path.join(folder, f"{stem}_{suffix}.png")
            suffix += 1
        # PNG is lossless; keep the original crop dimensions and pixel values.
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ok, encoded = cv2.imencode(".png", crop, params)
        if not ok:
            return None
        # Write and fsync a temp file before atomic rename; a crash cannot leave
        # a partially written image at the final event path.
        fd, temp_path = tempfile.mkstemp(prefix=".event_", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "wb") as image_file:
                image_file.write(encoded.tobytes())
                image_file.flush()
                os.fsync(image_file.fileno())
            os.replace(temp_path, path)
            return path
        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

    def log_event(self, event_type: str, face_id: int, track_id: int, frame_idx: int, crop,
                  timestamp: datetime = None) -> None:
        ts = timestamp or datetime.now()
        try:
            path = self._save_image(event_type, face_id, crop, ts)
        except Exception as e:  # never let disk issues kill the stream
            self.log.error("Image save failed: %s", e)
            path = None
        if path is None:
            self.log.error(
                "EVENT INCOMPLETE | type=%s | face=%s | timestamp=%s | track=%d frame=%d | image unavailable; DB row omitted",
                event_type.upper(), face_label(face_id), ts.isoformat(timespec="milliseconds"),
                track_id, frame_idx,
            )
            return
        try:
            event_id, audit_id = self.db.add_event(
                face_id, event_type, ts, path, track_id, frame_idx
            )
        except Exception as e:
            # Avoid an orphan image if the durable DB transaction did not commit.
            try:
                os.remove(path)
            except OSError:
                pass
            self.log.error(
                "EVENT INCOMPLETE | type=%s | face=%s | timestamp=%s | DB write failed: %s",
                event_type.upper(), face_label(face_id), ts.isoformat(timespec="milliseconds"), e,
            )
            return
        self.log.info(
            "EVENT | audit_id=%d | type=%s | event_id=%d | face=%s | timestamp=%s | track=%d frame=%d image=%s",
            audit_id, event_type.upper(), event_id, face_label(face_id),
            ts.isoformat(timespec="milliseconds"), track_id, frame_idx, path,
        )
