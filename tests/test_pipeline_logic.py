"""Logic test for logging + counting, without needing YOLO/InsightFace or a camera.

Run:  python -m tests.test_pipeline_logic
"""
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np

SCENE = {
    "A": ([(1, 60), (120, 160)], (100, 100, 160, 160)),
    "B": ([(40, 90)], (300, 100, 360, 160)),
    "C": ([(200, 220)], (500, 100, 560, 160)),
}
EMB = {}
rng = np.random.default_rng(0)
for k in SCENE:
    v = rng.normal(size=512).astype(np.float32)
    EMB[k] = v / np.linalg.norm(v)


class FakeDetector:
    def __init__(self, cfg): pass

    def detect(self, frame):
        idx = int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])
        out = []
        for _, (spans, box) in SCENE.items():
            if any(a <= idx <= b for a, b in spans):
                out.append((*box, 0.9))
        return out


class FakeRecognizer:
    def __init__(self, cfg): pass

    def detect_faces(self, frame, tile_fraction=0.5, input_size=512):
        """Fake SCRFD fallback: this synthetic test scene uses FakeDetector only."""
        return []

    def embed(self, frame, box):
        for k, (_, b) in SCENE.items():
            if abs(box[0] - b[0]) < 30:
                e = EMB[k] + rng.normal(scale=0.01, size=512).astype(np.float32)
                return (e / np.linalg.norm(e)).astype(np.float32)
        return None


sys.modules["src.detector"] = types.SimpleNamespace(FaceDetector=FakeDetector)
sys.modules["src.recognizer"] = types.SimpleNamespace(FaceRecognizer=FakeRecognizer)


class FakeCapture:
    def __init__(self, src): self.i = 0
    def isOpened(self): return True
    def get(self, prop): return 25
    def release(self): pass

    def read(self):
        if self.i >= 260:
            return False, None
        self.i += 1
        f = np.zeros((480, 640, 3), np.uint8)
        f[0, 0, 0], f[0, 0, 1] = self.i % 256, self.i // 256
        return True, f


cv2.VideoCapture = FakeCapture

from src.pipeline import Pipeline  # noqa: E402


class PipelineLogicTest(unittest.TestCase):
    def test_counting_images_and_audit_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open("config.json", encoding="utf-8") as cfg_file:
                cfg = json.load(cfg_file)
            cfg["input"].update(mode="video", video_path="fake.mp4")
            cfg["detection"]["detection_skip_frames"] = 2
            cfg["recognition"].update(min_sharpness=0.0, min_samples=3, min_face_size=25)
            cfg["storage"].update(
                db_path=os.path.join(tmp, "t.db"), logs_dir=os.path.join(tmp, "logs"),
                log_file=os.path.join(tmp, "logs", "events.log"), reset_on_start=True,
            )
            cfg["output"].update(annotated_video_path="", show_window=False)

            unique = Pipeline(cfg).run()

            conn = sqlite3.connect(cfg["storage"]["db_path"])
            try:
                ev = conn.execute(
                    "SELECT face_id, event_type, image_path FROM events ORDER BY event_id"
                ).fetchall()
                audit_types = {row[0] for row in conn.execute(
                    "SELECT DISTINCT event_type FROM system_events"
                )}
            finally:
                conn.close()
            entries = [event for event in ev if event[1] == "entry"]
            exits = [event for event in ev if event[1] == "exit"]

            self.assertEqual(unique, 3)
            # Face A reappears within the configured 120-frame grace period,
            # so this is one continuous visit rather than another event pair.
            self.assertEqual((len(entries), len(exits)), (3, 3))
            self.assertTrue(all(path and os.path.exists(path) for _, _, path in ev))
            ids = [face_id for face_id, _, _ in entries]
            self.assertEqual(len(set(ids)), 3)
            self.assertTrue(all(ids.count(face_id) == 1 for face_id in set(ids)))
            self.assertTrue({
                "pipeline_started", "tracking_start", "tracking_end", "embedding_generated",
                "registration", "recognition", "visit_resumed", "entry", "exit",
            }.issubset(audit_types))

            with open(cfg["storage"]["log_file"], encoding="utf-8") as log_file:
                log = log_file.read()
            for event_type in (
                "EMBEDDING_GENERATED", "REGISTRATION", "RECOGNITION", "ENTRY", "EXIT",
                "VISIT_RESUMED", "TRACKING_START", "TRACKING_END",
            ):
                self.assertIn(f"type={event_type}", log)
            self.assertNotIn("SCRFD fallback failed", log)


if __name__ == "__main__":
    unittest.main()
