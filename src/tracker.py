"""Lightweight IoU tracker with constant-velocity prediction.

On frames where detection is skipped (config: detection_skip_frames) boxes are
propagated using their estimated velocity, so IDs stay stable between cycles.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    last_obs: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    since_obs: int = 0
    observed_now: bool = True
    face_id: Optional[int] = None
    embeddings: list = field(default_factory=list)
    attempts: int = 0
    rejected: bool = False
    best_crop: Optional[np.ndarray] = None
    best_sharp: float = -1.0
    last_crop: Optional[np.ndarray] = None


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost_frames
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 1

    def step(self, detections: Optional[Sequence[Sequence[float]]] = None) -> List[Track]:
        """Advance one frame. Pass detections=None on skipped frames.

        Returns tracks removed this frame (i.e. faces that have exited)."""
        for t in self.tracks:
            t.box = t.box + t.vel
            t.since_obs += 1
            t.observed_now = False
            t.vel = t.vel * 0.9

        if detections is not None:
            dets = np.array([d[:4] for d in detections], dtype=float).reshape(-1, 4)
            matched_d = set()
            if self.tracks and len(dets):
                iou = iou_matrix([t.box for t in self.tracks], dets)
                rows, cols = linear_sum_assignment(-iou)
                for i, j in zip(rows, cols):
                    if iou[i, j] >= self.iou_threshold:
                        t, new = self.tracks[i], dets[j]
                        gap = max(1, t.since_obs)
                        t.vel = 0.5 * t.vel + 0.5 * (new - t.last_obs) / gap
                        t.box, t.last_obs = new.copy(), new.copy()
                        t.since_obs, t.hits, t.observed_now = 0, t.hits + 1, True
                        matched_d.add(j)
            for j in range(len(dets)):
                if j not in matched_d:
                    self.tracks.append(Track(self._next_id, dets[j].copy(), dets[j].copy()))
                    self._next_id += 1

        removed = [t for t in self.tracks if t.since_obs > self.max_lost]
        self.tracks = [t for t in self.tracks if t.since_obs <= self.max_lost]
        return removed
