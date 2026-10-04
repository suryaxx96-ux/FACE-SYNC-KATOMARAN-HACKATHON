"""ArcFace embeddings via InsightFace (buffalo_l = SCRFD + ArcFace w600k_r50).

The YOLO box is expanded by a margin and re-run through InsightFace's light
detector so the face gets 5-point landmark alignment before embedding. That
alignment is what makes ArcFace accurate.
"""
from typing import Optional, Sequence

import numpy as np


class FaceRecognizer:
    def __init__(self, cfg: dict):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if has_cuda else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name=cfg.get("model_name", "buffalo_l"),
                                allowed_modules=["detection", "recognition"],
                                providers=providers)
        det_size = max(160, int(cfg.get("det_size", 192)))
        self.app.prepare(ctx_id=0 if has_cuda else -1, det_size=(det_size, det_size),
                         det_thresh=float(cfg.get("det_threshold", 0.25)))
        self.margin = cfg.get("crop_margin", 0.3)
        self.min_face_detection_score = cfg.get("min_face_detection_score", 0.55)
        self.min_embedding_box_iou = cfg.get("min_embedding_box_iou", 0.1)
        self.max_landmark_offset_ratio = cfg.get("max_landmark_offset_ratio", 0.65)
        self.max_eye_vertical_offset_ratio = cfg.get("max_eye_vertical_offset_ratio", 0.55)
        self.det_size = det_size

    def detect_faces(self, frame, tile_fraction: float = 0.5,
                     input_size: int = 512):
        """Return SCRFD face-box proposals to recover misses from YOLO."""
        detector = getattr(self.app, "det_model", None)
        if detector is None:
            return []
        height, width = frame.shape[:2]
        tile_w = min(width, max(1, int(width * tile_fraction)))
        tile_h = min(height, max(1, int(height * tile_fraction)))
        xs = [0] if tile_w >= width else sorted({0, width - tile_w})
        ys = [0] if tile_h >= height else sorted({0, height - tile_h})
        regions = [(0, 0, frame)]
        regions.extend(
            (x, y, frame[y:y + tile_h, x:x + tile_w])
            for y in ys for x in xs
            if tile_w < width or tile_h < height
        )
        proposals = []
        min_score = float(getattr(detector, "det_thresh", 0.25))
        for offset_x, offset_y, region in regions:
            if region.size == 0:
                continue
            boxes, _ = detector.detect(
                region, input_size=(input_size, input_size),
                max_num=0, metric="default",
            )
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2, score = map(float, box[:5])
                if score < min_score:
                    continue
                x1, x2 = max(0.0, x1 + offset_x), min(float(width), x2 + offset_x)
                y1, y2 = max(0.0, y1 + offset_y), min(float(height), y2 + offset_y)
                if x2 > x1 and y2 > y1:
                    proposals.append((x1, y1, x2, y2, score))
        return proposals

    @staticmethod
    def _box_iou(a, b) -> float:
        """IoU for InsightFace's crop-relative box and the YOLO face box."""
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
        area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        return intersection / max(1e-9, area_a + area_b - intersection)

    def _face_quality(self, face, target_box) -> Optional[float]:
        """Reject weak, unrelated, or strongly turned detections before embedding."""
        score = float(getattr(face, "det_score", 0.0))
        if score < self.min_face_detection_score:
            return None
        bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
        if bbox.size < 4:
            return None
        overlap = self._box_iou(bbox[:4], target_box)
        if overlap < self.min_embedding_box_iou:
            return None

        # InsightFace's 5-point keypoints are ordered as eyes, nose, and mouth
        # corners. A weak eye span or a nose far from the eye midpoint usually
        # means an unusable profile/back-of-head or a false face detection.
        points = np.asarray(getattr(face, "kps", None), dtype=np.float32)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            return None
        eye_left, eye_right, nose = points[:3]
        mouth_left, mouth_right = points[3:]
        eye_span = float(np.linalg.norm(eye_right - eye_left))
        face_width = max(1.0, float(bbox[2] - bbox[0]))
        if eye_span < max(3.0, face_width * 0.12):
            return None
        eye_mid = (eye_left + eye_right) / 2.0
        # A large vertical eye mismatch is a strong profile/false-landmark cue.
        # Normalize to eye distance so this behaves similarly at different scales.
        if abs(float(eye_right[1] - eye_left[1])) / eye_span > self.max_eye_vertical_offset_ratio:
            return None
        pose_offset = abs(float(nose[0] - eye_mid[0])) / eye_span
        if pose_offset > self.max_landmark_offset_ratio:
            return None
        eye_y = float((eye_left[1] + eye_right[1]) / 2.0)
        mouth_y = float((mouth_left[1] + mouth_right[1]) / 2.0)
        face_height = max(1.0, float(bbox[3] - bbox[1]))
        vertical_tolerance = face_height * 0.1
        if eye_y > float(nose[1]) + vertical_tolerance:
            return None
        if float(nose[1]) > mouth_y + vertical_tolerance:
            return None
        return overlap * score

    def embed(self, frame, box: Sequence[float]) -> Optional[np.ndarray]:
        """Return an L2-normalised 512-d embedding, or None if no usable face."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - bw * self.margin)), max(0, int(y1 - bh * self.margin))
        cx2, cy2 = min(w, int(x2 + bw * self.margin)), min(h, int(y2 + bh * self.margin))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        faces = self.app.get(crop)
        if not faces:
            return None
        # Only use an InsightFace detection that corresponds to this YOLO box.
        # The expanded crop can contain another person's face near the target.
        target_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        candidates = [
            (quality, face)
            for face in faces
            if (quality := self._face_quality(face, target_box)) is not None
        ]
        if not candidates:
            return None
        _, f = max(candidates, key=lambda item: item[0])
        return f.normed_embedding.astype(np.float32)
