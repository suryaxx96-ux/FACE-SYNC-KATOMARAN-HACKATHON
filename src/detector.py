"""YOLO face detector with full-frame and overlapping corner-tile coverage."""
from typing import List, Tuple

from ultralytics import YOLO

Detection = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, conf


class FaceDetector:
    def __init__(self, cfg: dict):
        self.model = YOLO(cfg["model_path"])
        self.conf = cfg.get("conf_threshold", 0.5)
        self.imgsz = cfg.get("imgsz", 640)
        self.tile_inference = cfg.get("tile_inference", False)
        self.tile_fraction = cfg.get("tile_fraction", 0.60)
        self.nms_iou = cfg.get("nms_iou_threshold", 0.45)
        self.nms_ios = cfg.get("nms_ios_threshold", 0.75)
        self.nms_center_threshold = cfg.get("nms_center_threshold", 0.60)
        device = cfg.get("device", "auto")
        self.device = None if device == "auto" else device

    @staticmethod
    def _starts(length: int, tile_length: int) -> List[int]:
        """Return edge-aligned starts so the overlapping tiles include both corners."""
        if tile_length >= length:
            return [0]
        return sorted({0, length - tile_length})

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Suppress duplicate boxes, including nested boxes from tiled detectors.

        IoU alone misses duplicates when one model/view returns a tight box and
        another returns a much larger box around the same face.  In that case
        intersection-over-smaller-area (IoS) is high even though IoU is low.
        The center-distance guard prevents nearby, partially overlapping faces
        from being merged just because one box contains much of the other.
        """
        if len(detections) < 2:
            return detections
        detections = sorted(detections, key=lambda det: det[4], reverse=True)
        kept: List[Detection] = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            remaining = []
            for candidate in detections:
                xx1 = max(best[0], candidate[0])
                yy1 = max(best[1], candidate[1])
                xx2 = min(best[2], candidate[2])
                yy2 = min(best[3], candidate[3])
                intersection = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                area_best = max(0.0, best[2] - best[0]) * max(0.0, best[3] - best[1])
                area_candidate = max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
                iou = intersection / max(1e-9, area_best + area_candidate - intersection)
                ios = intersection / max(1e-9, min(area_best, area_candidate))

                # Only apply the containment rule when both boxes describe the
                # same central face location. Distinct people can be close in
                # crowded footage and must remain separate detections.
                best_cx, best_cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
                cand_cx, cand_cy = (candidate[0] + candidate[2]) / 2, (candidate[1] + candidate[3]) / 2
                min_side = max(1.0, min(
                    best[2] - best[0], best[3] - best[1],
                    candidate[2] - candidate[0], candidate[3] - candidate[1],
                ))
                center_distance = ((best_cx - cand_cx) ** 2 + (best_cy - cand_cy) ** 2) ** 0.5
                nested_duplicate = (
                    ios >= self.nms_ios
                    and center_distance <= self.nms_center_threshold * min_side
                )

                if iou <= self.nms_iou and not nested_duplicate:
                    remaining.append(candidate)
            detections = remaining
        return kept

    def detect(self, frame) -> List[Detection]:
        height, width = frame.shape[:2]
        regions = [(0, 0, width, height, frame)]

        # The full image preserves global context. Overlapping quadrants add
        # pixels per face and catch small/edge faces lost during global resize.
        if self.tile_inference and min(height, width) >= 2:
            tile_w = min(width, max(1, int(width * self.tile_fraction)))
            tile_h = min(height, max(1, int(height * self.tile_fraction)))
            xs, ys = self._starts(width, tile_w), self._starts(height, tile_h)
            for y in ys:
                for x in xs:
                    if x == 0 and y == 0 and tile_w == width and tile_h == height:
                        continue
                    regions.append((x, y, x + tile_w, y + tile_h,
                                    frame[y:y + tile_h, x:x + tile_w]))

        results = self.model.predict(
            [region[4] for region in regions], imgsz=self.imgsz, conf=self.conf,
            device=self.device, verbose=False,
        )
        out: List[Detection] = []
        for result, (offset_x, offset_y, _, _, _) in zip(results, regions):
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Tile-relative coordinates become full-frame coordinates.
                x1, x2 = max(0.0, x1 + offset_x), min(float(width), x2 + offset_x)
                y1, y2 = max(0.0, y1 + offset_y), min(float(height), y2 + offset_y)
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2, y2, float(box.conf[0])))
        return self._nms(out)
