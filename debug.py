import json
import sys

import cv2

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer

video = sys.argv[1] if len(sys.argv) > 1 else "data/sample4.mp4"
cfg = json.load(open("config.json", encoding="utf-8"))
rc = cfg["recognition"]

cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {video} | opened={cap.isOpened()} | frames={total} | "
      f"size={int(cap.get(3))}x{int(cap.get(4))} | fps={cap.get(cv2.CAP_PROP_FPS):.1f}")
if not cap.isOpened() or total <= 0:
    sys.exit("Video could not be read.")

det = FaceDetector(cfg["detection"])
rec = FaceRecognizer(rc)

step = max(1, total // 40)
frames_with_faces = n_faces = n_big = n_sharp = n_embed = 0
sizes = []
for idx in range(0, total, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    dets = det.detect(frame)
    if dets:
        frames_with_faces += 1
    for x1, y1, x2, y2, conf in dets:
        n_faces += 1
        w, h = int(x2 - x1), int(y2 - y1)
        sizes.append(min(w, h))
        crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if crop.size == 0:
            continue
        big = min(w, h) >= rc["min_face_size"]
        sharp_val = cv2.Laplacian(cv2.cvtColor(cv2.resize(crop, (112, 112)),
                                               cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        sharp = sharp_val >= rc["min_sharpness"]
        emb = rec.embed(frame, (x1, y1, x2, y2)) is not None
        n_big += big
        n_sharp += sharp
        n_embed += emb
        print(f"  frame {idx}: conf={conf:.2f} size={w}x{h} big_enough={big} "
              f"sharpness={sharp_val:.0f} sharp_enough={sharp} embedding_ok={emb}")

print("\n===== SUMMARY =====")
print(f"Frames sampled with >=1 face : {frames_with_faces}")
print(f"Faces detected by YOLO       : {n_faces}")
print(f"  big enough (>= {rc['min_face_size']}px)    : {n_big}")
print(f"  sharp enough (>= {rc['min_sharpness']})   : {n_sharp}")
print(f"  ArcFace embedding succeeded : {n_embed}")
if sizes:
    print(f"Face sizes (min side px): min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.0f}")