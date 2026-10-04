"""Latest-frame video capture for live (RTSP) streams.

A background thread reads the camera continuously and keeps only the newest
frame. If processing is slower than the camera, old frames are dropped instead
of queueing up, so the system always works on what the camera sees *now*.
"""
import threading
import time

import cv2


class LatestFrameCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0          # frames received from the camera
        self._last_seq = 0     # frames handed to the pipeline
        self._alive = self.cap.isOpened()
        self._stop = False
        self.dropped = 0       # frames skipped because processing was too slow
        self._thread = threading.Thread(target=self._loop, daemon=True)
        if self._alive:
            self._thread.start()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok:  # stream ended or connection dropped
                    self._alive = False
                    self._cond.notify_all()
                    break
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 10.0):
        """Return (True, newest_frame), or (False, None) if the stream is dead/stalled."""
        with self._cond:
            end = time.time() + timeout
            while self._seq == self._last_seq and self._alive:
                remaining = end - time.time()
                if remaining <= 0:
                    return False, None
                self._cond.wait(remaining)
            if self._seq == self._last_seq:  # stream died, nothing new
                return False, None
            self.dropped += self._seq - self._last_seq - 1
            self._last_seq = self._seq
            return True, self._frame

    def get(self, prop):
        return self.cap.get(prop)

    def release(self) -> None:
        self._stop = True
        self._thread.join(timeout=2)
        self.cap.release()
