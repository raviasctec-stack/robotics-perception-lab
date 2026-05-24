"""Step 1 (live): stream OAK + MediaPipe Hands overlay to the browser.

This is the interactive variant of step1_mediapipe_still.py. Instead of
snapshotting one frame, it runs continuously and renders the detected hand
skeleton on top of the live RGB feed. Open http://localhost:8080 in any
browser to view.

Stop with Ctrl-C in the terminal.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = REPO_ROOT / "models" / "hand_landmarker.task"

HOST, PORT = "0.0.0.0", 8080
WIDTH, HEIGHT = 1280, 720
JPEG_QUALITY = 80

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_latest_jpeg: bytes | None = None
_lock = threading.Lock()
_stop = threading.Event()


def draw_landmarks(bgr, landmarks):
    h, w = bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(bgr, (x, y), 5, (0, 0, 255), -1)


def capture_loop():
    global _latest_jpeg

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        out = cam.requestOutput(size=(WIDTH, HEIGHT), type=dai.ImgFrame.Type.NV12)
        q = out.createOutputQueue()
        pipeline.start()
        print(f"[capture] OAK pipeline started at {WIDTH}x{HEIGHT}")

        t_start = time.time()
        frames = 0
        last_log = t_start

        while not _stop.is_set():
            bgr = q.get().getCvFrame()
            ts_ms = int((time.time() - t_start) * 1000)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, ts_ms)

            hands_found = len(result.hand_landmarks)
            if hands_found:
                draw_landmarks(bgr, result.hand_landmarks[0])

            label = f"hands: {hands_found}"
            cv2.putText(bgr, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with _lock:
                    _latest_jpeg = buf.tobytes()

            frames += 1
            now = time.time()
            if now - last_log > 2.0:
                fps = frames / (now - t_start)
                print(f"[capture] {frames} frames, {fps:.1f} FPS, last detection: {hands_found} hand(s)")
                last_log = now

    detector.close()


INDEX_HTML = """<!doctype html>
<html><head><title>Step 1 live: hand landmarks</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh}
h2{margin:8px 0;font-weight:400}
img{max-width:95vw;max-height:85vh;border:1px solid #333}</style></head>
<body>
<h2>step 1 / live - MediaPipe Hands on OAK</h2>
<img src="/stream.mjpg">
</body></html>
""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _stop.is_set():
                    with _lock:
                        jpg = _latest_jpeg
                    if jpg is None:
                        time.sleep(0.05); continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg); self.wfile.write(b"\r\n")
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        self.send_response(404); self.end_headers()


def main():
    if not MODEL.exists():
        raise SystemExit(f"Missing model: {MODEL}")

    cap = threading.Thread(target=capture_loop, daemon=True)
    cap.start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[server] open http://localhost:{PORT}   (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopping...")
    finally:
        _stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
