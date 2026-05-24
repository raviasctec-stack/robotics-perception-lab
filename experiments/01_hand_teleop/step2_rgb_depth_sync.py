"""Step 2: synchronized RGB + depth from the OAK-D, with MediaPipe overlay.

Improvements over the first cut:
- Stereo tuned for close-range hand work (extended disparity, no subpixel)
- Post-processing: median + speckle + spatial (with hole-fill) + temporal filters
- Depth clamped to the hand workspace (100 - 2500 mm) so the colormap uses its
  full dynamic range for the region we care about
- MediaPipe Hands draws the skeleton on the RGB pane
- Depth is sampled with a 5x5 median at the wrist landmark, not just image
  center, so the readout is stable and meaningful

Open: http://localhost:8080
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
W, H = 1280, 720
JPEG_QUALITY = 75

DEPTH_MIN_MM = 100    # threshold filter min
DEPTH_MAX_MM = 1500   # threshold filter max + colormap clamp; close-range tuned
DISPARITY_SHIFT = 30  # shifts stereo search closer; min depth ~18 cm, max ~2 m

WRIST_IDX = 0
WRIST_SAMPLE_HALF = 2  # 5x5 patch (half-side 2 = 5 pixels)

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


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Map uint16 depth (mm) to a JET false-color image (0 -> black)."""
    valid = depth_mm > 0
    clipped = np.clip(depth_mm, 0, DEPTH_MAX_MM)
    norm = (clipped.astype(np.float32) / DEPTH_MAX_MM * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[~valid] = (0, 0, 0)
    return colored


def sample_depth_patch(depth_mm: np.ndarray, u: int, v: int) -> int:
    """Median of the nonzero depth values in a small patch around (u, v).

    Single-pixel sampling is noisy and often hits an invalid (zero) pixel near
    the skin/edge boundary. Taking the median of a small patch + ignoring
    zeros gives a stable estimate.
    """
    h, w = depth_mm.shape
    u0, u1 = max(0, u - WRIST_SAMPLE_HALF), min(w, u + WRIST_SAMPLE_HALF + 1)
    v0, v1 = max(0, v - WRIST_SAMPLE_HALF), min(h, v + WRIST_SAMPLE_HALF + 1)
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[patch > 0]
    return int(np.median(valid)) if valid.size else 0


def draw_skeleton(bgr: np.ndarray, landmarks):
    h, w = bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(bgr, (x, y), 4, (0, 0, 255), -1)
    return pts


def build_oak_pipeline(pipeline: dai.Pipeline):
    """Color (CAM_A) + stereo depth (CAM_B/CAM_C) aligned to color. Tuned for hands."""
    color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = color_cam.requestOutput(size=(W, H), type=dai.ImgFrame.Type.NV12)

    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    left_out = left_cam.requestOutput(size=(1280, 800))
    right_out = right_cam.requestOutput(size=(1280, 800))

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(W, H)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)         # subpixel softens edges; off = crisper hand outline
    stereo.setExtendedDisparity(True) # extends close-range
    stereo.initialConfig.setDisparityShift(DISPARITY_SHIFT)  # push the searchable window
                                                              # closer (min ~18 cm)

    # Post-processing: filters applied in order below
    pp = stereo.initialConfig.postProcessing
    pp.thresholdFilter.minRange = DEPTH_MIN_MM
    pp.thresholdFilter.maxRange = DEPTH_MAX_MM
    pp.speckleFilter.enable = True
    pp.speckleFilter.speckleRange = 50
    pp.spatialFilter.enable = True
    pp.spatialFilter.holeFillingRadius = 2
    pp.spatialFilter.numIterations = 1
    pp.temporalFilter.enable = True

    # Median filter on the raw disparity (separate from spatial)
    stereo.initialConfig.setMedianFilter(
        dai.StereoDepthConfig.MedianFilter.KERNEL_5x5
    )

    left_out.link(stereo.left)
    right_out.link(stereo.right)

    return color_out.createOutputQueue(), stereo.depth.createOutputQueue()


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
        color_q, depth_q = build_oak_pipeline(pipeline)
        pipeline.start()
        print(f"[capture] pipeline up: color={W}x{H}, depth aligned + filtered")

        t0 = time.time()
        frames = 0
        last_log = t0

        while not _stop.is_set():
            color_msg = color_q.get()
            depth_msg = depth_q.get()
            bgr = color_msg.getCvFrame()
            depth_mm = depth_msg.getFrame()

            # Hand detection on RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t0) * 1000)
            result = detector.detect_for_video(mp_img, ts_ms)

            wrist_depth_mm = 0
            wrist_pixel = None
            if result.hand_landmarks:
                pts = draw_skeleton(bgr, result.hand_landmarks[0])
                wu, wv = pts[WRIST_IDX]
                wu = int(np.clip(wu, 0, W - 1))
                wv = int(np.clip(wv, 0, H - 1))
                wrist_depth_mm = sample_depth_patch(depth_mm, wu, wv)
                wrist_pixel = (wu, wv)

            # Overlay readouts
            depth_vis = colorize_depth(depth_mm)
            if wrist_pixel:
                cv2.drawMarker(bgr, wrist_pixel, (0, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.drawMarker(depth_vis, wrist_pixel, (255, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)
                label = f"wrist depth: {wrist_depth_mm} mm" if wrist_depth_mm \
                        else "wrist depth: (no measurement)"
            else:
                label = "no hand detected"
            cv2.putText(bgr, label, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(depth_vis, f"range: {DEPTH_MIN_MM}-{DEPTH_MAX_MM} mm",
                        (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            side_by_side = np.hstack([bgr, depth_vis])

            ok, buf = cv2.imencode(".jpg", side_by_side,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with _lock:
                    _latest_jpeg = buf.tobytes()

            frames += 1
            now = time.time()
            if now - last_log > 2.0:
                fps = frames / (now - t0)
                print(f"[capture] {frames} fr, {fps:.1f} FPS, wrist={wrist_depth_mm} mm")
                last_log = now

    detector.close()


INDEX_HTML = """<!doctype html>
<html><head><title>Step 2: RGB + depth + hand</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui;
display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh}
h2{margin:8px 0;font-weight:400}
img{max-width:98vw;max-height:88vh;border:1px solid #333}</style></head>
<body>
<h2>step 2 - synchronized RGB + depth with hand overlay</h2>
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
