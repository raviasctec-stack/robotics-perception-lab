"""Step 3: unproject the wrist pixel + depth into 3D meters (camera frame).

The math (pinhole camera model):

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth_at(u, v)  in meters

where (fx, fy, cx, cy) are the OAK's factory-calibrated intrinsics for CAM_A
at the resolution we requested. This is THE foundational equation that turns
a 2D detection in an image into a real point in 3D space.

By the end of this step the wrist (X, Y, Z) overlay should:
  - X grow more positive as you move your hand right
  - X grow more negative as you move your hand left
  - Y grow more positive as you move your hand down  (OAK convention: Y points down)
  - Y grow more negative as you move your hand up
  - Z grow as you move your hand away from the camera

Camera frame axes (OAK / OpenCV convention):
    +X = right (as the camera sees it)
    +Y = down
    +Z = forward (out of the lens, away from camera)

Open: http://localhost:8080
"""

import sys
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
sys.path.insert(0, str(REPO_ROOT))
from shared.oak import Intrinsics, get_intrinsics  # noqa: E402
from shared.smoothing import EmaFilter  # noqa: E402

MODEL = REPO_ROOT / "models" / "hand_landmarker.task"
EMA_ALPHA = 0.30  # ~3-sample averaging window; tune for taste

HOST, PORT = "0.0.0.0", 8080
W, H = 1280, 720
JPEG_QUALITY = 75

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 1500
DISPARITY_SHIFT = 30

WRIST_IDX = 0
WRIST_SAMPLE_HALF = 2

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
    valid = depth_mm > 0
    clipped = np.clip(depth_mm, 0, DEPTH_MAX_MM)
    norm = (clipped.astype(np.float32) / DEPTH_MAX_MM * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[~valid] = (0, 0, 0)
    return colored


def sample_depth_patch(depth_mm: np.ndarray, u: int, v: int) -> int:
    h, w = depth_mm.shape
    u0, u1 = max(0, u - WRIST_SAMPLE_HALF), min(w, u + WRIST_SAMPLE_HALF + 1)
    v0, v1 = max(0, v - WRIST_SAMPLE_HALF), min(h, v + WRIST_SAMPLE_HALF + 1)
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[patch > 0]
    return int(np.median(valid)) if valid.size else 0


def draw_overlay_box(bgr: np.ndarray, lines: list[str], scale: float = 0.9):
    """Draw a semi-transparent black panel + bright text. Stays readable when
    the browser scales the image down."""
    if not lines:
        return
    line_h = int(36 * scale)
    pad = int(14 * scale)
    box_h = pad * 2 + line_h * len(lines)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (bgr.shape[1], box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0, dst=bgr)
    for i, line in enumerate(lines):
        y = pad + line_h * (i + 1) - 8
        cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), max(1, int(2 * scale)),
                    cv2.LINE_AA)


def draw_skeleton(bgr: np.ndarray, landmarks):
    h, w = bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(bgr, (x, y), 4, (0, 0, 255), -1)
    return pts


def build_oak_pipeline(pipeline: dai.Pipeline):
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
    stereo.setSubpixel(False)
    stereo.setExtendedDisparity(True)
    stereo.initialConfig.setDisparityShift(DISPARITY_SHIFT)

    pp = stereo.initialConfig.postProcessing
    pp.thresholdFilter.minRange = DEPTH_MIN_MM
    pp.thresholdFilter.maxRange = DEPTH_MAX_MM
    pp.speckleFilter.enable = True
    pp.speckleFilter.speckleRange = 50
    pp.spatialFilter.enable = True
    pp.spatialFilter.holeFillingRadius = 2
    pp.spatialFilter.numIterations = 1
    pp.temporalFilter.enable = True

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
    smoother = EmaFilter(alpha=EMA_ALPHA, max_dropouts=10)

    device = dai.Device()
    intr: Intrinsics = get_intrinsics(device, dai.CameraBoardSocket.CAM_A, W, H)
    print(f"[calib] CAM_A intrinsics @ {W}x{H}: "
          f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f}")

    with dai.Pipeline(device) as pipeline:
        color_q, depth_q = build_oak_pipeline(pipeline)
        pipeline.start()
        print("[capture] pipeline up, unprojecting wrist to 3D")

        t0 = time.time()
        frames = 0
        last_log = t0

        while not _stop.is_set():
            color_msg = color_q.get()
            depth_msg = depth_q.get()
            bgr = color_msg.getCvFrame()
            depth_mm = depth_msg.getFrame()

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t0) * 1000)
            result = detector.detect_for_video(mp_img, ts_ms)

            wrist_xyz = None
            wrist_pixel = None
            wrist_depth_mm = 0

            if result.hand_landmarks:
                pts = draw_skeleton(bgr, result.hand_landmarks[0])
                wu, wv = pts[WRIST_IDX]
                wu = int(np.clip(wu, 0, W - 1))
                wv = int(np.clip(wv, 0, H - 1))
                wrist_depth_mm = sample_depth_patch(depth_mm, wu, wv)
                wrist_pixel = (wu, wv)

                if wrist_depth_mm > 0:
                    # *** THE STEP-3 LINE: pixel + depth -> meters in camera frame ***
                    wrist_xyz = intr.unproject(wu, wv, wrist_depth_mm)

            wrist_xyz_smooth = smoother.update(wrist_xyz)

            depth_vis = colorize_depth(depth_mm)
            if wrist_pixel:
                cv2.drawMarker(bgr, wrist_pixel, (0, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.drawMarker(depth_vis, wrist_pixel, (255, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)

            # 4-line overlay on the RGB pane: raw vs smoothed
            if wrist_xyz is not None:
                lines = [
                    f"pixel  : ({wrist_pixel[0]:4d}, {wrist_pixel[1]:4d})    depth: {wrist_depth_mm} mm",
                    f"RAW    : X={wrist_xyz[0]:+.3f}  Y={wrist_xyz[1]:+.3f}  Z={wrist_xyz[2]:.3f}",
                    (f"SMOOTH : X={wrist_xyz_smooth[0]:+.3f}  "
                     f"Y={wrist_xyz_smooth[1]:+.3f}  "
                     f"Z={wrist_xyz_smooth[2]:.3f}   (a={EMA_ALPHA})")
                    if wrist_xyz_smooth is not None else "SMOOTH : -",
                ]
            elif wrist_pixel:
                lines = [
                    f"pixel  : ({wrist_pixel[0]:4d}, {wrist_pixel[1]:4d})    depth: (no measurement)",
                    "RAW    : -",
                    "SMOOTH : -",
                ]
            else:
                lines = ["no hand detected", "", ""]

            draw_overlay_box(bgr, lines)

            depth_label = [f"depth range  {DEPTH_MIN_MM} - {DEPTH_MAX_MM} mm"
                           f"   (blue near, red far, black = no measurement)"]
            draw_overlay_box(depth_vis, depth_label)

            # Vertical stack: RGB on top, depth below. Browser-friendly aspect.
            composite = np.vstack([bgr, depth_vis])
            ok, buf = cv2.imencode(".jpg", composite,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with _lock:
                    _latest_jpeg = buf.tobytes()

            frames += 1
            now = time.time()
            if now - last_log > 2.0:
                fps = frames / (now - t0)
                if wrist_xyz_smooth is not None:
                    msg = (f"smooth X={wrist_xyz_smooth[0]:+.3f} "
                           f"Y={wrist_xyz_smooth[1]:+.3f} "
                           f"Z={wrist_xyz_smooth[2]:.3f} m")
                else:
                    msg = "(no wrist 3D)"
                print(f"[capture] {frames} fr, {fps:.1f} FPS, {msg}")
                last_log = now

    detector.close()


INDEX_HTML = """<!doctype html>
<html><head><title>Step 3: wrist in 3D</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui;
display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh}
h2{margin:8px 0;font-weight:400}
img{max-width:98vw;max-height:88vh;border:1px solid #333}</style></head>
<body>
<h2>step 3 - wrist unprojected to camera-frame (X, Y, Z) meters</h2>
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
