"""Step 4: hand-to-robot retargeting, live.

What's new vs step 3:
- Uses PALM CENTER (avg of LM 0, 5, 17) instead of raw wrist -> stable when
  fingers move. This fixes the 12 mm pinch-induced wrist jitter we measured.
- Outputs both CAMERA-frame and ROBOT-frame coordinates. Robot frame is what
  a real arm driver would consume (step 5 / future SO-101 hookup).
- Workspace mapping with per-axis clipping. When the user reaches outside
  the configured hand workspace, the target stays at the robot's workspace
  edge instead of going wild. A small WS indicator shows when each axis is
  clipped.
- Gripper value (0=closed, 1=open) from scale-invariant pinch ratio,
  rendered as a horizontal bar.

Open http://localhost:8080.
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
from shared.retarget import (  # noqa: E402
    CAM_WORKSPACE, ROBOT_WORKSPACE,
    RetargetResult,
    camera_to_robot, gripper_value,
    map_workspace, palm_center_pixel,
    sample_depth_patch,
)
from shared.smoothing import EmaFilter  # noqa: E402

MODEL = REPO_ROOT / "models" / "hand_landmarker.task"
EMA_ALPHA = 0.30

HOST, PORT = "0.0.0.0", 8080
W, H = 1280, 720
JPEG_QUALITY = 75

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 1500
DISPARITY_SHIFT = 30

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


def draw_overlay_box(bgr: np.ndarray, lines: list[str], scale: float = 0.85):
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


def draw_gripper_bar(bgr: np.ndarray, value: float):
    """Horizontal bar at the bottom of the RGB pane: full = open, empty = closed."""
    h, w = bgr.shape[:2]
    bar_h = 40
    pad = 12
    x0, x1 = pad, w - pad
    y0, y1 = h - bar_h - pad, h - pad
    # frame
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (255, 255, 255), 2)
    # fill
    fill_x = int(x0 + (x1 - x0) * value)
    color = (0, 220, 0) if value > 0.4 else (0, 120, 255)
    cv2.rectangle(bgr, (x0 + 2, y0 + 2), (fill_x, y1 - 2), color, -1)
    label = f"gripper {value:.2f}  ({'OPEN' if value > 0.4 else 'CLOSED'})"
    cv2.putText(bgr, label, (x0 + 8, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_palm_marker(bgr: np.ndarray, u: int, v: int):
    cv2.drawMarker(bgr, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 28, 3)
    cv2.circle(bgr, (u, v), 18, (0, 255, 255), 2)


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
    print(f"[calib] fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.cx:.1f} cy={intr.cy:.1f}")
    print(f"[ws] cam   X{CAM_WORKSPACE['x_min']:+.2f}..{CAM_WORKSPACE['x_max']:+.2f}  "
          f"Y{CAM_WORKSPACE['y_min']:+.2f}..{CAM_WORKSPACE['y_max']:+.2f}  "
          f"Z{CAM_WORKSPACE['z_min']:.2f}..{CAM_WORKSPACE['z_max']:.2f}")
    print(f"[ws] robot X{ROBOT_WORKSPACE['x_min']:+.2f}..{ROBOT_WORKSPACE['x_max']:+.2f}  "
          f"Y{ROBOT_WORKSPACE['y_min']:+.2f}..{ROBOT_WORKSPACE['y_max']:+.2f}  "
          f"Z{ROBOT_WORKSPACE['z_min']:.2f}..{ROBOT_WORKSPACE['z_max']:.2f}")

    with dai.Pipeline(device) as pipeline:
        color_q, depth_q = build_oak_pipeline(pipeline)
        pipeline.start()
        print("[capture] retargeting pipeline up")

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

            retargeted = None
            palm_pixel = None
            xyz_cam_raw = None
            xyz_cam_smooth = None

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_skeleton(bgr, landmarks)
                u, v = palm_center_pixel(landmarks, W, H)
                u = int(np.clip(u, 0, W - 1)); v = int(np.clip(v, 0, H - 1))
                palm_pixel = (u, v)
                d_mm = sample_depth_patch(depth_mm, u, v)
                if d_mm > 0:
                    xyz_cam_raw = intr.unproject(u, v, d_mm)
                    xyz_cam_smooth = smoother.update(xyz_cam_raw)
                    xyz_for_map = xyz_cam_smooth if xyz_cam_smooth is not None else xyz_cam_raw
                    xyz_rob = camera_to_robot(xyz_for_map)
                    xyz_target, clipped = map_workspace(xyz_rob)
                    grip = gripper_value(landmarks, W, H)
                    retargeted = RetargetResult(
                        palm_pixel=palm_pixel,
                        palm_depth_mm=d_mm,
                        xyz_cam=xyz_cam_raw,
                        xyz_cam_smooth=xyz_cam_smooth,
                        xyz_robot=xyz_target,
                        gripper=grip,
                        clipped=clipped,
                    )
                else:
                    # no depth - keep smoother in sync by passing None
                    smoother.update(None)
                # always draw the palm marker if we have the pixel
                draw_palm_marker(bgr, u, v)
            else:
                smoother.update(None)

            depth_vis = colorize_depth(depth_mm)
            if palm_pixel:
                cv2.drawMarker(depth_vis, palm_pixel, (255, 255, 255),
                               cv2.MARKER_CROSS, 28, 3)

            # Build overlay text
            if retargeted is not None:
                r = retargeted
                clip_tag = "".join(c.upper() if cl else c.lower()
                                   for c, cl in zip("xyz", r.clipped))
                lines = [
                    f"palm pixel: ({palm_pixel[0]:4d}, {palm_pixel[1]:4d})    "
                    f"depth: {r.palm_depth_mm} mm",
                    f"CAM 3D   : X={r.xyz_cam[0]:+.3f}  Y={r.xyz_cam[1]:+.3f}  Z={r.xyz_cam[2]:.3f}",
                    (f"CAM smth : X={r.xyz_cam_smooth[0]:+.3f}  "
                     f"Y={r.xyz_cam_smooth[1]:+.3f}  "
                     f"Z={r.xyz_cam_smooth[2]:.3f}")
                    if r.xyz_cam_smooth is not None else "CAM smth : -",
                    f"ROBOT TGT: X={r.xyz_robot[0]:+.3f}  Y={r.xyz_robot[1]:+.3f}  "
                    f"Z={r.xyz_robot[2]:.3f}    clip:{clip_tag}",
                ]
            elif palm_pixel:
                lines = [
                    f"palm pixel: ({palm_pixel[0]:4d}, {palm_pixel[1]:4d})    "
                    "depth: (none)",
                    "CAM 3D   : -",
                    "CAM smth : -",
                    "ROBOT TGT: -",
                ]
            else:
                lines = ["no hand detected", "", "", ""]

            draw_overlay_box(bgr, lines)
            if retargeted is not None:
                draw_gripper_bar(bgr, retargeted.gripper)
            draw_overlay_box(depth_vis,
                             [f"depth   {DEPTH_MIN_MM}-{DEPTH_MAX_MM} mm"])

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
                if retargeted is not None:
                    r = retargeted
                    print(f"[capture] {frames} fr, {fps:.1f} FPS  "
                          f"robot ({r.xyz_robot[0]:+.3f}, {r.xyz_robot[1]:+.3f}, "
                          f"{r.xyz_robot[2]:.3f})  grip {r.gripper:.2f}")
                else:
                    print(f"[capture] {frames} fr, {fps:.1f} FPS  (no retarget)")
                last_log = now

    detector.close()


INDEX_HTML = """<!doctype html>
<html><head><title>Step 4: retarget</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui;
display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh}
h2{margin:8px 0;font-weight:400}
img{max-width:98vw;max-height:88vh;border:1px solid #333}</style></head>
<body>
<h2>step 4 - hand -> robot retargeting (live)</h2>
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
