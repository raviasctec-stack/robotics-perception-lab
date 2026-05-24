"""Step 3 (recording mode): capture a session for offline analysis.

Outputs a self-contained folder under captures/session_TIMESTAMP/ containing:

  recording.mp4   the composite RGB|depth view with overlays (for visual review)
  telemetry.csv   per-frame data: time, hand_detected, u, v, depth_mm, X, Y, Z,
                  depth_valid_pct, fps_inst
  frame_NNN.png   evenly-spaced sample frames (~one every 2s) for inspection
  summary.txt     auto-computed stats: detection rate, jitter (std of X/Y/Z when
                  the hand is held still), dropout rate, FPS distribution

Workflow:

    # Hold your hand at a steady pose; the script records for N seconds.
    python step3_record.py 20

    # Then share the captures/session_*/ folder for analysis.
"""

import argparse
import csv
import sys
import threading
import time
from datetime import datetime
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

HOST, PORT = "0.0.0.0", 8080
JPEG_QUALITY = 75

_latest_jpeg: bytes | None = None
_lock = threading.Lock()
_stop = threading.Event()


INDEX_HTML = """<!doctype html>
<html><head><title>Step 3 recording - live preview</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui;
display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh}
h2{margin:8px 0;font-weight:400;color:#f55}
img{max-width:98vw;max-height:88vh;border:2px solid #f55}</style></head>
<body>
<h2>* RECORDING - step 3 *</h2>
<img src="/stream.mjpg">
</body></html>
""".encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
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
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
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


def _start_live_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((HOST, PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from shared.oak import Intrinsics, get_intrinsics  # noqa: E402
from shared.smoothing import EmaFilter  # noqa: E402

MODEL = REPO_ROOT / "models" / "hand_landmarker.task"
EMA_ALPHA = 0.30  # same as live viewer; log raw + smoothed both for comparison

W, H = 1280, 720

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 1500
DISPARITY_SHIFT = 30

WRIST_IDX = 0
WRIST_SAMPLE_HALF = 2

SAMPLE_PNG_INTERVAL_S = 2.0  # save a sample frame every 2s

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_overlay_box(bgr: np.ndarray, lines: list[str], scale: float = 0.9):
    """Semi-transparent panel + bright text; readable after browser scale-down."""
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


def run_recording(duration_s: float, out_dir: Path, session_label: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "recording.mp4"
    csv_path = out_dir / "telemetry.csv"
    summary_path = out_dir / "summary.txt"

    # Reset module-level state so multiple calls in one process are safe.
    global _latest_jpeg
    _stop.clear()
    with _lock:
        _latest_jpeg = None

    server = _start_live_server()
    print(f"[live] preview at http://localhost:{PORT}  (during recording)")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    device = dai.Device()
    intr: Intrinsics = get_intrinsics(device, dai.CameraBoardSocket.CAM_A, W, H)
    print(f"[calib] fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.cx:.1f} cy={intr.cy:.1f}")

    # MP4 writer: vertical stack -> W wide, 2*H tall
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # We don't know real FPS until after recording, so use 30 as a stand-in
    writer = cv2.VideoWriter(str(mp4_path), fourcc, 30.0, (W, 2 * H))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter at {mp4_path}")

    csv_file = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_file)
    csv_w.writerow([
        "t_s", "hand_detected", "u", "v", "depth_mm",
        "X_m", "Y_m", "Z_m",
        "X_smooth_m", "Y_smooth_m", "Z_smooth_m",
        "depth_valid_pct", "fps_inst",
    ])

    smoother = EmaFilter(alpha=EMA_ALPHA, max_dropouts=10)

    with dai.Pipeline(device) as pipeline:
        color_q, depth_q = build_oak_pipeline(pipeline)
        pipeline.start()
        print(f"[record] starting {duration_s:.1f}s session  ->  {out_dir}")
        print(f"[record] hold your hand steady at a comfortable pose")

        t0 = time.time()
        last_log = t0
        last_sample_t = -SAMPLE_PNG_INTERVAL_S  # ensure first save
        sample_idx = 0
        frames_in_window = 0
        win_start = t0
        frames_total = 0

        while True:
            now = time.time()
            elapsed = now - t0
            if elapsed >= duration_s:
                break

            color_msg = color_q.get()
            depth_msg = depth_q.get()
            bgr = color_msg.getCvFrame()
            depth_mm = depth_msg.getFrame()

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, int(elapsed * 1000))

            hand_detected = bool(result.hand_landmarks)
            wu = wv = -1
            wrist_depth_mm = 0
            X = Y = Z = float("nan")
            wrist_pixel = None
            raw_xyz = None

            if hand_detected:
                pts = draw_skeleton(bgr, result.hand_landmarks[0])
                wu, wv = pts[WRIST_IDX]
                wu = int(np.clip(wu, 0, W - 1))
                wv = int(np.clip(wv, 0, H - 1))
                wrist_depth_mm = sample_depth_patch(depth_mm, wu, wv)
                wrist_pixel = (wu, wv)
                if wrist_depth_mm > 0:
                    raw_xyz = intr.unproject(wu, wv, wrist_depth_mm)
                    X, Y, Z = float(raw_xyz[0]), float(raw_xyz[1]), float(raw_xyz[2])

            smooth_xyz = smoother.update(raw_xyz)
            Xs = float(smooth_xyz[0]) if smooth_xyz is not None else float("nan")
            Ys = float(smooth_xyz[1]) if smooth_xyz is not None else float("nan")
            Zs = float(smooth_xyz[2]) if smooth_xyz is not None else float("nan")

            depth_valid_pct = 100.0 * (depth_mm > 0).mean()

            # Composite frame (same as live)
            depth_vis = colorize_depth(depth_mm)
            if wrist_pixel:
                cv2.drawMarker(bgr, wrist_pixel, (0, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.drawMarker(depth_vis, wrist_pixel, (255, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)

            tag = f"REC t={elapsed:5.1f}s / {duration_s:.0f}s"
            if session_label:
                tag = f"{session_label}   {tag}"

            if not np.isnan(X):
                lines = [
                    tag,
                    f"RAW    : X={X:+.3f}  Y={Y:+.3f}  Z={Z:.3f}   depth: {wrist_depth_mm} mm",
                    (f"SMOOTH : X={Xs:+.3f}  Y={Ys:+.3f}  Z={Zs:.3f}   "
                     f"a={EMA_ALPHA}    valid: {depth_valid_pct:.0f}%")
                    if not np.isnan(Xs) else
                    f"SMOOTH : -    valid: {depth_valid_pct:.0f}%",
                ]
            elif hand_detected:
                lines = [
                    tag,
                    "RAW    : - (no depth at wrist)",
                    f"SMOOTH : -    valid: {depth_valid_pct:.0f}%",
                ]
            else:
                lines = [tag, "no hand",
                         f"valid: {depth_valid_pct:.0f}%"]

            draw_overlay_box(bgr, lines)
            draw_overlay_box(depth_vis,
                             [f"depth   {DEPTH_MIN_MM}-{DEPTH_MAX_MM} mm"])

            composite = np.vstack([bgr, depth_vis])
            writer.write(composite)

            # also push to live preview server (global declared at top of fn)
            ok, buf = cv2.imencode(".jpg", composite,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with _lock:
                    _latest_jpeg = buf.tobytes()

            # FPS over a 1s rolling window
            frames_in_window += 1
            if now - win_start >= 1.0:
                fps_inst = frames_in_window / (now - win_start)
                win_start = now
                frames_in_window = 0
            else:
                fps_inst = float("nan")

            csv_w.writerow([
                f"{elapsed:.3f}", int(hand_detected), wu, wv, wrist_depth_mm,
                f"{X:.4f}", f"{Y:.4f}", f"{Z:.4f}",
                f"{Xs:.4f}", f"{Ys:.4f}", f"{Zs:.4f}",
                f"{depth_valid_pct:.2f}", f"{fps_inst:.2f}",
            ])

            if elapsed - last_sample_t >= SAMPLE_PNG_INTERVAL_S:
                sample_idx += 1
                png = out_dir / f"frame_{sample_idx:03d}.png"
                cv2.imwrite(str(png), composite)
                last_sample_t = elapsed

            frames_total += 1
            if now - last_log > 2.0:
                print(f"[record] t={elapsed:5.1f}s  frames={frames_total}  "
                      f"hand={'Y' if hand_detected else 'n'}  "
                      f"depth={wrist_depth_mm} mm  "
                      f"valid={depth_valid_pct:.0f}%")
                last_log = now

    writer.release()
    csv_file.close()
    detector.close()
    _stop.set()
    server.shutdown()

    write_summary(csv_path, summary_path, frames_total,
                  time.time() - t0, intr)
    print(f"\n[done] recorded {frames_total} frames in "
          f"{time.time() - t0:.1f}s -> {out_dir}")


def write_summary(csv_path: Path, summary_path: Path, n_frames: int,
                  duration_s: float, intr: Intrinsics):
    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)

    detected = [r for r in rows if r["hand_detected"] == "1"]
    with_depth = [r for r in detected if int(r["depth_mm"]) > 0]

    xyz = np.array([[float(r["X_m"]), float(r["Y_m"]), float(r["Z_m"])]
                    for r in with_depth]) if with_depth else np.zeros((0, 3))
    depth_pcts = np.array([float(r["depth_valid_pct"]) for r in rows])

    lines = [
        "=== recording summary ===",
        f"frames               : {n}",
        f"duration             : {duration_s:.2f} s",
        f"avg FPS              : {n / max(duration_s, 1e-6):.1f}",
        f"intrinsics @ {intr.width}x{intr.height}: fx={intr.fx:.1f} fy={intr.fy:.1f} "
        f"cx={intr.cx:.1f} cy={intr.cy:.1f}",
        "",
        f"hand detection rate  : {100*len(detected)/n:.1f}% ({len(detected)}/{n})",
        f"  + with valid depth : {100*len(with_depth)/n:.1f}% ({len(with_depth)}/{n})",
        f"  - dropouts (hand   : {len(detected)-len(with_depth)} frames",
        f"    but no depth)",
        "",
        f"depth coverage (whole frame, avg of % valid pixels):",
        f"  mean               : {depth_pcts.mean():.1f}%",
        f"  median             : {np.median(depth_pcts):.1f}%",
        f"  min .. max         : {depth_pcts.min():.1f}% .. {depth_pcts.max():.1f}%",
        "",
    ]

    if len(xyz) >= 5:
        med = np.median(xyz, axis=0)
        std = np.std(xyz, axis=0)
        p95 = np.percentile(np.abs(xyz - med), 95, axis=0)
        lines += [
            "wrist 3D stats (over frames with valid hand+depth):",
            f"  median (m)         : X={med[0]:+.3f}  Y={med[1]:+.3f}  Z={med[2]:.3f}",
            f"  std    (mm)        : X={1000*std[0]:6.1f}  Y={1000*std[1]:6.1f}  Z={1000*std[2]:6.1f}",
            f"  p95 |dev|  (mm)    : X={1000*p95[0]:6.1f}  Y={1000*p95[1]:6.1f}  Z={1000*p95[2]:6.1f}",
            "",
            "interpretation:",
            f"  std X/Y < 5 mm and std Z < 15 mm = stable; held-still target.",
            f"  large p95 vs std = occasional spikes/outliers, not gaussian noise.",
        ]
    else:
        lines.append("not enough valid samples for jitter stats (need >= 5)")

    summary_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("duration_s", type=float, nargs="?", default=15.0,
                   help="recording length in seconds (default 15)")
    p.add_argument("--name", type=str, default=None,
                   help="optional session name (default: timestamp)")
    p.add_argument("--label", type=str, default="",
                   help="text shown on the overlay during recording")
    args = p.parse_args()

    if not MODEL.exists():
        raise SystemExit(f"Missing model: {MODEL}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"session_{stamp}" + (f"_{args.name}" if args.name else "")
    out_dir = REPO_ROOT / "captures" / name

    run_recording(args.duration_s, out_dir, session_label=args.label)


if __name__ == "__main__":
    main()
