"""Step 5: 3D viewer with virtual gripper, driven live by your hand.

This is the end of the perception-side pipeline. Same upstream as step 4
(OAK + MediaPipe + retarget), but instead of writing the target pose to
text overlays, it streams it to a three.js scene in your browser. You see
a workspace box, coordinate axes, and a parallel-jaw gripper that
moves and opens/closes in real time as you move your hand.

When/if an SO-101 arm is added later, the upstream pipeline is unchanged;
only the consumer of the (xyz, gripper) stream swaps from "three.js viewer"
to "serial driver for the arm".

Streams:
    /             - three.js page with viewer + tiny camera thumbnail
    /state        - Server-Sent Events: one JSON per frame
    /preview.mjpg - MJPEG thumbnail of the camera + skeleton overlay

Open http://localhost:8080
"""

import json
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
    camera_to_robot, gripper_value,
    map_workspace, palm_center_pixel,
    sample_depth_patch,
)
from shared.smoothing import EmaFilter  # noqa: E402

MODEL = REPO_ROOT / "models" / "hand_landmarker.task"
EMA_ALPHA = 0.30

HOST, PORT = "0.0.0.0", 8080
W, H = 1280, 720
JPEG_QUALITY = 70
THUMB_DOWNSCALE = 2   # preview = W/2 wide (640x360 at the source)

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 1500
DISPARITY_SHIFT = 30

# Per-finger BGR colors (visible against a wide range of backgrounds).
FINGER_COLORS = {
    "thumb":  (180,  80, 255),   # pink/magenta
    "index":  ( 80, 230,  80),   # green
    "middle": ( 80, 230, 230),   # yellow
    "ring":   ( 80, 150, 255),   # orange
    "pinky":  (255, 120, 200),   # light purple
    "palm":   (220, 220, 220),   # light grey
}

# (a, b, finger-group) for each landmark pair
HAND_CONNECTIONS = [
    (0, 1, "thumb"), (1, 2, "thumb"), (2, 3, "thumb"), (3, 4, "thumb"),
    (0, 5, "palm"),
    (5, 6, "index"), (6, 7, "index"), (7, 8, "index"),
    (5, 9, "palm"),
    (9, 10, "middle"), (10, 11, "middle"), (11, 12, "middle"),
    (9, 13, "palm"),
    (13, 14, "ring"), (14, 15, "ring"), (15, 16, "ring"),
    (13, 17, "palm"),
    (17, 18, "pinky"), (18, 19, "pinky"), (19, 20, "pinky"),
    (0, 17, "palm"),
]

# Per-landmark colors (use the finger the landmark belongs to)
LANDMARK_COLORS = [
    "palm",                                 # 0 wrist
    "thumb", "thumb", "thumb", "thumb",     # 1-4
    "index", "index", "index", "index",     # 5-8
    "middle", "middle", "middle", "middle", # 9-12
    "ring", "ring", "ring", "ring",         # 13-16
    "pinky", "pinky", "pinky", "pinky",     # 17-20
]

_state_lock = threading.Lock()
_latest_state: dict = {"connected": False}

_thumb_lock = threading.Lock()
_latest_thumb: bytes | None = None

_stop = threading.Event()


def draw_skeleton(bgr: np.ndarray, landmarks):
    """Color-coded MediaPipe-style hand skeleton.

    Each finger gets its own color; palm connections are light grey; joints
    are white-haloed dots colored by their finger.
    """
    h, w = bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b, group in HAND_CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], FINGER_COLORS[group], 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        col = FINGER_COLORS[LANDMARK_COLORS[i]]
        cv2.circle(bgr, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)  # halo
        cv2.circle(bgr, (x, y), 3, col, -1, cv2.LINE_AA)


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
    global _latest_state, _latest_thumb

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
    print(f"[calib] fx={intr.fx:.1f} fy={intr.fy:.1f}")

    with dai.Pipeline(device) as pipeline:
        color_q, depth_q = build_oak_pipeline(pipeline)
        pipeline.start()
        print("[capture] step5 pipeline up")

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

            state = {
                "t": round(time.time() - t0, 3),
                "connected": True,
                "hand": False,
                "robot": None,
                "gripper": None,
                "clipped": [False, False, False],
            }

            palm_pixel_full = None
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                u, v = palm_center_pixel(landmarks, W, H)
                u = int(np.clip(u, 0, W - 1)); v = int(np.clip(v, 0, H - 1))
                palm_pixel_full = (u, v)
                d_mm = sample_depth_patch(depth_mm, u, v)
                if d_mm > 0:
                    xyz_cam = intr.unproject(u, v, d_mm)
                    smoothed = smoother.update(xyz_cam)
                    src = smoothed if smoothed is not None else xyz_cam
                    xyz_rob = camera_to_robot(src)
                    xyz_target, clipped = map_workspace(xyz_rob)
                    grip = gripper_value(landmarks, W, H)
                    state["hand"] = True
                    state["robot"] = [round(float(x), 4) for x in xyz_target]
                    state["gripper"] = round(float(grip), 3)
                    # clipped values come back as numpy bool_ from map_workspace -
                    # json.dumps refuses to serialize those, so coerce to Python bool.
                    state["clipped"] = [bool(c) for c in clipped]
                else:
                    smoother.update(None)
                    state["hand"] = True
            else:
                smoother.update(None)

            with _state_lock:
                _latest_state = state

            # Tiny camera thumbnail. Draw skeleton AFTER resize so the colors
            # stay crisp at thumbnail resolution (lines drawn on full-res then
            # downscaled get washed out by the 3x interpolation).
            thumb_w = W // THUMB_DOWNSCALE
            thumb_h = H // THUMB_DOWNSCALE
            thumb = cv2.resize(bgr, (thumb_w, thumb_h),
                               interpolation=cv2.INTER_AREA)
            if result.hand_landmarks:
                draw_skeleton(thumb, result.hand_landmarks[0])
                if palm_pixel_full is not None:
                    tu = palm_pixel_full[0] // THUMB_DOWNSCALE
                    tv = palm_pixel_full[1] // THUMB_DOWNSCALE
                    cv2.circle(thumb, (tu, tv), 9, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.drawMarker(thumb, (tu, tv), (0, 255, 255),
                                   cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", thumb,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with _thumb_lock:
                    _latest_thumb = buf.tobytes()

            frames += 1
            now = time.time()
            if now - last_log > 2.0:
                fps = frames / (now - t0)
                if state["robot"]:
                    print(f"[capture] {frames} fr, {fps:.1f} FPS  "
                          f"robot {tuple(state['robot'])}  grip {state['gripper']}")
                else:
                    print(f"[capture] {frames} fr, {fps:.1f} FPS  (no hand/depth)")
                last_log = now

    detector.close()


# ----- HTML / SSE / MJPEG --------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>step 5 - hand teleop viewer</title>
<style>
  html, body { margin: 0; height: 100%; background: #0e0e10; color: #eee;
               font-family: system-ui, sans-serif; }
  #scene { position: fixed; inset: 0; }
  #hud { position: fixed; top: 12px; left: 12px; padding: 10px 14px;
         background: rgba(0,0,0,0.55); border: 1px solid #333; border-radius: 6px;
         font-size: 13px; line-height: 1.5; min-width: 280px; }
  #hud .row { display: flex; justify-content: space-between; gap: 16px; }
  #hud .lbl { opacity: 0.6; }
  #hud .val { font-variant-numeric: tabular-nums; }
  #thumb-wrap { position: fixed; bottom: 12px; right: 12px;
                border: 1px solid #333; background: #000; }
  #thumb { display: block; width: 480px; height: auto; }
  #thumb-label { padding: 4px 8px; font-size: 11px; background: #181818; }
  #status-dot { display: inline-block; width: 8px; height: 8px;
                border-radius: 50%; background: #888; margin-right: 6px; }
  .ok { background: #44dd55 !important; }
  .bad { background: #dd4444 !important; }
  .clip-flag { display: inline-block; width: 14px; text-align: center;
               border-radius: 3px; padding: 0 2px; margin: 0 1px; font-weight: 600; }
  .clip-flag.on { background: #dd4444; color: white; }
  .clip-flag.off { color: #555; }
</style>

<script type="importmap">
{ "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
</head>
<body>
  <canvas id="scene"></canvas>

  <div id="hud">
    <div class="row"><span><span id="status-dot"></span>state stream</span><span class="val" id="state-text">connecting</span></div>
    <div class="row"><span class="lbl">robot X</span><span class="val" id="robot-x">-</span></div>
    <div class="row"><span class="lbl">robot Y</span><span class="val" id="robot-y">-</span></div>
    <div class="row"><span class="lbl">robot Z</span><span class="val" id="robot-z">-</span></div>
    <div class="row"><span class="lbl">gripper</span><span class="val" id="grip-val">-</span></div>
    <div class="row"><span class="lbl">clipped</span>
      <span>
        <span class="clip-flag off" id="clip-x">X</span>
        <span class="clip-flag off" id="clip-y">Y</span>
        <span class="clip-flag off" id="clip-z">Z</span>
      </span></div>
  </div>

  <div id="thumb-wrap">
    <div id="thumb-label">camera + detection</div>
    <img id="thumb" src="/preview.mjpg">
  </div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Robot workspace (must match shared/retarget.py). Used for the wireframe box.
const WS = { x: [-0.20, +0.20], y: [0.05, 0.30], z: [0.10, 0.35] };

// ----- scene setup -------------------------------------------------------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0e10);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10);
camera.position.set(0.55, 0.45, 0.55);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.18, -0.22);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0x808080, 1.0));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(1, 2, 1);
scene.add(dirLight);

// Floor grid at Y=0 (robot base plane)
const grid = new THREE.GridHelper(1.5, 30, 0x444444, 0x282828);
scene.add(grid);

// Coordinate axes (R=X, G=Y, B=Z) at origin
scene.add(new THREE.AxesHelper(0.12));

// Workspace wireframe box (in three.js coords: Z flipped from robot Z)
{
  const cx = (WS.x[0] + WS.x[1]) / 2;
  const cy = (WS.y[0] + WS.y[1]) / 2;
  const cz = -(WS.z[0] + WS.z[1]) / 2;  // flip
  const sx = WS.x[1] - WS.x[0];
  const sy = WS.y[1] - WS.y[0];
  const sz = WS.z[1] - WS.z[0];
  const geom = new THREE.BoxGeometry(sx, sy, sz);
  const edges = new THREE.EdgesGeometry(geom);
  const lines = new THREE.LineSegments(edges,
      new THREE.LineBasicMaterial({ color: 0x4499ff }));
  lines.position.set(cx, cy, cz);
  scene.add(lines);
}

// Gripper: a base block + two parallel jaws
const gripper = new THREE.Group();
const matBody = new THREE.MeshStandardMaterial({ color: 0xcccccc, metalness: 0.4, roughness: 0.5 });
const matJaw  = new THREE.MeshStandardMaterial({ color: 0x44dd55, metalness: 0.3, roughness: 0.5 });

// "wrist/base" cube
const baseMesh = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.025, 0.025), matBody);
baseMesh.position.set(0, 0.025, 0);
gripper.add(baseMesh);

// two jaws (extending down from the base)
const jawGeom = new THREE.BoxGeometry(0.006, 0.04, 0.018);
const leftJaw = new THREE.Mesh(jawGeom, matJaw);
const rightJaw = new THREE.Mesh(jawGeom, matJaw);
leftJaw.position.set(-0.02, 0, 0);
rightJaw.position.set(+0.02, 0, 0);
gripper.add(leftJaw);
gripper.add(rightJaw);

scene.add(gripper);

// ----- state + interpolation --------------------------------------------
const targetPos = new THREE.Vector3(0, (WS.y[0]+WS.y[1])/2, -(WS.z[0]+WS.z[1])/2);
let targetGripper = 0.5;
let connected = false;
let lastDataTs = 0;

function setStatusUI(text, ok) {
  const dot = document.getElementById('status-dot');
  dot.className = ''; dot.classList.add(ok ? 'ok' : 'bad');
  document.getElementById('state-text').textContent = text;
}

const es = new EventSource('/state');
es.addEventListener('open', () => setStatusUI('connected', true));
es.addEventListener('error', () => setStatusUI('disconnected', false));
es.addEventListener('state', (e) => {
  const d = JSON.parse(e.data);
  lastDataTs = performance.now();
  connected = true;
  if (d.robot) {
    setStatusUI('hand tracking', true);
    targetPos.set(d.robot[0], d.robot[1], -d.robot[2]);  // flip Z to three.js
    targetGripper = d.gripper;
    document.getElementById('robot-x').textContent = d.robot[0].toFixed(3);
    document.getElementById('robot-y').textContent = d.robot[1].toFixed(3);
    document.getElementById('robot-z').textContent = d.robot[2].toFixed(3);
    document.getElementById('grip-val').textContent = d.gripper.toFixed(2)
        + (d.gripper > 0.4 ? '  open' : '  closed');
    ['x','y','z'].forEach((ax,i) => {
      const el = document.getElementById('clip-'+ax);
      el.className = 'clip-flag ' + (d.clipped[i] ? 'on' : 'off');
    });
  } else {
    setStatusUI('no hand', false);
  }
});

// ----- render loop ------------------------------------------------------
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  // smooth toward target (extra display-side easing on top of EMA)
  gripper.position.lerp(targetPos, 0.30);
  // jaws: open 0 = ~4mm half-width; open 1 = ~24mm half-width
  const half = 0.004 + 0.020 * targetGripper;
  leftJaw.position.x = -half;
  rightJaw.position.x = +half;
  // color: green when open, red when closed
  matJaw.color.setHex(targetGripper > 0.4 ? 0x44dd55 : 0xdd4444);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return

        if self.path == "/state":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                last_t = -1.0
                while not _stop.is_set():
                    with _state_lock:
                        s = _latest_state
                    if s.get("t", -1.0) != last_t:
                        last_t = s.get("t", -1.0)
                        payload = json.dumps(s)
                        self.wfile.write(b"event: state\n")
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        try: self.wfile.flush()
                        except Exception: return
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        if self.path == "/preview.mjpg":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _stop.is_set():
                    with _thumb_lock:
                        jpg = _latest_thumb
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
