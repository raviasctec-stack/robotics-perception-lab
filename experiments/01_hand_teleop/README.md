# Experiment 01 — Hand teleoperation (perception side)

Build the entire **camera-side pipeline of a teleoperated robot arm**, using
your hand as the leader. No physical robot needed — a virtual gripper
rendered live in your browser plays the follower.

When/if a real SO-101 arm is added, the only thing that changes is the
consumer of the final `(x, y, z, gripper)` stream. Everything upstream
(camera, hand tracking, smoothing, retargeting) stays the same.

![pipeline diagram](../../docs/01_pipeline.png)
*(add a screenshot at `docs/01_pipeline.png` to embed it here)*

## Why this matters

This is the exact pipeline that [lerobot](https://github.com/huggingface/lerobot),
Tesla Optimus, and Physical Intelligence (π0) use to collect human
demonstrations. Once you have a real arm, the swap from "render to browser" to
"send joints to the arm + record (observation, action) pairs" is a one-file
change — and the recorded data is ready for imitation-learning training.

## The pipeline

```
┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Your hand   │──►│  OAK-D        │──►│  MediaPipe   │──►│   Retarget   │──►│   3D viewer     │
│ (real world) │   │  RGB + depth  │   │  Hands       │   │   palm + pinch│   │   in browser    │
│              │   │  on-camera    │   │  21 landmarks│   │   → robot pose│   │  (three.js)     │
└──────────────┘   └───────────────┘   └──────────────┘   └──────────────┘   └─────────────────┘
   30 Hz             on MyriadX           on Mac CPU         shared/retarget    SSE + MJPEG
```

| Box | Lives in |
|-----|----------|
| OAK pipeline (RGB + tuned stereo depth + post-filters) | `shared/oak.py` |
| Smoothing (EMA, dropout-aware) | `shared/smoothing.py` |
| Retargeting (palm center, pinch ratio, frame transform, workspace map) | `shared/retarget.py` |
| Live runners (steps 1-5) | `step*.py` |

## Quick start

Each step is independently runnable. Start at step 1 and walk down.

```bash
# 1. Live hand skeleton overlaid on the OAK feed
python step1_mediapipe_live.py

# 2. Synced RGB + tuned stereo depth, with wrist-depth readout
python step2_rgb_depth_sync.py

# 3. Unproject wrist (u, v, depth) -> (X, Y, Z) meters in camera frame
python step3_wrist_3d.py

# 4. Hand frame -> robot frame retargeting; (x,y,z,gripper) on overlay
python step4_retarget.py

# 5. 3D virtual gripper in browser, driven live by your hand
python step5_viewer.py
```

All five serve a live page at **<http://localhost:8080>**.

## Step-by-step

| # | Script | What it adds over the previous step |
|---|--------|--------------------------------------|
| **1 still** | `step1_mediapipe_still.py` | Single OAK frame → MediaPipe → save annotated PNG. Sanity check that the hand model loads and detects. |
| **1 live** | `step1_mediapipe_live.py` | Same but as a continuous MJPEG stream with the skeleton drawn in real time. |
| **2** | `step2_rgb_depth_sync.py` | Adds the stereo depth stream, aligned pixel-for-pixel to color. Heavy stereo tuning for close-range hands (HIGH_DETAIL preset, extended disparity, `disparityShift=30` → min ~18 cm, median + speckle + spatial + temporal filters). Live wrist depth in mm with a 5×5 patch median. |
| **3 live** | `step3_wrist_3d.py` | Reads OAK factory intrinsics; pinhole-unprojects `(u, v, depth)` to `(X, Y, Z)` meters in camera frame. Shows raw and EMA-smoothed values side-by-side. |
| **3 record** | `step3_record.py` | Same pipeline, plus writes `recording.mp4`, `telemetry.csv` (per-frame X/Y/Z/depth/FPS, both raw and smoothed), sample PNGs every 2 s, and an auto-computed `summary.txt`. |
| **3 batch** | `step3_record_batch.py` | Drives 10 back-to-back sessions across diverse poses (static at 30/40/60 cm, sweeps, finger motion, pinch) for cross-condition analysis. |
| **3 analyze** | `analyze_batch.py` | Aggregate analyzer over a batch folder: per-session table + jitter + frame-to-frame jump + smoothing improvement. |
| **4** | `step4_retarget.py` | Switches from raw wrist to **palm-center** landmark (avg of LM 0, 5, 17) → much more stable when fingers move. Adds frame transform (camera → robot), per-axis workspace clipping, and scale-invariant pinch ratio for the gripper command. |
| **5** | `step5_viewer.py` | three.js scene rendering the robot workspace + a parallel-jaw gripper that translates and opens/closes live. State streamed over Server-Sent Events; camera thumbnail over MJPEG. |

## Hardware tuning, learned the hard way

These are the numbers behind the scripts. Don't change without re-running the batch analyzer.

| Parameter | Value | Why |
|---|---|---|
| `DEPTH_MIN_MM` | 100 | Hand-workspace floor; threshold filter rejects below this |
| `DEPTH_MAX_MM` | 1500 | Hand-workspace ceiling; everything beyond becomes invalid for cleaner heatmaps |
| `DISPARITY_SHIFT` | 30 | Shifts stereo search window to handle close-range; pushes min depth from ~70 cm (default) down to ~18 cm. Tradeoff: max depth drops from ∞ to ~2 m |
| `setSubpixel(False)` | — | Subpixel improves precision but blurs edges; off keeps finger outlines crisp |
| `setExtendedDisparity(True)` | — | Doubles disparity range; still need disparity shift on top for sub-30 cm |
| Stereo preset | `HIGH_DETAIL` | Best for finger-detail preservation; tried `ROBOTICS` first, this is better here |
| Median + speckle + spatial + temporal filters | all on | Sequentially clean up the depth map: median removes salt-pepper, speckle removes small blobs, spatial fills holes, temporal stabilizes across frames |
| `EMA_ALPHA` | 0.30 | ~3-sample window; gives ~20% per-frame noise reduction without visible lag |
| MediaPipe `num_hands` | 1 | One hand → one robot arm; bump to 2 for bimanual setups |

## Performance summary (from the 10-session batch)

| Condition | Detection | Depth at wrist | Jitter (raw, mm) | Notes |
|-----------|-----------|----------------|-------------------|-------|
| Static 40 cm | 88-96 % | 54-74 % | X≈18, Y≈22, Z≈28 | Sweet spot |
| Static 30 cm | 87 % | 27 % | X=24, Y=35, Z=44 | Below disparity-shift's safe zone |
| Static 60 cm | 96 % | 13 % | X=52, Y=26, Z=37 | Hand too small for stereo on skin |
| Sweep / motion | 96-97 % | 50-96 % | (motion, not jitter) | Pipeline keeps up |
| Pinch toggle | 83 % | 62 % | (12 mm/frame at wrist) | Why step 4 uses **palm center** instead of raw wrist |
| FPS | always | always | — | Steady **~15 FPS** end-to-end on Apple M4 |

**User-facing implication:** sit so your hand naturally falls in the **40–55 cm zone**. Closer or farther produces measurably more noise; the pipeline still works, just with reduced precision.

## Concepts you'll have learned by the end

- **Pinhole camera model** — `(u, v, depth) → (X, Y, Z)` using `fx, fy, cx, cy`
- **Stereo depth** — disparity → distance; why textureless surfaces / skin / specular highlights produce holes; min/max range physics
- **Camera intrinsics & calibration** — factory calibration on the OAK; how to read and use it
- **Coordinate frames & transforms** — camera frame ≠ robot frame; how to convert; why convention matters
- **Workspace mapping** — bounded linear transforms with clipping; why a robot's "reachable box" matters
- **EMA smoothing on vector signals** — dropout-aware; trade-off between latency and noise reduction
- **Scale-invariant gestures** — pinch distance / hand size as a control signal that works at any distance
- **Server-Sent Events for live streaming** — when you don't need full WebSocket bidirectionality
- **Per-frame vs aggregate jitter** — `std` over a window vs frame-to-frame jump; different things to measure

## Limitations & known issues

- **~15 FPS, not 30.** MediaPipe inference + JPEG encoding + stereo are serial. Optimizing requires either MediaPipe at lower resolution or threading.
- **Depth coverage on skin is ~15-30 % of the frame.** Passive stereo + textureless surfaces is a hard limit. Active IR projectors (OAK-D Pro, RealSense D435i) solve this; OAK-D doesn't have one.
- **Workspace bounds are static defaults.** Tune `CAM_WORKSPACE` and `ROBOT_WORKSPACE` in `shared/retarget.py` for your physical setup.
- **macOS only tested.** Linux and Windows should work since everything is Python + WebKit; not validated.

## Files

```
experiments/01_hand_teleop/
├── README.md                       this file
├── step1_mediapipe_still.py        single-frame sanity check
├── step1_mediapipe_live.py         live MediaPipe overlay
├── step2_rgb_depth_sync.py         RGB + tuned stereo depth + wrist depth
├── step3_wrist_3d.py               2D + depth -> 3D meters (live)
├── step3_record.py                 record a session (MP4 + CSV + summary)
├── step3_record_batch.py           run 10 sessions for cross-condition analysis
├── analyze_batch.py                aggregate analyzer for a batch_*/ folder
├── step4_retarget.py               hand frame -> robot frame retargeting (live)
└── step5_viewer.py                 3D virtual gripper in browser, SSE-driven
```
