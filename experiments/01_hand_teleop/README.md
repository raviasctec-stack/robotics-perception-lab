# Experiment 01 — Hand teleoperation (perception side)

Build the full camera-side pipeline of a teleoperated robot arm, using your hand
as the leader. No physical robot needed — a virtual gripper rendered in the
browser plays the follower.

## Why this matters

This is the *exact* perception pipeline that lerobot, Tesla Optimus data
collection, and Physical Intelligence's π0 dataset collection use to gather
human demonstrations for training imitation-learning policies. Once you have
a real arm, the renderer is swapped for a driver and nothing else changes.

```
OAK RGB+depth ──► MediaPipe Hands ──► 2D→3D unproject ──► retarget ──► viewer (or real arm)
```

## Pipeline stages

| Stage | What it does | File |
|-------|--------------|------|
| 1 | Run MediaPipe Hands on a single OAK frame; verify 21 landmarks come back | `step1_mediapipe_still.py` |
| 2 | Stream synchronized RGB + depth from the OAK | `step2_rgb_depth_sync.py` |
| 3 | Unproject the wrist pixel + depth → (X,Y,Z) meters in camera frame | `step3_wrist_3d.py` |
| 4 | Retarget hand pose → robot end-effector pose (frame transform + workspace scaling + gripper) | `step4_retarget.py` |
| 5 | Render a virtual gripper in the browser at the target pose | `step5_viewer.py` |

Each step is independently runnable. Start with `step1_*` and work down.

## Run

```bash
cd experiments/01_hand_teleop
python step1_mediapipe_still.py
```
