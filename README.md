# robotics-perception-lab

Hands-on experiments in robotics perception, built around a Luxonis OAK-D depth camera.
Each experiment is self-contained inside `experiments/NN_name/` with its own README and
runnable scripts.

## Hardware

- **Camera:** Luxonis OAK-D (3 sensors: 12MP color IMX378, stereo pair OV9282 + OV9282, 9-DOF BNO086 IMU, on-device Movidius MyriadX inference)
- **Host:** Mac mini (Apple Silicon). Most pipelines also work on Linux/Windows with the same code.

## Install

```bash
# 1. Create / activate a Python 3.10+ environment (conda example)
conda create -n perception python=3.12 -y
conda activate perception

# 2. Install dependencies
pip install -r requirements.txt

# 3. Plug in the OAK-D over USB and verify it enumerates
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

## Experiments

| # | Name | What it teaches |
|---|------|-----------------|
| 01 | [Hand teleoperation](experiments/01_hand_teleop/) | Stereo depth, 2D→3D unprojection, hand tracking with MediaPipe, frame retargeting — the perception side of a teleoperated robot arm. |

More to come. Each experiment is independent — you can clone the repo and run any one of them without touching the others.

## Repository layout

```
.
├── shared/              Reusable helpers (OAK pipeline setup, common math)
├── experiments/NN_name/ One self-contained experiment per folder
└── captures/            Scratch directory for output images / videos (gitignored)
```

## License

MIT.
