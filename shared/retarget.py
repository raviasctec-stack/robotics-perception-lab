"""Hand-to-robot retargeting math.

Takes a MediaPipe-detected hand + a depth frame + camera intrinsics, and
produces a target pose in the robot's base frame plus a gripper value.

Pipeline:
  1. palm_center_pixel  : average of LM[0,5,17]  (stable to finger motion)
  2. depth_at_pixel     : patch median around the palm
  3. unproject          : (u, v, depth) -> (X, Y, Z) meters in camera frame
  4. camera_to_robot    : axis convention swap (Y flips so up is positive)
  5. map_workspace      : linear scale + clip from hand box to robot box
  6. gripper_value      : scale-invariant pinch ratio -> [0, 1] open value

Smoothing is applied externally (caller passes a pre-smoothed XYZ if desired).

Coordinate conventions:
    OAK camera frame : +X right, +Y down, +Z forward (away from lens)
    Robot base frame : +X right, +Y up,   +Z forward (into workspace)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# -- workspace definitions ---------------------------------------------------
#
# CAM_WORKSPACE is the rectangular box of camera-frame positions we'll map
# from. Tuned for a user sitting ~40-50 cm in front of the OAK with normal
# arm reach. Tweak if your setup is different.
#
# ROBOT_WORKSPACE is the box of valid robot-base-frame targets. The numbers
# below are loose defaults for an SO-101-class arm; in a real deployment you
# would constrain to the arm's actual reachable hemisphere.
CAM_WORKSPACE = {
    "x_min": -0.30, "x_max": +0.30,   # 60 cm horizontal range
    "y_min": -0.25, "y_max": +0.15,   # 40 cm vertical range
    "z_min": 0.30,  "z_max": 0.70,    # 40 cm depth range
}

ROBOT_WORKSPACE = {
    "x_min": -0.20, "x_max": +0.20,   # ~40 cm wide
    "y_min": 0.05,  "y_max": 0.30,    # ~25 cm tall
    "z_min": 0.10,  "z_max": 0.35,    # ~25 cm deep
}


# -- landmark indices --------------------------------------------------------
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
THUMB_TIP = 4
INDEX_TIP = 8

PALM_LANDMARKS = (WRIST, INDEX_MCP, PINKY_MCP)


GRIPPER_PINCH_RATIO_MIN = 0.20   # below this, gripper is fully closed
GRIPPER_PINCH_RATIO_MAX = 1.50   # above this, gripper is fully open


@dataclass
class RetargetResult:
    palm_pixel: tuple[int, int]
    palm_depth_mm: int
    xyz_cam: np.ndarray            # (3,) meters in camera frame, RAW
    xyz_cam_smooth: np.ndarray | None  # (3,) meters in camera frame, smoothed
    xyz_robot: np.ndarray          # (3,) meters in robot frame, post-map+clip
    gripper: float                 # 0 = closed, 1 = open
    clipped: tuple[bool, bool, bool]  # was each robot axis at a workspace edge


def palm_center_pixel(landmarks, W: int, H: int) -> tuple[int, int]:
    """Average pixel position of wrist + index-MCP + pinky-MCP."""
    u = sum(landmarks[i].x for i in PALM_LANDMARKS) * W / len(PALM_LANDMARKS)
    v = sum(landmarks[i].y for i in PALM_LANDMARKS) * H / len(PALM_LANDMARKS)
    return int(u), int(v)


def pinch_ratio(landmarks, W: int, H: int) -> float:
    """Distance(thumb_tip, index_tip) / distance(wrist, middle_MCP).

    Dividing by hand size makes this scale-invariant: a pinch reads the same
    whether the hand is close to or far from the camera.
    """
    def px(i):
        return np.array([landmarks[i].x * W, landmarks[i].y * H])
    pinch = np.linalg.norm(px(THUMB_TIP) - px(INDEX_TIP))
    hand_scale = np.linalg.norm(px(WRIST) - px(MIDDLE_MCP))
    if hand_scale < 1.0:
        return 0.5  # degenerate; return neutral
    return float(pinch / hand_scale)


def gripper_value(landmarks, W: int, H: int) -> float:
    """Map pinch ratio -> [0, 1] open value with linear clamp."""
    r = pinch_ratio(landmarks, W, H)
    t = (r - GRIPPER_PINCH_RATIO_MIN) / (GRIPPER_PINCH_RATIO_MAX - GRIPPER_PINCH_RATIO_MIN)
    return max(0.0, min(1.0, t))


def camera_to_robot(xyz_cam: np.ndarray) -> np.ndarray:
    """Convert OAK camera frame -> robot base frame.

    Just flip Y so up is +Y. X and Z (depth) stay as-is.
    """
    return np.array([xyz_cam[0], -xyz_cam[1], xyz_cam[2]], dtype=np.float64)


def _lerp_clip(v, src_min, src_max, dst_min, dst_max):
    t = (v - src_min) / (src_max - src_min)
    clipped = t < 0.0 or t > 1.0
    t = max(0.0, min(1.0, t))
    return dst_min + t * (dst_max - dst_min), clipped


def map_workspace(xyz_robot_unclipped: np.ndarray
                  ) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
    """Linearly map hand-workspace coords into robot-workspace coords, with
    per-axis clipping. Returns (mapped_xyz, (clipped_x, clipped_y, clipped_z))."""
    cam = CAM_WORKSPACE
    rob = ROBOT_WORKSPACE

    # Camera Y was flipped to robot Y in camera_to_robot; the hand-frame
    # vertical range becomes [-y_max, -y_min] in robot frame.
    x, cx = _lerp_clip(xyz_robot_unclipped[0], cam["x_min"], cam["x_max"],
                       rob["x_min"], rob["x_max"])
    y, cy = _lerp_clip(xyz_robot_unclipped[1], -cam["y_max"], -cam["y_min"],
                       rob["y_min"], rob["y_max"])
    z, cz = _lerp_clip(xyz_robot_unclipped[2], cam["z_min"], cam["z_max"],
                       rob["z_min"], rob["z_max"])
    return np.array([x, y, z], dtype=np.float64), (cx, cy, cz)


def sample_depth_patch(depth_mm: np.ndarray, u: int, v: int, half: int = 2) -> int:
    """Median of nonzero depth values in a small patch around (u, v)."""
    h, w = depth_mm.shape
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[patch > 0]
    return int(np.median(valid)) if valid.size else 0
