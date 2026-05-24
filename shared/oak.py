"""Reusable OAK-D pipeline helpers.

Each experiment in this repo builds on top of these so the depthai boilerplate
lives in one place. Functions here are intentionally small and composable —
construct a pipeline, grab a frame, read intrinsics. No global state.
"""

from dataclasses import dataclass

import depthai as dai
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics for one camera socket."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def unproject(self, u: float, v: float, z_mm: float) -> np.ndarray:
        """Pixel (u,v) + depth in mm  ->  (X, Y, Z) point in meters, camera frame."""
        z = z_mm / 1000.0
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float32)


def get_intrinsics(
    device: dai.Device,
    socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
    width: int = 1280,
    height: int = 720,
) -> Intrinsics:
    """Read factory calibration for the given camera at the given resolution."""
    calib = device.readCalibration()
    matrix = np.array(calib.getCameraIntrinsics(socket, width, height))
    return Intrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        width=width,
        height=height,
    )
