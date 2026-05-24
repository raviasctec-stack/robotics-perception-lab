"""Step 1: capture one OAK frame, run MediaPipe Hands, save an annotated PNG.

Goal: confirm MediaPipe finds a hand and returns 21 landmarks. No 3D yet,
no depth yet — pure 2D detection sanity check before adding complexity.

Usage:
    python step1_mediapipe_still.py
    # Then open: ../../captures/step1_hand_landmarks.png
"""

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
CAPTURES = REPO_ROOT / "captures"
OUT = CAPTURES / "step1_hand_landmarks.png"

WARMUP_FRAMES = 20
WIDTH, HEIGHT = 1280, 720

# MediaPipe hand topology: pairs of landmark indices to connect with a line
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                 # palm closure
]


def capture_one_frame() -> np.ndarray:
    """Open OAK, discard warmup frames, return the next BGR frame."""
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        out = cam.requestOutput(size=(WIDTH, HEIGHT), type=dai.ImgFrame.Type.NV12)
        q = out.createOutputQueue()
        pipeline.start()

        img = None
        for _ in range(WARMUP_FRAMES + 1):
            img = q.get().getCvFrame()
        return img


def draw_landmarks(bgr, landmarks):
    """Draw 21 landmarks + skeleton connections on a BGR image (in place)."""
    h, w = bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(bgr, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(bgr, str(i), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    if not MODEL.exists():
        raise SystemExit(
            f"Missing model file: {MODEL}\n"
            "Download with:\n"
            "  curl -L -o models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )
    CAPTURES.mkdir(parents=True, exist_ok=True)

    print("Capturing frame from OAK...")
    frame = capture_one_frame()
    print(f"  frame shape: {frame.shape}")

    print("Running MediaPipe Hands...")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    with HandLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)

    if not result.hand_landmarks:
        print("  NO HAND DETECTED. Hold your hand in front of the camera and rerun.")
        cv2.imwrite(str(OUT), frame)
        print(f"Saved raw frame (no overlay): {OUT}")
        return

    landmarks = result.hand_landmarks[0]
    handedness = result.handedness[0][0].category_name
    print(f"  detected {handedness} hand with {len(landmarks)} landmarks")
    for i, lm in enumerate(landmarks):
        u, v = lm.x * WIDTH, lm.y * HEIGHT
        print(f"    LM[{i:2d}]  pixel=({u:7.1f},{v:7.1f})  rel_z={lm.z:+.3f}")

    draw_landmarks(frame, landmarks)
    cv2.imwrite(str(OUT), frame)
    print(f"Saved annotated image: {OUT}")


if __name__ == "__main__":
    main()
