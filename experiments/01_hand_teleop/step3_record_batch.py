"""Record a batch of N sessions back-to-back, each with a different pose.

Designed for the analyze-fix loop: you get a single timestamped folder
containing per-pose sub-folders so trends across conditions can be
compared in one analysis pass.

Between sessions the script prints the upcoming pose and counts down so
you can reposition; the live preview at http://localhost:8080 stays the
same URL across all sessions (the server is restarted internally for each).

Usage:
    python step3_record_batch.py
        # runs the default 10-pose schedule

    python step3_record_batch.py --duration 10
        # 10s per session instead of 15s

    python step3_record_batch.py --prep 8
        # 8s countdown between sessions instead of 5s
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from step3_record import run_recording  # noqa: E402


# (folder_suffix, on-screen label, prep instruction shown in terminal)
DEFAULT_POSES = [
    ("01_static_40cm_a",
     "static 40cm A",
     "Hand centered, palm to camera, ~40 cm away. Hold STILL."),
    ("02_static_40cm_b",
     "static 40cm B",
     "Same pose as #1. Repeat - measures session-to-session repeatability."),
    ("03_static_40cm_c",
     "static 40cm C",
     "Same pose as #1 again. Third repeat."),
    ("04_static_near_30cm",
     "static 30cm",
     "Hand centered, ~30 cm from camera. Closer than usual. Hold STILL."),
    ("05_static_far_60cm",
     "static 60cm",
     "Hand centered, ~60 cm from camera. Farther. Hold STILL."),
    ("06_slow_horizontal_sweep",
     "slow horizontal sweep",
     "Slowly sweep hand LEFT-RIGHT-LEFT-RIGHT at ~40 cm. About 1 sweep per 5s."),
    ("07_slow_vertical_sweep",
     "slow vertical sweep",
     "Slowly sweep hand UP-DOWN-UP-DOWN at ~40 cm. About 1 sweep per 5s."),
    ("08_slow_depth_sweep",
     "slow depth sweep",
     "Slowly push/pull hand toward/away from camera, 30 cm <-> 60 cm."),
    ("09_finger_motion",
     "fingers open/close, wrist still",
     "Hold wrist still at ~40 cm. Repeatedly OPEN and CLOSE fingers."),
    ("10_pinch_toggle",
     "pinch toggle",
     "Hold wrist still at ~40 cm. Touch thumb+index together, then release. Repeat."),
]


def countdown(seconds: int, message: str):
    """Print a countdown to stdout (with flush so it actually shows)."""
    for s in range(seconds, 0, -1):
        sys.stdout.write(f"\r  {message}   starting in {s:2d}s ...   ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r  " + " " * 80 + "\r")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=15.0,
                   help="seconds of recording per session (default 15)")
    p.add_argument("--prep", type=int, default=10,
                   help="seconds of countdown between sessions (default 10)")
    p.add_argument("--from-session", type=int, default=1,
                   help="resume from session N (1-indexed). Default 1.")
    p.add_argument("--batch-dir", type=str, default=None,
                   help="resume into an existing batch_* folder (under captures/).")
    args = p.parse_args()

    if args.batch_dir:
        batch_dir = REPO_ROOT / "captures" / args.batch_dir
        if not batch_dir.is_dir():
            raise SystemExit(f"--batch-dir not found: {batch_dir}")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = REPO_ROOT / "captures" / f"batch_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch output dir: {batch_dir}")
    print(f"  {len(DEFAULT_POSES)} sessions, {args.duration:.0f}s each, "
          f"{args.prep}s prep between.")
    print(f"  Total time ~ {len(DEFAULT_POSES) * (args.duration + args.prep + 6):.0f} s "
          f"({(len(DEFAULT_POSES) * (args.duration + args.prep + 6))/60:.1f} min).")
    print()
    print("Live preview during each session: http://localhost:8080")
    print("=" * 70)

    sessions_to_run = DEFAULT_POSES[args.from_session - 1:]
    for i, (suffix, label, instruction) in enumerate(sessions_to_run,
                                                     start=args.from_session):
        print()
        print(f"--- Session {i}/{len(DEFAULT_POSES)} : {label} ---")
        print(f"    {instruction}")
        countdown(args.prep, f"Session {i}/{len(DEFAULT_POSES)}")

        out_dir = batch_dir / f"{i:02d}_{suffix.split('_', 1)[1]}"
        run_recording(args.duration, out_dir,
                      session_label=f"#{i} {label}")

        print(f"  -> {out_dir}")

    print()
    print("=" * 70)
    print(f"BATCH COMPLETE: {batch_dir}")
    print()
    print(f"To analyze, share the folder name:")
    print(f"   {batch_dir.name}")


if __name__ == "__main__":
    main()
