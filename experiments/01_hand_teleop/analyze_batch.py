"""Aggregate analyzer for a batch of recording sessions.

Reads every session_*/telemetry.csv under captures/batch_TIMESTAMP/ and
produces a single report comparing jitter, dropout, detection rate, and FPS
across all sessions.

For static poses, jitter is meaningful (lower std = better). For sweep
poses, jitter is dominated by real motion; what matters there is dropout
rate and FPS.

Usage:
    python analyze_batch.py batch_20260524_223XXX
    python analyze_batch.py /absolute/path/to/batch_dir
"""

import argparse
import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_session(session_dir: Path) -> dict:
    csv_path = session_dir / "telemetry.csv"
    if not csv_path.exists():
        return {"name": session_dir.name, "error": "no telemetry.csv"}

    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)
    if n == 0:
        return {"name": session_dir.name, "error": "empty CSV"}

    detected = [r for r in rows if r["hand_detected"] == "1"]
    with_depth = [r for r in detected if int(r["depth_mm"]) > 0]

    raw = np.array([[float(r["X_m"]), float(r["Y_m"]), float(r["Z_m"])]
                    for r in with_depth]) if with_depth else np.zeros((0, 3))
    smooth = np.array([[float(r.get("X_smooth_m", "nan")),
                        float(r.get("Y_smooth_m", "nan")),
                        float(r.get("Z_smooth_m", "nan"))]
                       for r in with_depth]) if with_depth else np.zeros((0, 3))
    # smooth may have NaNs from filter warmup; drop them
    smooth = smooth[~np.isnan(smooth).any(axis=1)] if smooth.size else smooth

    duration_s = float(rows[-1]["t_s"]) if rows else 0.0
    depth_pcts = np.array([float(r["depth_valid_pct"]) for r in rows])
    fps_vals = np.array([float(r["fps_inst"]) for r in rows
                         if r["fps_inst"] != "nan" and r["fps_inst"] != ""])

    s = {
        "name": session_dir.name,
        "frames": n,
        "duration_s": duration_s,
        "fps_avg": n / max(duration_s, 1e-6),
        "fps_median": float(np.median(fps_vals)) if fps_vals.size else float("nan"),
        "detection_pct": 100 * len(detected) / n,
        "with_depth_pct": 100 * len(with_depth) / n,
        "dropouts": len(detected) - len(with_depth),
        "depth_coverage_mean": float(depth_pcts.mean()),
        "depth_coverage_median": float(np.median(depth_pcts)),
        "n_raw": len(raw),
        "raw_median_xyz": np.median(raw, axis=0).tolist() if len(raw) else None,
        "raw_std_mm": (1000 * np.std(raw, axis=0)).tolist() if len(raw) >= 5 else None,
        "raw_p95_mm": (1000 * np.percentile(np.abs(raw - np.median(raw, axis=0)),
                                            95, axis=0)).tolist()
                      if len(raw) >= 5 else None,
        "n_smooth": len(smooth),
        "smooth_std_mm": (1000 * np.std(smooth, axis=0)).tolist() if len(smooth) >= 5 else None,
        "smooth_p95_mm": (1000 * np.percentile(np.abs(smooth - np.median(smooth, axis=0)),
                                               95, axis=0)).tolist()
                         if len(smooth) >= 5 else None,
        # Frame-to-frame jump magnitude (a different jitter measure that doesn't
        # assume the median is the truth)
        "raw_d_frame_mm": float(1000 * np.median(np.linalg.norm(np.diff(raw, axis=0), axis=1)))
                          if len(raw) >= 2 else float("nan"),
        "smooth_d_frame_mm": float(1000 * np.median(np.linalg.norm(np.diff(smooth, axis=0), axis=1)))
                             if len(smooth) >= 2 else float("nan"),
    }
    return s


STATIC_SESSIONS = {1, 2, 3, 4, 5}  # 1-indexed positions where stillness is expected
MOVING_SESSIONS = {6, 7, 8, 9, 10}


def fmt_xyz(v, fmt="{:7.1f}"):
    if v is None or (isinstance(v, list) and any(x != x for x in v)):
        return "    -        -        -"
    return f"X={fmt.format(v[0])}  Y={fmt.format(v[1])}  Z={fmt.format(v[2])}"


def print_report(batch_dir: Path, sessions: list[dict]):
    print(f"\n=== batch analysis : {batch_dir.name} ===")
    print(f"sessions : {len(sessions)}")
    print()

    # FPS & detection table
    print(f"{'#':>2} {'session':<32} {'fps':>5} {'det%':>5} {'+dep%':>6} "
          f"{'drop':>5} {'cov%':>5}")
    print("-" * 70)
    for s in sessions:
        if "error" in s:
            print(f"   {s['name']:<32}  ERROR: {s['error']}")
            continue
        print(f"   {s['name']:<32} {s['fps_avg']:5.1f} "
              f"{s['detection_pct']:5.1f} {s['with_depth_pct']:6.1f} "
              f"{s['dropouts']:5d} {s['depth_coverage_mean']:5.1f}")

    print()
    print("=== jitter analysis (static sessions only) ===")
    print(f"{'#':>2} {'session':<32}  raw std (mm)         smooth std (mm)")
    print("-" * 85)
    for i, s in enumerate(sessions, 1):
        if i not in STATIC_SESSIONS or "error" in s:
            continue
        raw_s = fmt_xyz(s["raw_std_mm"])
        sm_s = fmt_xyz(s["smooth_std_mm"])
        print(f"   {s['name']:<32}  {raw_s}    {sm_s}")

    print()
    print("=== frame-to-frame jump (median mm/frame -- a per-frame noise metric) ===")
    print(f"{'#':>2} {'session':<32}  raw         smooth     ratio")
    print("-" * 70)
    for i, s in enumerate(sessions, 1):
        if "error" in s:
            continue
        r = s["raw_d_frame_mm"]; sm = s["smooth_d_frame_mm"]
        ratio = sm / r if r and r > 0 else float("nan")
        print(f"   {s['name']:<32}  {r:7.2f}    {sm:7.2f}    {ratio:5.2f}x")

    print()
    print("=== aggregates ===")
    static = [s for i, s in enumerate(sessions, 1) if i in STATIC_SESSIONS and "error" not in s]
    moving = [s for i, s in enumerate(sessions, 1) if i in MOVING_SESSIONS and "error" not in s]
    if static:
        avg_raw_std = np.mean([s["raw_std_mm"] for s in static if s["raw_std_mm"]], axis=0)
        avg_sm_std = np.mean([s["smooth_std_mm"] for s in static if s["smooth_std_mm"]], axis=0)
        print(f"static avg fps          : {np.mean([s['fps_avg'] for s in static]):.1f}")
        print(f"static avg detection    : {np.mean([s['detection_pct'] for s in static]):.1f}%")
        print(f"static avg +depth       : {np.mean([s['with_depth_pct'] for s in static]):.1f}%")
        print(f"static avg raw std (mm) : {fmt_xyz(avg_raw_std.tolist())}")
        print(f"static avg smooth (mm)  : {fmt_xyz(avg_sm_std.tolist())}")
        if avg_raw_std is not None and avg_sm_std is not None:
            improvement = (1 - avg_sm_std / avg_raw_std) * 100
            print(f"smoothing improvement   : {fmt_xyz(improvement.tolist(), '{:5.1f}%')}")
    if moving:
        print()
        print(f"moving avg fps          : {np.mean([s['fps_avg'] for s in moving]):.1f}")
        print(f"moving avg detection    : {np.mean([s['detection_pct'] for s in moving]):.1f}%")
        print(f"moving avg +depth       : {np.mean([s['with_depth_pct'] for s in moving]):.1f}%")
        print(f"  (jitter not meaningful here -- look at dropout & FPS)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", help="batch folder (name under captures/ or absolute path)")
    args = p.parse_args()

    batch_dir = Path(args.batch)
    if not batch_dir.is_absolute():
        batch_dir = REPO_ROOT / "captures" / batch_dir
    if not batch_dir.is_dir():
        raise SystemExit(f"Not a directory: {batch_dir}")

    session_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
    if not session_dirs:
        raise SystemExit(f"No session subfolders in {batch_dir}")

    sessions = [load_session(d) for d in session_dirs]
    print_report(batch_dir, sessions)


if __name__ == "__main__":
    main()
