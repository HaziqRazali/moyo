"""
Convert MOYO session-level camera parameters to per-sequence per-camera JSON files
in the format expected by mocap_mainloader / read_cam_params().

MOYO source:
    /data/haziq/moyo/data/cameras/{YYYYMMDD}/*_PROCESSED_CAMERA_PARAMS/cameras_param.json
    Keys per camera: focal, position (mm, world coords), princpt, rotation (world-to-cam R)

Target format (fit3d style), written to:
    /data/haziq/mocap/data/moyo/{split}/{seq}/camera_parameters/{cam}/result.json
    {
        "extrinsics": {
            "R": [[3x3]],       # world-to-camera rotation
            "T": [[tx, ty, tz]] # translation in metres, shape [1,3]
        },
        "intrinsics_wo_distortion": {
            "f": [fx, fy],
            "c": [cx, cy]
        }
    }

Camera name mapping: cam_N → YOGI_Cam_{N:02d}

Usage:
    python create_moyo_cam_params.py [--dry-run]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

MOYO_CAM_ROOT = Path.home() / "data/moyo/data/cameras"
MOCAP_MOYO    = Path.home() / "data/mocap/data/moyo"
INTR_SCALE    = 0.5   # MOYO videos/mmpose annotations are at 0.5x calibration resolution

# Map first 6 digits of sequence name (YYMMDD) → camera JSON path
# Sequences like 220923_... → date folder 20220923
SESSION_CAM_FILES = {
    "220923": MOYO_CAM_ROOT / "20220923" / "220923_Afternoon_PROCESSED_CAMERA_PARAMS" / "cameras_param.json",
    "220926": MOYO_CAM_ROOT / "20220926" / "220926_Morning_PROCESSED_CAMERA_PARAMS"   / "cameras_param.json",
    "221004": MOYO_CAM_ROOT / "20221004" / "Morning_PROCESSED_CAMERA_PARAMS"          / "cameras_param.json",
}


def moyo_cam_to_fit3d(cam: dict) -> dict:
    """Convert a single MOYO camera dict to fit3d format."""
    R = np.array(cam["rotation"], dtype=np.float64)          # (3,3) world-to-cam
    pos_mm = np.array(cam["position"], dtype=np.float64)     # (3,) camera centre, mm
    pos_m  = pos_mm / 1000.0                                  # → metres
    # project_world_to_pixel convention: X_cam = R @ (X_world - T)  →  T = camera centre (C)
    # Do NOT store -R @ pos_m (translation vector); store pos_m directly.
    T = pos_m.reshape(1, 3)                                   # (1,3) camera centre in world, metres

    # Scale intrinsics to match stored video/mmpose resolution (0.5x of calibration)
    focal = float(cam["focal"]) * INTR_SCALE
    cx, cy = float(cam["princpt"][0]) * INTR_SCALE, float(cam["princpt"][1]) * INTR_SCALE

    return {
        "extrinsics": {
            "R": R.tolist(),
            "T": T.tolist(),
        },
        "intrinsics_wo_distortion": {
            "f": [focal, focal],
            "c": [cx, cy],
        },
    }


def write_json(path: Path, data: dict, dry_run: bool, force: bool = False) -> None:
    if path.exists() and not force:
        return
    if dry_run:
        print(f"  [dry] {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def main(dry_run: bool, force: bool) -> None:
    # Pre-load all session camera dicts
    session_cams: dict[str, dict] = {}
    for date_prefix, cam_file in SESSION_CAM_FILES.items():
        raw = json.load(open(cam_file))
        # Convert each camera to fit3d format
        session_cams[date_prefix] = {
            f"YOGI_Cam_{int(k.split('_')[1]):02d}": moyo_cam_to_fit3d(v)
            for k, v in raw.items()
        }
        print(f"Loaded {cam_file.name} ({len(raw)} cameras) for prefix {date_prefix}")

    # Discover sequences from the MOYO mmpose source tree (independent of symlink tree)
    mmpose_root = MOYO_CAM_ROOT.parent / "mmpose"
    seen_seqs: dict[str, str] = {}   # seq → split  (dedup across sessions)

    for model_dir in mmpose_root.iterdir():
        for session_dir in model_dir.iterdir():
            for split in ("train", "val"):
                split_dir = session_dir / split
                if not split_dir.exists():
                    continue
                for seq_dir in sorted(split_dir.iterdir()):
                    if seq_dir.name not in seen_seqs:
                        seen_seqs[seq_dir.name] = split

    total_written = 0
    missing_prefix = []

    for seq, split in sorted(seen_seqs.items()):
        date_prefix = seq[:6]  # e.g. "220923"

        if date_prefix not in session_cams:
            if date_prefix not in missing_prefix:
                print(f"[warn] no camera file for prefix {date_prefix} (seq: {seq})")
                missing_prefix.append(date_prefix)
            continue

        cams = session_cams[date_prefix]

        # Write one result.json per camera under camera_parameters/
        for cam_name, cam_data in cams.items():
            dst = MOCAP_MOYO / split / seq / "camera_parameters" / cam_name / "result.json"
            write_json(dst, cam_data, dry_run, force)
            total_written += 1

    mode = "DRY RUN" if dry_run else "DONE"
    print(f"\n[{mode}] {total_written} camera parameter files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    main(args.dry_run, args.force)
