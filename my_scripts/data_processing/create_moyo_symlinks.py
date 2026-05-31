"""
Create a symlink tree at ~/data/mocap/data/moyo/ that maps MOYO's native
layout into the structure expected by mocap_mainloader, AND writes converted
camera parameter JSON files (format mismatch prevents symlinking those).

  mocap/data/moyo/{dtype}/{sequence}/
    videos/{camera}/result.mp4         → moyo/data/mmpose/{model}/{session}/{dtype}/{sequence}/{camera}/result.mp4
    mmpose/{model}/{camera}/result.json → moyo/data/mmpose/{model}/{session}/{dtype}/{sequence}/{camera}/result.json
    mhr/result.npz                      → moyo/data/mhr/{dtype}/{sequence}.npz
    camera_parameters/{camera}/result.json  (written, not symlinked — format conversion required)

Camera params conversion:
  Source: ~/data/moyo/data/cameras/{YYYYMMDD}/*_PROCESSED_CAMERA_PARAMS/cameras_param.json
    flat dict per camera: focal (px, full-res), position (mm), princpt, rotation (3x3)
  Target: fit3d format with extrinsics.R / extrinsics.T (metres) / intrinsics_wo_distortion.f,c
    intrinsics scaled by 0.5 to match stored video/mmpose resolution

Only sequences with an MHR .npz are linked (others lack ground truth).

Usage:
    python create_moyo_symlinks.py [--dry-run] [--force]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

MOYO_DATA    = Path.home() / "data/moyo/data"
MOCAP_MOYO   = Path.home() / "data/mocap/data/moyo"
MMPOSE_MODEL = "rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122"
INTR_SCALE   = 0.5   # MOYO videos/mmpose annotations are at 0.5x calibration resolution

SESSION_CAM_FILES = {
    "220923": MOYO_DATA / "cameras/20220923/220923_Afternoon_PROCESSED_CAMERA_PARAMS/cameras_param.json",
    "220926": MOYO_DATA / "cameras/20220926/220926_Morning_PROCESSED_CAMERA_PARAMS/cameras_param.json",
    "221004": MOYO_DATA / "cameras/20221004/Morning_PROCESSED_CAMERA_PARAMS/cameras_param.json",
}


def symlink(src: Path, dst: Path, dry_run: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [dry symlink] {dst} → {src}")
    else:
        os.symlink(src, dst)


def write_json(path: Path, data: dict, dry_run: bool, force: bool) -> None:
    if path.exists() and not force:
        return
    if dry_run:
        print(f"  [dry camjson] {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def moyo_cam_to_fit3d(cam: dict) -> dict:
    R      = np.array(cam["rotation"], dtype=np.float64)
    pos_m  = np.array(cam["position"], dtype=np.float64) / 1000.0   # mm → m
    T      = pos_m.reshape(1, 3)
    focal  = float(cam["focal"])  * INTR_SCALE
    cx, cy = float(cam["princpt"][0]) * INTR_SCALE, float(cam["princpt"][1]) * INTR_SCALE
    return {
        "extrinsics": {"R": R.tolist(), "T": T.tolist()},
        "intrinsics_wo_distortion": {"f": [focal, focal], "c": [cx, cy]},
    }


def main(dry_run: bool, force: bool) -> None:
    # ── Load + convert session camera params ──────────────────────────────────
    session_cams: dict[str, dict] = {}
    for date_prefix, cam_file in SESSION_CAM_FILES.items():
        raw = json.load(open(cam_file))
        session_cams[date_prefix] = {
            f"YOGI_Cam_{int(k.split('_')[1]):02d}": moyo_cam_to_fit3d(v)
            for k, v in raw.items()
        }
        print(f"Loaded {cam_file.name} ({len(raw)} cameras) for prefix {date_prefix}")

    # ── Create symlinks + camera param files ──────────────────────────────────
    mmpose_root = MOYO_DATA / "mmpose" / MMPOSE_MODEL
    mhr_base    = MOYO_DATA / "mhr"

    total_seqs = 0
    total_links = 0
    total_cam_files = 0
    missing_prefix = []

    for dtype in ("train", "val"):
        mhr_seqs = {p.stem for p in (mhr_base / dtype).glob("*.npz")}

        for session_dir in sorted(mmpose_root.iterdir()):
            dtype_dir = session_dir / dtype
            if not dtype_dir.exists():
                continue

            for seq_dir in sorted(dtype_dir.iterdir()):
                seq = seq_dir.name

                if seq not in mhr_seqs:
                    continue

                total_seqs += 1
                mhr_src  = mhr_base / dtype / f"{seq}.npz"
                seq_dst  = MOCAP_MOYO / dtype / seq

                for cam_dir in sorted(seq_dir.iterdir()):
                    cam = cam_dir.name
                    mp4_src  = cam_dir / "result.mp4"
                    json_src = cam_dir / "result.json"

                    if not mp4_src.exists() or not json_src.exists():
                        print(f"[warn] missing files in {cam_dir}")
                        continue

                    symlink(mp4_src,  seq_dst / "videos" / cam / "result.mp4",                 dry_run)
                    symlink(json_src, seq_dst / "mmpose" / MMPOSE_MODEL / cam / "result.json", dry_run)
                    total_links += 2

                symlink(mhr_src, seq_dst / "mhr" / "result.npz", dry_run)
                total_links += 1

                # camera params — written (not symlinked) due to format conversion
                date_prefix = seq[:6]
                if date_prefix not in session_cams:
                    if date_prefix not in missing_prefix:
                        print(f"[warn] no camera file for prefix {date_prefix} (seq: {seq})")
                        missing_prefix.append(date_prefix)
                else:
                    for cam_name, cam_data in session_cams[date_prefix].items():
                        dst = seq_dst / "camera_parameters" / cam_name / "result.json"
                        write_json(dst, cam_data, dry_run, force)
                        total_cam_files += 1

    mode = "DRY RUN" if dry_run else "DONE"
    print(f"\n[{mode}] {total_seqs} sequences, {total_links} symlinks, {total_cam_files} camera param files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--force",   action="store_true", help="Overwrite existing camera param files")
    args = parser.parse_args()
    main(args.dry_run, args.force)
