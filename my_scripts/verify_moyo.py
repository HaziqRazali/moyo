"""
MOYO dataset cross-validation script.
Checks that images, mosh, cameras, YOGI_2, pressure, vicon are aligned.
"""
import os
import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/data/haziq/moyo")
MOYO = ROOT / "MOYO"

# ─────────────────────────────────────────────
# 1. Collect sequence names from each modality
# ─────────────────────────────────────────────
def get_image_seqs():
    seqs = {}  # seq_name -> (split, set of frame indices per camera)
    for split_dir in MOYO.iterdir():
        images_root = split_dir / "images"
        if not images_root.exists():
            continue
        for tv in ("train", "val"):
            tv_dir = images_root / tv
            if not tv_dir.exists():
                continue
            for seq in tv_dir.iterdir():
                if not seq.is_dir():
                    continue
                cams = {}
                for cam in seq.iterdir():
                    if cam.is_dir() and cam.name.startswith("YOGI_Cam"):
                        frames = sorted(cam.glob("*.jpg"))
                        cams[cam.name] = len(frames)
                seqs[seq.name] = {"tv": tv, "cameras": cams, "split": split_dir.name}
    return seqs

def get_mosh_seqs():
    seqs = {}
    for tv in ("train", "val"):
        for pkl in (ROOT / "mosh" / tv).glob("*_stageii.pkl"):
            name = pkl.stem.replace("_stageii", "")
            seqs[name] = {"tv": tv, "path": pkl}
    return seqs

def get_mosh_smpl_seqs():
    seqs = {}
    for tv in ("train", "val"):
        for pkl in (ROOT / "mosh_smpl" / tv).glob("*_stageii_smpl.pkl"):
            name = pkl.stem.replace("_stageii_smpl", "")
            seqs[name] = {"tv": tv, "path": pkl}
    return seqs

def get_yogi2_seqs():
    seqs = {}
    for tv in ("train", "val"):
        for npz in (ROOT / "YOGI_2_latest_smplx_neutral" / tv).glob("*.npz"):
            name = npz.stem
            seqs[name] = {"tv": tv, "path": npz}
    return seqs

def get_pressure_seqs():
    seqs = set()
    for tv in ("train", "val"):
        pdir = ROOT / "pressure" / tv
        for subdir in ("c3d", "pressure_mat_c3d", "single_csv"):
            d = pdir / subdir
            if d.exists():
                for f in d.iterdir():
                    seqs.add(f.stem.split(".")[0])
    return seqs

def get_vicon_seqs():
    seqs = set()
    for tv in ("train", "val"):
        vdir = ROOT / "vicon" / tv / "c3d"
        if vdir.exists():
            for f in vdir.iterdir():
                seqs.add(f.stem)
    return seqs

# ─────────────────────────────────────────────
# 2. Load and compare
# ─────────────────────────────────────────────
print("Loading sequence lists...")
img_seqs = get_image_seqs()
mosh_seqs = get_mosh_seqs()
mosh_smpl_seqs = get_mosh_smpl_seqs()
yogi2_seqs = get_yogi2_seqs()
pressure_seqs = get_pressure_seqs()
vicon_seqs = get_vicon_seqs()

print(f"\n{'='*60}")
print(f"Sequence counts")
print(f"{'='*60}")
print(f"  Images:      {len(img_seqs):>4}")
print(f"  Mosh:        {len(mosh_seqs):>4}")
print(f"  Mosh_smpl:   {len(mosh_smpl_seqs):>4}")
print(f"  YOGI_2:      {len(yogi2_seqs):>4}  (improved fits subset)")
print(f"  Pressure:    {len(pressure_seqs):>4}")
print(f"  Vicon:       {len(vicon_seqs):>4}")

# ─────────────────────────────────────────────
# 3. Cross-check: images vs mosh
# ─────────────────────────────────────────────
img_set = set(img_seqs.keys())
mosh_set = set(mosh_seqs.keys())

in_img_not_mosh = img_set - mosh_set
in_mosh_not_img = mosh_set - img_set

print(f"\n{'='*60}")
print(f"Images ↔ Mosh alignment")
print(f"{'='*60}")
print(f"  In images but NOT in mosh ({len(in_img_not_mosh)}): {sorted(in_img_not_mosh)[:5]}")
print(f"  In mosh but NOT in images ({len(in_mosh_not_img)}): {sorted(in_mosh_not_img)[:5]}")

# ─────────────────────────────────────────────
# 4. Frame count check (images vs mosh)
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Frame count check (images vs mosh)")
print(f"{'='*60}")
mismatches = []
sample_checked = 0
for seq_name in sorted(img_set & mosh_set)[:20]:  # check first 20 for speed
    img_info = img_seqs[seq_name]
    mosh_path = mosh_seqs[seq_name]["path"]
    
    # Count frames from one camera
    cam_counts = img_info["cameras"]
    if not cam_counts:
        continue
    cam1 = sorted(cam_counts.keys())[0]
    n_img_frames = cam_counts[cam1]
    
    # Count frames from mosh pkl
    try:
        with open(mosh_path, "rb") as f:
            mosh_data = pickle.load(f, encoding="latin1")
        # Mosh stores poses; check common keys
        n_mosh_frames = None
        for key in ("poses", "pose_body", "trans"):
            if key in mosh_data:
                n_mosh_frames = len(mosh_data[key])
                break
        if n_mosh_frames is None and "root_orient" in mosh_data:
            n_mosh_frames = len(mosh_data["root_orient"])
    except Exception as e:
        print(f"  ERROR loading {seq_name}: {e}")
        continue
    
    sample_checked += 1
    if n_mosh_frames is not None and abs(n_img_frames - n_mosh_frames) > 2:
        mismatches.append((seq_name, n_img_frames, n_mosh_frames))
    else:
        pass  # OK

print(f"  Checked {sample_checked} sequences")
if mismatches:
    print(f"  MISMATCHES ({len(mismatches)}):")
    for s, ni, nm in mismatches:
        print(f"    {s}: {ni} img frames vs {nm} mosh frames")
else:
    print(f"  All OK - frames match")

# ─────────────────────────────────────────────
# 5. Camera params check
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Camera params check")
print(f"{'='*60}")
cam_root = ROOT / "cameras"
for date in ("20220923", "20220926", "20221004"):
    cam_dir = cam_root / date
    json_files = list(cam_dir.glob("**/cameras_param.json"))
    print(f"  {date}: {len(json_files)} cameras_param.json file(s)")
    for jf in json_files[:1]:
        try:
            with open(jf) as f:
                cam_data = json.load(f)
            cam_ids = list(cam_data.keys()) if isinstance(cam_data, dict) else "?"
            print(f"    -> keys: {cam_ids[:5]}")
        except Exception as e:
            print(f"    -> ERROR: {e}")

# ─────────────────────────────────────────────
# 6. Camera count per image sequence
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Camera count per sequence (images)")
print(f"{'='*60}")
cam_counts_dist = defaultdict(int)
for seq, info in img_seqs.items():
    cam_counts_dist[len(info["cameras"])] += 1
for n_cams, count in sorted(cam_counts_dist.items()):
    print(f"  {n_cams} cameras: {count} sequences")

print(f"\nDone.")
