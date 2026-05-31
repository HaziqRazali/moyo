#!/usr/bin/env python3
"""
MOYO dataset annotation visualizer — 3-panel output per frame:

    [ MMPose skeleton  |  SMPL-X mesh  |  MHR mesh ]

Usage:
    conda activate mhr_new
    python visualize_annotations.py --seq 220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a --n_frames -1
        [--cam YOGI_Cam_02]   # if omitted, processes all cameras
        [--n_frames 50]       # frames to render evenly-spaced (-1 = all)
        [--alpha 0.7]         # mesh blend opacity
        [--score_thr 0.3]     # keypoint confidence threshold for drawing
        [--use_mosh]          # use mosh pkl instead of YOGI_2 npz for SMPL-X
        [--no_video]          # skip MP4 generation
        [--out_dir ./visualize_annotations]

Data notes:
  - Camera positions are in mm; SMPL-X/MHR trans is in meters → divide cam pos by 1000
  - YOGI_2 @ 60 fps, images @ ~30 fps → frame mapping uses embedded frame number in filename
  - MHR verts from forward() are in cm → divide by 100 for meters before rendering
  - MMPose frame_ids are 1-based sequential: frame_id = img_idx + 1
  - MHR faces: mhr_model.character.mesh.faces
"""

import argparse
import json
import os
import pickle
import re
import sys

import cv2
import numpy as np
import torch
import trimesh
import pyrender
import smplx
from pathlib import Path

# MHR lives in /home/haziq/MHR
sys.path.insert(0, "/home/haziq/MHR")
from mhr.mhr import MHR

# Matches image stems ending in _XXXX (4-digit frame number)
_FRAME_RE = re.compile(r'_(\d{4})$')

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
ROOT             = Path("/data/haziq/moyo/data")
SMPLX_MODEL_PATH = "/data/haziq/mocap/data/models_smplx_v1_1/models"
MMPOSE_MODEL     = "rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122"

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ──────────────────────────────────────────────────────────────
# Shared helpers (camera, images, SMPL-X)
# ──────────────────────────────────────────────────────────────

def seq_to_date(seq_name: str) -> str:
    day = seq_name[:6]
    return "20" + day[:2] + day[2:]


def cam_name_to_key(cam_name: str) -> str:
    n = int(cam_name.split("_")[-1])
    return f"cam_{n}"


def load_cameras(date: str) -> dict:
    cam_dir = ROOT / "cameras" / date
    json_files = list(cam_dir.glob("*PROCESSED_CAMERA_PARAMS*/cameras_param.json"))
    if not json_files:
        raise FileNotFoundError(f"No cameras_param.json found for date {date}")
    with open(json_files[0]) as f:
        return json.load(f)


def find_seq_images(seq_name: str, cam_name: str):
    for split in ("20220923_20220926_with_hands", "20221004_with_com"):
        for tv in ("train", "val"):
            p = ROOT / "images" / split / tv / seq_name / cam_name
            if p.exists():
                return sorted(f for f in p.glob("*.jpg") if _FRAME_RE.search(f.stem))
    return []


def find_seq_cameras(seq_name: str):
    cameras = set()
    for split in ("20220923_20220926_with_hands", "20221004_with_com"):
        for tv in ("train", "val"):
            seq_dir = ROOT / "images" / split / tv / seq_name
            if seq_dir.exists():
                for cam_dir in seq_dir.iterdir():
                    if cam_dir.is_dir():
                        cameras.add(cam_dir.name)
    return sorted(cameras)


def find_smplx_pose_file(seq_name: str, prefer_yogi2: bool = True):
    if prefer_yogi2:
        for tv in ("train", "val"):
            p = ROOT / "YOGI_2_latest_smplx_neutral" / tv / f"{seq_name}_stageii.npz"
            if p.exists():
                return p, "yogi2"
    for tv in ("train", "val"):
        p = ROOT / "mosh" / tv / f"{seq_name}_stageii.pkl"
        if p.exists():
            return p, "mosh"
    return None, None


def load_smplx_poses(path: Path, pose_type: str) -> dict:
    if pose_type == "yogi2":
        d = np.load(path, allow_pickle=True)
        n_betas = int(d["num_betas"])
        return {
            "root_orient":     d["root_orient"],
            "body_pose":       d["pose_body"],
            "left_hand_pose":  d["pose_hand"][:, :45],
            "right_hand_pose": d["pose_hand"][:, 45:],
            "jaw_pose":        d["pose_jaw"],
            "leye_pose":       d["pose_eye"][:, :3],
            "reye_pose":       d["pose_eye"][:, 3:],
            "trans":           d["trans"],
            "betas":           d["betas"][:n_betas],
            "n_betas":         n_betas,
            "fps":             float(d["mocap_frame_rate"]),
        }
    else:
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        fp = d["fullpose"]
        return {
            "root_orient":     fp[:, :3],
            "body_pose":       fp[:, 3:66],
            "jaw_pose":        fp[:, 66:69],
            "leye_pose":       fp[:, 69:72],
            "reye_pose":       fp[:, 72:75],
            "left_hand_pose":  fp[:, 75:120],
            "right_hand_pose": fp[:, 120:165],
            "trans":           d["trans"],
            "betas":           d["betas"][:10],
            "n_betas":         10,
            "fps":             60.0,
        }


def get_camera_params(cam: dict, intr_scale: float = 0.5):
    R    = np.array(cam["rotation"], dtype=np.float64)
    C_m  = np.array(cam["position"], dtype=np.float64) / 1000.0
    t    = -R @ C_m
    focal = float(cam["focal"]) * intr_scale
    cx    = float(cam["princpt"][0]) * intr_scale
    cy    = float(cam["princpt"][1]) * intr_scale
    return R, t, focal, cx, cy


def build_smplx_model(n_betas: int, device: str):
    return smplx.create(
        SMPLX_MODEL_PATH, model_type="smplx", gender="neutral",
        num_betas=min(n_betas, 300), use_pca=False, flat_hand_mean=False,
        batch_size=1,
    ).to(device)


def get_smplx_verts(model, poses: dict, frame_idx: int, device: str) -> np.ndarray:
    def _t(key):
        return torch.tensor(poses[key][frame_idx:frame_idx+1], dtype=torch.float32).to(device)
    betas_t = torch.tensor(poses["betas"][:model.num_betas], dtype=torch.float32).unsqueeze(0).to(device)
    out = model(
        global_orient=_t("root_orient"), body_pose=_t("body_pose"),
        left_hand_pose=_t("left_hand_pose"), right_hand_pose=_t("right_hand_pose"),
        jaw_pose=_t("jaw_pose"), leye_pose=_t("leye_pose"), reye_pose=_t("reye_pose"),
        betas=betas_t, transl=_t("trans"), return_verts=True,
    )
    return out.vertices[0].detach().cpu().numpy()


def render_mesh_overlay(img_rgb: np.ndarray, verts: np.ndarray, faces: np.ndarray,
                        R: np.ndarray, t: np.ndarray,
                        focal: float, cx: float, cy: float,
                        alpha: float = 0.7,
                        color: tuple = (0.65, 0.74, 0.86)) -> np.ndarray:
    """Render a mesh overlay on img_rgb. Returns blended RGB image."""
    h, w = img_rgb.shape[:2]
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.6,
        baseColorFactor=(*color, 1.0),
    )
    mesh_pr = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=verts, faces=faces, process=False),
        material=material, smooth=True,
    )
    scene = pyrender.Scene(ambient_light=(0.4, 0.4, 0.4), bg_color=(0, 0, 0, 0))
    scene.add(mesh_pr)

    Rt44 = np.eye(4)
    Rt44[:3, :3] = R
    Rt44[:3, 3]  = t
    c2w = np.linalg.inv(Rt44)
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1

    scene.add(pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy, znear=0.05, zfar=50.0), pose=c2w)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=c2w)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
              pose=c2w @ np.diag([-1., 1., -1., 1.]))

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    mask = depth > 0
    out  = img_rgb.copy()
    out[mask] = (alpha * color_rgba[mask, :3].astype(np.float32)
                 + (1.0 - alpha) * img_rgb[mask].astype(np.float32)).clip(0, 255).astype(np.uint8)
    return out


# ──────────────────────────────────────────────────────────────
# MMPose helpers
# ──────────────────────────────────────────────────────────────

def find_mmpose_json(seq_name: str, cam_name: str):
    for split in ("20220923_20220926_with_hands", "20221004_with_com"):
        for tv in ("train", "val"):
            p = ROOT / "mmpose" / MMPOSE_MODEL / split / tv / seq_name / cam_name / "result.json"
            if p.exists():
                return p
    return None


def _parse_color_field(raw):
    """MMPose serializes numpy arrays as {'__ndarray__': [[R,G,B],...], ...}.
    Return a plain list-of-tuples or None."""
    if raw is None:
        return None
    if isinstance(raw, dict) and "__ndarray__" in raw:
        raw = raw["__ndarray__"]
    if isinstance(raw, list):
        return [tuple(int(v) for v in c[:3]) for c in raw]
    return None


def load_mmpose(path: Path):
    """
    Returns:
        frame2kpts   — dict: frame_id (1-based) → (kpts (133,2), scores (133,))
        skeleton     — list of (i, j) index pairs
        kpt_colors   — list of (R,G,B) per keypoint (or None)
        limb_colors  — list of (R,G,B) per skeleton link (or None)
    """
    with open(path) as f:
        data = json.load(f)
    meta = data["meta_info"]
    skeleton    = [tuple(e) for e in meta["skeleton_links"]]
    kpt_colors  = _parse_color_field(meta.get("keypoint_colors"))
    limb_colors = _parse_color_field(meta.get("skeleton_link_colors"))

    frame2kpts = {}
    for entry in data["instance_info"]:
        fid = entry["frame_id"]
        insts = entry.get("instances", [])
        if insts:
            det = insts[0]  # take first (highest-score) person
            frame2kpts[fid] = (
                np.array(det["keypoints"], dtype=np.float32),        # (133, 2)
                np.array(det["keypoint_scores"], dtype=np.float32),  # (133,)
            )
    return frame2kpts, skeleton, kpt_colors, limb_colors


_LIMB_COLORS_DEFAULT = [
    (255, 128,   0), (255, 153,  51), (255, 178, 102),
    (230, 230,   0), (255, 153, 255), (153, 204, 255),
    ( 51, 153, 255), (230, 230,   0), (255, 153, 255),
    (153, 204, 255), ( 51, 153, 255), (  0, 255,   0),
    ( 52, 153,  51), (128,   0, 255), (255,   0, 255),
    (  0, 255, 255), (255, 102, 255), (255, 102, 102),
    (  0, 128,   0), (  0,   0, 255), ( 51, 153,  51),
]


def draw_skeleton(img_rgb: np.ndarray,
                  kpts: np.ndarray, scores: np.ndarray,
                  skeleton: list, kpt_colors=None, limb_colors=None,
                  score_thr: float = 0.3) -> np.ndarray:
    """Draw 2-D keypoints and limbs on img_rgb. Returns a new RGB image."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    radius = max(3, int(min(h, w) * 0.007))
    thickness = max(2, int(min(h, w) * 0.004))

    # Draw limbs
    for idx, (i, j) in enumerate(skeleton):
        if i >= len(kpts) or j >= len(kpts):
            continue
        if scores[i] < score_thr or scores[j] < score_thr:
            continue
        xi, yi = int(round(kpts[i, 0])), int(round(kpts[i, 1]))
        xj, yj = int(round(kpts[j, 0])), int(round(kpts[j, 1]))
        if limb_colors is not None and idx < len(limb_colors):
            color = limb_colors[idx]
        else:
            color = _LIMB_COLORS_DEFAULT[idx % len(_LIMB_COLORS_DEFAULT)]
        cv2.line(out, (xi, yi), (xj, yj), color, thickness, cv2.LINE_AA)

    # Draw joints
    for idx, (kp, sc) in enumerate(zip(kpts, scores)):
        if sc < score_thr:
            continue
        x, y = int(round(kp[0])), int(round(kp[1]))
        if kpt_colors is not None and idx < len(kpt_colors):
            c = tuple(int(x) for x in kpt_colors[idx][:3])
        else:
            c = (255, 255, 255)
        cv2.circle(out, (x, y), radius, c, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius, (0, 0, 0), max(1, thickness // 2), cv2.LINE_AA)

    return out


# ──────────────────────────────────────────────────────────────
# MHR helpers
# ──────────────────────────────────────────────────────────────

def find_mhr_npz(seq_name: str):
    for tv in ("train", "val"):
        p = ROOT / "mhr" / tv / f"{seq_name}.npz"
        if p.exists():
            return p
    return None


def load_mhr_poses(path: Path) -> dict:
    d = np.load(path)
    return {
        "body_pose_params": d["body_pose_params"].astype(np.float32),  # (N, 130)
        "global_trans":     d["global_trans"].astype(np.float32),      # (N, 3)
        "global_orient":    d["global_orient"].astype(np.float32),     # (N, 3)
        "shape_params":     d["shape_params"].astype(np.float32),      # (N, 45)
        "expr_params":      d["expr_params"].astype(np.float32),       # (N, 72)
        "n_frames":         int(d["body_pose_params"].shape[0]),
        "fps":              60.0,
    }


def build_mhr_model(device: str):
    return MHR.from_files(device=torch.device(device), lod=1)


def get_mhr_verts(mhr_model, mhr_poses: dict, frame_idx: int, device: str) -> np.ndarray:
    """
    Run MHR forward for one frame.
    model_parameters layout (204,):
        [0:3]    global_trans   (stored in cm — same units as the fitted character)
        [3:6]    global_orient  (axis-angle)
        [6:136]  body_pose_params (130 joint angles)
        [136:204] zeros          (scale params)
    Returns vertices in **meters** (V, 3).
    """
    i = frame_idx
    model_params = np.concatenate([
        mhr_poses["global_trans"][i],   # (3,)
        mhr_poses["global_orient"][i],  # (3,)
        mhr_poses["body_pose_params"][i],  # (130,)
        np.zeros(68, dtype=np.float32),    # scale params
    ])[None]  # (1, 204)

    id_coeffs   = mhr_poses["shape_params"][i:i+1]  # (1, 45)
    expr_coeffs = mhr_poses["expr_params"][i:i+1]   # (1, 72)

    with torch.no_grad():
        verts_cm, _ = mhr_model(
            identity_coeffs=torch.tensor(id_coeffs,    dtype=torch.float32).to(device),
            model_parameters=torch.tensor(model_params, dtype=torch.float32).to(device),
            face_expr_coeffs=torch.tensor(expr_coeffs,  dtype=torch.float32).to(device),
        )
    return verts_cm[0].cpu().numpy() / 100.0  # cm → meters


# ──────────────────────────────────────────────────────────────
# Per-camera processing
# ──────────────────────────────────────────────────────────────

def _add_label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color,    2, cv2.LINE_AA)
    return img


def process_camera(seq_name: str, cam_name: str,
                   smplx_poses: dict | None, smplx_model,
                   mhr_poses: dict | None, mhr_model,
                   cam_data: dict,
                   out_base: Path, args) -> None:
    """Render 3-panel frames for one camera and optionally write a video."""

    # ── Images ────────────────────────────────────────────────
    img_paths = find_seq_images(seq_name, cam_name)
    if not img_paths:
        print(f"  WARNING: No images found for cam={cam_name}")
        return
    print(f"\n[{cam_name}]  {len(img_paths)} images  →  {img_paths[0].parent}")

    # ── Camera params ─────────────────────────────────────────
    cam_key = cam_name_to_key(cam_name)
    if cam_key not in cam_data:
        print(f"  WARNING: {cam_key} not found in cameras_param.json")
        return
    R, t, focal, cx, cy = get_camera_params(cam_data[cam_key], intr_scale=0.5)
    print(f"           focal={focal:.1f}px  cx={cx:.1f}  cy={cy:.1f}  t_z={t[2]:.3f}m")

    # ── MMPose ────────────────────────────────────────────────
    mmpose_path = find_mmpose_json(seq_name, cam_name)
    if mmpose_path is None:
        print(f"  WARNING: No MMPose JSON found for {cam_name} — skeleton panel will be blank")
        frame2kpts, skeleton, kpt_colors, limb_colors = {}, [], None, None
    else:
        frame2kpts, skeleton, kpt_colors, limb_colors = load_mmpose(mmpose_path)
        print(f"           mmpose: {len(frame2kpts)} frames  ({mmpose_path.name})")

    # ── Frame selection ───────────────────────────────────────
    n_imgs = len(img_paths)
    n_smplx = len(smplx_poses["trans"]) if smplx_poses else 0
    n_mhr   = mhr_poses["n_frames"]    if mhr_poses   else 0
    n_pose  = n_smplx or n_mhr or n_imgs

    mocap_fps = (smplx_poses["fps"] if smplx_poses else
                 mhr_poses["fps"]   if mhr_poses   else 60.0)
    mocap_duration = n_pose / mocap_fps
    cam_fps = round(n_imgs / mocap_duration) if mocap_duration > 0 else 30
    if cam_fps <= 0:
        cam_fps = 30
    print(f"           mocap={mocap_fps:.0f}fps  cam≈{cam_fps}fps  ratio={mocap_fps/cam_fps:.2f}x")

    if args.n_frames == -1 or args.n_frames >= n_imgs:
        img_indices = list(range(n_imgs))
    else:
        img_indices = np.linspace(0, n_imgs - 1, args.n_frames, dtype=int).tolist()
    print(f"           rendering {len(img_indices)} frames...")

    # ── MHR faces (loaded once) ───────────────────────────────
    mhr_faces = np.asarray(mhr_model.character.mesh.faces, dtype=np.int32) if mhr_model else None

    # ── SMPL-X faces ─────────────────────────────────────────
    smplx_faces = smplx_model.faces if smplx_model else None

    out_dir = out_base / seq_name / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for loop_i, img_idx in enumerate(img_indices):
        img_path = img_paths[img_idx]

        # Frame number → pose index
        m = _FRAME_RE.search(img_path.stem)
        if m:
            frame_number = int(m.group(1))
            pose_idx = int(round(frame_number * mocap_fps / cam_fps))
        else:
            pose_idx = int(round(img_idx * n_pose / n_imgs))
        pose_idx_smplx = min(max(pose_idx, 0), n_smplx - 1) if n_smplx else 0
        pose_idx_mhr   = min(max(pose_idx, 0), n_mhr   - 1) if n_mhr   else 0

        # MMPose frame_id = img_idx + 1 (1-based sequential over image list)
        mmpose_frame_id = img_idx + 1

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: could not read {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Panel 1: MMPose skeleton ──────────────────────────
        if frame2kpts and mmpose_frame_id in frame2kpts:
            kpts, scores = frame2kpts[mmpose_frame_id]
            panel_kpts = draw_skeleton(img_rgb, kpts, scores, skeleton, kpt_colors, limb_colors, args.score_thr)
        else:
            panel_kpts = img_rgb.copy()
        panel_kpts = _add_label(panel_kpts, "MMPose")

        # ── Panel 2: SMPL-X mesh ──────────────────────────────
        if smplx_poses and smplx_model:
            smplx_verts = get_smplx_verts(smplx_model, smplx_poses, pose_idx_smplx, args.device)
            panel_smplx = render_mesh_overlay(img_rgb, smplx_verts, smplx_faces,
                                              R, t, focal, cx, cy, args.alpha,
                                              color=(0.65, 0.74, 0.86))
        else:
            panel_smplx = img_rgb.copy()
        panel_smplx = _add_label(panel_smplx, "SMPL-X")

        # ── Panel 3: MHR mesh ─────────────────────────────────
        if mhr_poses and mhr_model:
            mhr_verts = get_mhr_verts(mhr_model, mhr_poses, pose_idx_mhr, args.device)
            panel_mhr = render_mesh_overlay(img_rgb, mhr_verts, mhr_faces,
                                            R, t, focal, cx, cy, args.alpha,
                                            color=(0.95, 0.80, 0.70))
        else:
            panel_mhr = img_rgb.copy()
        panel_mhr = _add_label(panel_mhr, "MHR")

        # ── Combine ───────────────────────────────────────────
        combined_rgb = np.hstack([panel_kpts, panel_smplx, panel_mhr])
        out_path = out_dir / f"frame_{img_idx:05d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

        if (loop_i + 1) % 10 == 0 or loop_i == 0 or (loop_i + 1) == len(img_indices):
            print(f"  [{loop_i+1:>4}/{len(img_indices)}]  img={img_idx:05d}  pose_smplx={pose_idx_smplx:05d}  pose_mhr={pose_idx_mhr:05d}  mmpose_fid={mmpose_frame_id}")

    print(f"  Saved {len(saved)} frames to {out_dir}\n")

    # ── Video ─────────────────────────────────────────────────
    if not args.no_video and saved:
        try:
            first = cv2.imread(str(saved[0]))
            h, w = first.shape[:2]
            video_path = out_base / f"{seq_name}__{cam_name}__annotations.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
            )
            for fp in sorted(saved):
                frame = cv2.imread(str(fp))
                if frame is not None:
                    writer.write(frame)
            writer.release()
            print(f"  Video: {video_path.name}\n")
        except Exception as e:
            print(f"  Video creation failed: {e}\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MOYO 3-panel annotation visualizer: MMPose | SMPL-X | MHR"
    )
    parser.add_argument("--seq",       required=True, help="Sequence name")
    parser.add_argument("--cam",       default=None,  help="Camera name (default: all)")
    parser.add_argument("--out_dir",   default="./visualize_annotations")
    parser.add_argument("--n_frames",  type=int,   default=30,  help="-1 = all")
    parser.add_argument("--alpha",     type=float, default=0.7)
    parser.add_argument("--score_thr", type=float, default=0.3, help="MMPose confidence threshold")
    parser.add_argument("--use_mosh",  action="store_true", help="Use mosh pkl for SMPL-X")
    parser.add_argument("--no_video",  action="store_true")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_base = Path(args.out_dir)

    # ── SMPL-X ────────────────────────────────────────────────
    smplx_path, smplx_type = find_smplx_pose_file(args.seq, prefer_yogi2=not args.use_mosh)
    if smplx_path is None:
        print("[WARN] No SMPL-X pose file found — SMPL-X panel will be blank")
        smplx_poses, smplx_model = None, None
    else:
        smplx_poses = load_smplx_poses(smplx_path, smplx_type)
        print(f"[smplx]  {len(smplx_poses['trans'])} frames @ {smplx_poses['fps']}fps  ({smplx_type})")
        print(f"[smplx]  loading model  n_betas={min(smplx_poses['n_betas'], 300)}  device={args.device}")
        smplx_model = build_smplx_model(smplx_poses["n_betas"], args.device)

    # ── MHR ───────────────────────────────────────────────────
    mhr_path = find_mhr_npz(args.seq)
    if mhr_path is None:
        print("[WARN] No MHR npz found — MHR panel will be blank")
        mhr_poses, mhr_model = None, None
    else:
        mhr_poses = load_mhr_poses(mhr_path)
        print(f"[mhr]    {mhr_poses['n_frames']} frames @ {mhr_poses['fps']}fps  ({mhr_path})")
        print(f"[mhr]    loading model  lod=1  device={args.device}")
        mhr_model = build_mhr_model(args.device)

    # ── Cameras ───────────────────────────────────────────────
    if args.cam is None:
        cameras = find_seq_cameras(args.seq)
        if not cameras:
            print(f"ERROR: No cameras found for {args.seq}")
            sys.exit(1)
        print(f"[cams]   {len(cameras)}: {', '.join(cameras)}\n")
    else:
        cameras = [args.cam]

    date     = seq_to_date(args.seq)
    cam_data = load_cameras(date)

    for cam_name in cameras:
        process_camera(args.seq, cam_name,
                       smplx_poses, smplx_model,
                       mhr_poses, mhr_model,
                       cam_data, out_base, args)


if __name__ == "__main__":
    main()
