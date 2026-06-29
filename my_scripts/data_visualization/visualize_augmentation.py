#!/usr/bin/env python3
"""
Visualize the 3D virtual-camera augmentation pipeline on MOYO sequences.

Output: one PNG per frame showing N rows
  Row 1 : original image with SMPL-X mesh | MHR mesh | MMPose skeleton overlaid
  Row 2 : FRONT camera (from compute_median_front_camera) → SMPL-X | MHR | MMPose on white
  Rows 3+ : randomly sampled augmented cameras → SMPL-X | MHR | MMPose on white

This lets you verify that:
  1. Camera params are correct (row 1 overlays should align)
  2. The front camera points at the person from the front (row 2)
  3. Random samples are plausible perturbations around that (rows 3+)

Usage:
    conda activate mhr_new
    python visualize_augmentation.py \
        --seq 220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a \
        --cam YOGI_Cam_02 \
        --n_rows 10 \
        --n_frames 10
"""

import argparse
import json
import os
import re
import sys
import warnings

import cv2
import numpy as np
import torch
import trimesh
import pyrender
import smplx
from pathlib import Path
from scipy.spatial.transform import Rotation as _Rsci

sys.path.insert(0, "/home/haziq/MHR")
sys.path.insert(0, "/home/haziq/Collab_AI/dataloaders")
from mhr.mhr import MHR
from utils_3d_aug import (
    triangulate_ransac_single,
    compute_front_camera, sample_aug_camera,
    project_world_to_pixel, project_kpts_through_camera,
)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
warnings.filterwarnings("ignore")

_FRAME_RE     = re.compile(r'_(\d{4})$')
ROOT          = Path("/data/haziq/moyo/data")
SMPLX_PATH    = "/data/haziq/mocap/data/models_smplx_v1_1/models"
MMPOSE_MODEL  = "rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122"
MOCAP_FPS     = 60.0
CAM_FPS       = 30.0

# ── virtual canvas (same as ref camera, scaled down) ─────────────────────────
CANVAS_W = 2160   # will be derived from ref camera at runtime
CANVAS_H = 2880

# ── skeleton drawing defaults ─────────────────────────────────────────────────
_LIMB_COLORS = [
    (255,128,0),(255,153,51),(255,178,102),(230,230,0),(255,153,255),
    (153,204,255),(51,153,255),(230,230,0),(255,153,255),(153,204,255),
    (51,153,255),(0,255,0),(52,153,51),(128,0,255),(255,0,255),
    (0,255,255),(255,102,255),(255,102,102),(0,128,0),(0,0,255),(51,153,51),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Camera helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_cameras(seq_name: str) -> dict:
    """Load session-level camera params for a sequence's date."""
    date = "20" + seq_name[:2] + seq_name[2:4] + seq_name[4:6]
    cam_dir = ROOT / "cameras" / date
    files = list(cam_dir.glob("*PROCESSED_CAMERA_PARAMS*/cameras_param.json"))
    if not files:
        raise FileNotFoundError(f"No cameras_param.json for date {date}")
    with open(files[0]) as f:
        return json.load(f)


INTR_SCALE = 0.5  # MOYO images are stored at 0.5x calibration resolution


def cam_params_for(cam_json: dict, intr_scale: float = INTR_SCALE):
    """
    Returns R (3,3), T (3,) camera-centre, f (scalar), cx, cy.
    Convention: X_cam = R @ (X_world - T)   [T = camera centre in world, metres]
    Intrinsics are scaled to match the 0.5x stored images.
    """
    R    = np.array(cam_json["rotation"], dtype=np.float64)
    T    = np.array(cam_json["position"], dtype=np.float64) / 1000.0   # mm → m
    f    = float(cam_json["focal"]) * intr_scale
    cx   = float(cam_json["princpt"][0]) * intr_scale
    cy   = float(cam_json["princpt"][1]) * intr_scale
    t    = -R @ T    # translation vector  (for render_mesh_overlay / pyrender)
    return R, T, t, f, cx, cy


def cam_key(cam_name: str) -> str:
    return "cam_" + str(int(cam_name.split("_")[-1]))


def read_cam_params_dict(cam_json: dict, intr_scale: float = INTR_SCALE) -> dict:
    """Build a cam_params dict in the format expected by utils_3d_aug."""
    R  = np.array(cam_json["rotation"], dtype=np.float64)
    T  = np.array(cam_json["position"], dtype=np.float64) / 1000.0
    f  = float(cam_json["focal"]) * intr_scale
    cx = float(cam_json["princpt"][0]) * intr_scale
    cy = float(cam_json["princpt"][1]) * intr_scale
    return {
        "extrinsics": {
            "R": R,
            "T": T,          # camera centre (metres) — as expected by project_world_to_pixel
        },
        "intrinsics_wo_distortion": {
            "f": np.array([f, f]),
            "c": np.array([cx, cy]),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Image / MMPose helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_images(seq_name: str, cam_name: str):
    for split in ("20220923_20220926_with_hands", "20221004_with_com"):
        for tv in ("train", "val"):
            p = ROOT / "images" / split / tv / seq_name / cam_name
            if p.exists():
                return sorted(f for f in p.glob("*.jpg") if _FRAME_RE.search(f.stem))
    return []


def find_mmpose(seq_name: str, cam_name: str):
    for split in ("20220923_20220926_with_hands", "20221004_with_com"):
        for tv in ("train", "val"):
            p = ROOT / "mmpose" / MMPOSE_MODEL / split / tv / seq_name / cam_name / "result.json"
            if p.exists():
                return p
    return None


def _parse_colors(raw):
    if raw is None:
        return None
    if isinstance(raw, dict) and "__ndarray__" in raw:
        raw = raw["__ndarray__"]
    if isinstance(raw, list):
        return [tuple(int(v) for v in c[:3]) for c in raw]
    return None


def load_mmpose_json(path: Path):
    with open(path) as f:
        data = json.load(f)
    meta   = data["meta_info"]
    skel   = [tuple(e) for e in meta["skeleton_links"]]
    kpt_c  = _parse_colors(meta.get("keypoint_colors"))
    lnk_c  = _parse_colors(meta.get("skeleton_link_colors"))
    fid2kp = {}
    for entry in data["instance_info"]:
        fid  = entry["frame_id"]
        inst = entry.get("instances", [])
        if inst:
            fid2kp[fid] = (
                np.array(inst[0]["keypoints"],       dtype=np.float32),
                np.array(inst[0]["keypoint_scores"], dtype=np.float32),
            )
    return fid2kp, skel, kpt_c, lnk_c, data["instance_info"]


def draw_skeleton(img, kpts, scores, skel, kpt_c=None, lnk_c=None, thr=0.3):
    out = img.copy()
    h, w = out.shape[:2]
    r  = max(3, int(min(h, w) * 0.007))
    th = max(2, int(min(h, w) * 0.004))
    for idx, (i, j) in enumerate(skel):
        if i >= len(kpts) or j >= len(kpts):
            continue
        if scores[i] < thr or scores[j] < thr:
            continue
        xi, yi = int(round(kpts[i,0])), int(round(kpts[i,1]))
        xj, yj = int(round(kpts[j,0])), int(round(kpts[j,1]))
        c = lnk_c[idx] if lnk_c and idx < len(lnk_c) else _LIMB_COLORS[idx % len(_LIMB_COLORS)]
        cv2.line(out, (xi, yi), (xj, yj), c, th, cv2.LINE_AA)
    for idx, (kp, sc) in enumerate(zip(kpts, scores)):
        if sc < thr:
            continue
        x, y = int(round(kp[0])), int(round(kp[1]))
        c = tuple(int(v) for v in kpt_c[idx][:3]) if kpt_c and idx < len(kpt_c) else (255,255,255)
        cv2.circle(out, (x,y), r, c, -1, cv2.LINE_AA)
        cv2.circle(out, (x,y), r, (0,0,0), max(1, th//2), cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SMPL-X helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_smplx(seq_name: str):
    for tv in ("train", "val"):
        p = ROOT / "YOGI_2_latest_smplx_neutral" / tv / f"{seq_name}_stageii.npz"
        if p.exists():
            return p
    return None


def load_smplx_poses(path: Path):
    d = np.load(path, allow_pickle=True)
    nb = int(d["num_betas"])
    return {
        "root_orient":     d["root_orient"],
        "body_pose":       d["pose_body"],
        "left_hand_pose":  d["pose_hand"][:, :45],
        "right_hand_pose": d["pose_hand"][:, 45:],
        "jaw_pose":        d["pose_jaw"],
        "leye_pose":       d["pose_eye"][:, :3],
        "reye_pose":       d["pose_eye"][:, 3:],
        "trans":           d["trans"],
        "betas":           d["betas"][:nb],
        "n_betas":         nb,
        "fps":             float(d["mocap_frame_rate"]),
    }


def build_smplx_model(n_betas: int, device: str):
    return smplx.create(
        SMPLX_PATH, model_type="smplx", gender="neutral",
        num_betas=min(n_betas, 300), use_pca=False, flat_hand_mean=False,
        batch_size=1,
    ).to(device)


def get_smplx_verts(model, poses, idx, device):
    def _t(k): return torch.tensor(poses[k][idx:idx+1], dtype=torch.float32).to(device)
    betas = torch.tensor(poses["betas"][:model.num_betas], dtype=torch.float32).unsqueeze(0).to(device)
    out = model(
        global_orient=_t("root_orient"), body_pose=_t("body_pose"),
        left_hand_pose=_t("left_hand_pose"), right_hand_pose=_t("right_hand_pose"),
        jaw_pose=_t("jaw_pose"), leye_pose=_t("leye_pose"), reye_pose=_t("reye_pose"),
        betas=betas, transl=_t("trans"), return_verts=True,
    )
    return out.vertices[0].detach().cpu().numpy()  # world-space, metres


# ═══════════════════════════════════════════════════════════════════════════════
# MHR helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_mhr(seq_name: str):
    for tv in ("train", "val"):
        p = ROOT / "mhr" / tv / f"{seq_name}.npz"
        if p.exists():
            return p
    return None


def load_mhr_poses(path: Path):
    d = np.load(path)
    return {
        "body_pose_params": d["body_pose_params"].astype(np.float32),
        "global_trans":     d["global_trans"].astype(np.float32),
        "global_orient":    d["global_orient"].astype(np.float32),
        "shape_params":     d["shape_params"].astype(np.float32),
        "expr_params":      d["expr_params"].astype(np.float32),
    }


def build_mhr_model(device: str):
    return MHR.from_files(device=torch.device(device), lod=1)


def get_mhr_verts(mhr_model, mhr_poses, idx, device):
    mp = np.concatenate([
        mhr_poses["global_trans"][idx],
        mhr_poses["global_orient"][idx],
        mhr_poses["body_pose_params"][idx],
        np.zeros(68, dtype=np.float32),
    ])[None]
    ic = mhr_poses["shape_params"][idx:idx+1]
    ec = mhr_poses["expr_params"][idx:idx+1]
    with torch.no_grad():
        v, _ = mhr_model(
            identity_coeffs=torch.tensor(ic, dtype=torch.float32).to(device),
            model_parameters=torch.tensor(mp, dtype=torch.float32).to(device),
            face_expr_coeffs=torch.tensor(ec, dtype=torch.float32).to(device),
        )
    return v[0].cpu().numpy() / 100.0   # cm → metres (world-space)


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh rendering
# ═══════════════════════════════════════════════════════════════════════════════

def render_mesh(bg: np.ndarray,
                verts: np.ndarray, faces: np.ndarray,
                R: np.ndarray, t: np.ndarray,
                focal: float, cx: float, cy: float,
                alpha: float = 0.75,
                color: tuple = (0.65, 0.74, 0.86)) -> np.ndarray:
    """
    Render mesh onto bg using extrinsics (R, t) where t = -R @ C (translation vector).
    R and t define:  X_cam = R @ X_world + t
    """
    h, w = bg.shape[:2]
    mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.6,
        baseColorFactor=(*color, 1.0),
    )
    mesh_pr = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=verts, faces=faces, process=False),
        material=mat, smooth=True,
    )
    scene = pyrender.Scene(ambient_light=(0.4,0.4,0.4), bg_color=(0,0,0,0))
    scene.add(mesh_pr)

    Rt44 = np.eye(4)
    Rt44[:3,:3] = R
    Rt44[:3, 3] = t
    c2w = np.linalg.inv(Rt44)
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1

    cam_pr = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy, znear=0.05, zfar=50.0)
    scene.add(cam_pr,  pose=c2w)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=c2w)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
              pose=c2w @ np.diag([-1.,1.,-1.,1.]))

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    mask = depth > 0
    out  = bg.copy()
    out[mask] = (alpha * rgba[mask,:3].astype(np.float32)
                 + (1-alpha) * bg[mask].astype(np.float32)).clip(0,255).astype(np.uint8)
    return out


def make_virtual_cam_extrinsics(R_aug: np.ndarray, T_aug: np.ndarray):
    """
    R_aug, T_aug from sample_aug_camera / compute_median_front_camera.
    Convention used there: X_cam = R @ (X_world - T),  T = camera centre.
    Returns (R, t) where t = -R @ T  for pyrender / render_mesh.
    """
    t = -R_aug @ T_aug
    return R_aug, t


# ═══════════════════════════════════════════════════════════════════════════════
# Triangulation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def triangulate_frame(fi: int, mm_data_all: list, cp_list: list, n: int = 133):
    """Triangulate a single frame index. Returns kpts_3d (n, 3) or None."""
    pts_per_cam, cps = [], []
    for mm, cp in zip(mm_data_all, cp_list):
        if fi >= len(mm["instance_info"]):
            continue
        insts = mm["instance_info"][fi]["instances"]
        if insts:
            pts_per_cam.append(np.array(insts[0]["keypoints"], dtype=np.float32))
            cps.append(cp)
    if len(pts_per_cam) < 2:
        return None
    kpts_3d = np.full((n, 3), np.nan, np.float32)
    for j in range(n):
        pts = [k[j] for k in pts_per_cam]
        X, _ = triangulate_ransac_single(pts, cps, thresh=10.0)
        if X is not None and not np.any(np.isnan(X)):
            kpts_3d[j] = X
    return kpts_3d


def triangulate_frames(frame_indices: list, mm_data_all: list, cp_list: list,
                       n: int = 133) -> dict:
    """Triangulate only the requested frame indices. Returns dict: fi → kpts_3d."""
    result = {}
    for fi in frame_indices:
        kpts_3d = triangulate_frame(fi, mm_data_all, cp_list, n)
        if kpts_3d is not None:
            result[fi] = kpts_3d
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Label helper
# ═══════════════════════════════════════════════════════════════════════════════

def _label(img, text, color=(255,255,255)):
    img = img.copy()
    cv2.putText(img, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color,   2, cv2.LINE_AA)
    return img


def _label_bottom(img, text, color=(200,200,200)):
    img = img.copy()
    h = img.shape[0]
    cv2.putText(img, text, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,   1, cv2.LINE_AA)
    return img


def _info_panel(H: int, W: int, lines: list) -> np.ndarray:
    """Dark panel (H x W) with text lines. lines = [(text, bgr_color), ...]"""
    img  = np.full((H, W, 3), 18, dtype=np.uint8)
    sc   = max(0.38, W / 650)
    lh   = max(22, int(sc * 42))
    y    = lh
    for text, color in lines:
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, sc, color,   1, cv2.LINE_AA)
        y += lh
    return img


def _cam_delta(T_aug, R_aug, front_T, front_R):
    """
    Decompose augmented camera into camera-local translation offsets and
    relative Euler angles (degrees) with respect to the front camera.
    Returns (tx, ty, tz, pitch_deg, yaw_deg, roll_deg).
    Sign convention matches sample_aug_camera:
      tx+  = right,  ty+  = down,   tz+  = closer
      pitch+ = tilt down,  yaw+ = turn right,  roll+ = clockwise
    """
    dT = T_aug - front_T
    tx = float(dT @ front_R[0])
    ty = float(dT @ front_R[1])
    tz = float(dT @ front_R[2])
    R_rel  = R_aug @ front_R.T
    angles = _Rsci.from_matrix(R_rel).as_euler('xyz', degrees=True)
    return tx, ty, tz, float(angles[0]), float(angles[1]), float(angles[2])


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def make_three_panels_original(img: np.ndarray,
                                smplx_verts, smplx_faces,
                                mhr_verts, mhr_faces,
                                R, t, focal, cx, cy,
                                kpts2d, scores, skel, kpt_c, lnk_c,
                                alpha, thr) -> np.ndarray:
    """
    Row 1: three side-by-side panels on the original image:
        [ SMPL-X  |  MHR  |  MMPose ]
    """
    p_smplx = img.copy()
    if smplx_verts is not None:
        p_smplx = render_mesh(p_smplx, smplx_verts, smplx_faces, R, t, focal, cx, cy,
                               alpha=alpha, color=(0.65, 0.74, 0.86))

    p_mhr = img.copy()
    if mhr_verts is not None:
        p_mhr = render_mesh(p_mhr, mhr_verts, mhr_faces, R, t, focal, cx, cy,
                             alpha=alpha, color=(0.9, 0.55, 0.3))

    p_kpt = img.copy()
    if kpts2d is not None:
        p_kpt = draw_skeleton(p_kpt, kpts2d, scores, skel, kpt_c, lnk_c, thr)

    p_smplx = _label(p_smplx, "SMPL-X")
    p_mhr   = _label(p_mhr,   "MHR")
    p_kpt   = _label(p_kpt,   "MMPose")
    return np.hstack([p_smplx, p_mhr, p_kpt])


def make_three_panels_virtual(kpts_3d: np.ndarray,
                               smplx_verts, smplx_faces,
                               mhr_verts, mhr_faces,
                               R_aug, T_aug, f_ref, c_ref, W, H,
                               skel, kpt_c, lnk_c,
                               alpha, thr) -> np.ndarray:
    """
    Rows 2+: three white-background panels:
        [ SMPL-X  |  MHR  |  MMPose ]
    projected through the virtual camera (R_aug, T_aug).
    """
    white = np.full((int(H), int(W), 3), 255, dtype=np.uint8)
    R_v, t_v = make_virtual_cam_extrinsics(R_aug, T_aug)

    p_smplx = white.copy()
    if smplx_verts is not None:
        p_smplx = render_mesh(p_smplx, smplx_verts, smplx_faces, R_v, t_v,
                               f_ref[0], c_ref[0], c_ref[1],
                               alpha=alpha, color=(0.65, 0.74, 0.86))

    p_mhr = white.copy()
    if mhr_verts is not None:
        p_mhr = render_mesh(p_mhr, mhr_verts, mhr_faces, R_v, t_v,
                             f_ref[0], c_ref[0], c_ref[1],
                             alpha=alpha, color=(0.9, 0.55, 0.3))

    p_kpt = white.copy()
    if kpts_3d is not None:
        kpts_px, visible = project_kpts_through_camera(
            kpts_3d, R_aug, T_aug, f_ref, c_ref, W, H)
        valid = ~np.isnan(kpts_px[:, 0]) & (visible > 0)
        if valid.sum() > 0:
            sc = visible.astype(np.float32)
            p_kpt = draw_skeleton(p_kpt, kpts_px, sc, skel, kpt_c, lnk_c, thr=0.5)

    p_smplx = _label(p_smplx, "SMPL-X")
    p_mhr   = _label(p_mhr,   "MHR")
    p_kpt   = _label(p_kpt,   "MMPose")
    return np.hstack([p_smplx, p_mhr, p_kpt])


def process_frame(frame_idx: int,              # 0-based index into image list
                  img_path: Path,
                  pose_idx: int,               # index into body-model arrays
                  fid: int,                    # 1-based mmpose frame_id
                  cam_json: dict,              # raw MOYO camera dict for this cam
                  all_cam_jsons: dict,         # all cam dicts keyed by cam_X
                  smplx_model, smplx_poses,
                  mhr_model, mhr_poses,
                  fid2kp: dict, skel, kpt_c, lnk_c,
                  front_T, front_R, world_up,
                  kpts_3d_by_frame: dict,
                  n_rows: int,
                  aug_trans, aug_rot, front_dist,
                  alpha: float = 0.7, thr: float = 0.3,
                  device: str = "cuda") -> np.ndarray:

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_img, W_img = img_rgb.shape[:2]

    # Camera extrinsics for original view (0.5x intrinsics to match stored images)
    R, T, t, focal, cx, cy = cam_params_for(cam_json, intr_scale=INTR_SCALE)

    # SMPL-X vertices (world-space)
    smplx_verts = None
    smplx_faces = None
    if smplx_model is not None and smplx_poses is not None:
        if pose_idx < smplx_poses["root_orient"].shape[0]:
            smplx_verts = get_smplx_verts(smplx_model, smplx_poses, pose_idx, device)
            smplx_faces = smplx_model.faces

    # MHR vertices (world-space, metres)
    mhr_verts = None
    mhr_faces = None
    if mhr_model is not None and mhr_poses is not None:
        if pose_idx < mhr_poses["body_pose_params"].shape[0]:
            mhr_verts = get_mhr_verts(mhr_model, mhr_poses, pose_idx, device)
            mhr_faces = mhr_model.character.mesh.faces

    # MMPose 2D keypoints
    kpts2d, scores = fid2kp.get(fid, (None, None))

    # ── Row 1: 3-panel original image ─────────────────────────
    W_info = max(200, W_img // 4)
    info1  = _info_panel(H_img, W_info, [
        (f"Fr {frame_idx:05d}",        (255, 255, 255)),
        (f"cam {img_path.parent.name}", (180, 180, 180)),
        (f"f={focal:.0f}px",            (180, 180, 180)),
    ])
    row1 = np.hstack([info1, make_three_panels_original(
        img_rgb, smplx_verts, smplx_faces, mhr_verts, mhr_faces,
        R, t, focal, cx, cy,
        kpts2d, scores, skel, kpt_c, lnk_c, alpha, thr)])

    # ── 3-D keypoints for this frame ─────────────────────────
    kpts_3d = kpts_3d_by_frame.get(frame_idx)  # (133,3) or None

    # reference intrinsics for virtual rows: centred principal point.
    # cy=H/2 is correct because compute_front_camera places the camera at
    # shoulder height and backs up until the feet land at (1-margin)*H.
    f_ref = np.array([focal, focal], dtype=np.float64)
    c_ref = np.array([W_img / 2.0, H_img / 2.0], dtype=np.float64)
    W_ref = float(W_img)
    H_ref = float(H_img)

    # ── Row 2: front camera ───────────────────────────────────
    rows = []
    if front_T is not None and front_R is not None and kpts_3d is not None:
        row2_panels = make_three_panels_virtual(
            kpts_3d, smplx_verts, smplx_faces, mhr_verts, mhr_faces,
            front_R, front_T, f_ref, c_ref, W_ref, H_ref,
            skel, kpt_c, lnk_c, alpha, thr)
        info2 = _info_panel(H_img, W_info, [
            ("Front cam",      (0, 200, 255)),
            ("T-pose fr 0",    (0, 180, 200)),
            (f"T=[{front_T[0]:+.2f}",  (180, 180, 180)),
            (f"  {front_T[1]:+.2f}",   (180, 180, 180)),
            (f"  {front_T[2]:+.2f}]m", (180, 180, 180)),
        ])
        row2 = np.hstack([info2, row2_panels])
    else:
        info2 = _info_panel(H_img, W_info, [("Front cam N/A", (100, 100, 100))])
        row2  = np.hstack([info2, np.full((H_img, 3 * W_img, 3), 200, dtype=np.uint8)])
    rows.append(row2)

    # ── Rows 3+: sampled augmented cameras ───────────────────
    for aug_i in range(n_rows - 2):
        if front_T is not None and kpts_3d is not None:
            T_aug, R_aug = sample_aug_camera(
                kpts_3d, world_up,
                front_dist=front_dist,
                W=W_ref, H=H_ref, f=f_ref, c=c_ref,
                trans_range=(
                    (aug_trans[0], aug_trans[1]),
                    (aug_trans[2], aug_trans[3]),
                    (aug_trans[4], aug_trans[5]),
                ),
                rot_limit=(
                    (aug_rot[0], aug_rot[1]),
                    (aug_rot[2], aug_rot[3]),
                    (aug_rot[4], aug_rot[5]),
                ),
                base_T=front_T, base_R=front_R,
            )
        else:
            T_aug, R_aug = None, None

        if T_aug is not None:
            aug_panels = make_three_panels_virtual(
                kpts_3d, smplx_verts, smplx_faces, mhr_verts, mhr_faces,
                R_aug, T_aug, f_ref, c_ref, W_ref, H_ref,
                skel, kpt_c, lnk_c, alpha, thr)
            if front_T is not None:
                tx, ty, tz, pt, yw, ro = _cam_delta(T_aug, R_aug, front_T, front_R)
                info_a = _info_panel(H_img, W_info, [
                    (f"Aug {aug_i+1}",     (0, 255, 128)),
                    (f"TX={tx:+.2f}m",    (200, 200, 200)),
                    (f"TY={ty:+.2f}m",    (200, 200, 200)),
                    (f"TZ={tz:+.2f}m",    (200, 200, 200)),
                    (f"Pt={pt:+.1f}d",    (200, 200, 200)),
                    (f"Yw={yw:+.1f}d",    (200, 200, 200)),
                    (f"Ro={ro:+.1f}d",    (200, 200, 200)),
                ])
            else:
                info_a = _info_panel(H_img, W_info, [(f"Aug {aug_i+1}", (0, 255, 128))])
            row_aug = np.hstack([info_a, aug_panels])
        else:
            info_a  = _info_panel(H_img, W_info, [(f"Aug {aug_i+1} N/A", (100, 100, 100))])
            row_aug = np.hstack([info_a, np.full((H_img, 3 * W_img, 3), 200, dtype=np.uint8)])
        rows.append(row_aug)

    # ── Stack rows into final panel ───────────────────────────
    target_w = row1.shape[1]
    final_rows = [row1]
    for r in rows:
        if r.shape[1] != target_w:
            r = cv2.resize(r, (target_w, int(r.shape[0] * target_w / r.shape[1])))
        final_rows.append(r)

    return np.vstack(final_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",       required=True, help="Sequence name")
    ap.add_argument("--cam",       default="YOGI_Cam_02", help="Camera to use for row 1")
    ap.add_argument("--n_rows",    type=int, default=10,
                    help="Total rows (1 original + 1 front + N-2 random aug)")
    ap.add_argument("--n_frames",  type=int, default=5,
                    help="Number of frames to visualise (-1 = all)")
    ap.add_argument("--alpha",     type=float, default=0.7)
    ap.add_argument("--score_thr", type=float, default=0.3)
    # Translation sign convention (camera-local axes, offsets from auto-computed baseline ~5m):
    #   X+  = camera moves RIGHT    X-  = camera moves LEFT
    #   Y+  = camera moves DOWN     Y-  = camera moves UP   (higher placement)
    #   Z+  = camera moves CLOSER   Z-  = camera moves FARTHER BACK
    #   e.g. TZ_LO=-1.5 TZ_HI=1.5 at 5m baseline → range 3.5m–6.5m from person
    # Rotation sign convention (camera-local axes, degrees):
    #   RX  = pitch  (+ tilts down,   - tilts up)
    #   RY  = yaw    (+ turns right,  - turns left)
    #   RZ  = roll   (+ rolls clockwise, - rolls counter-clockwise)
    ap.add_argument("--aug_trans", nargs=6, type=float,
                    default=[-0.5, 0.5, -0.5, 0.5, -3, 1],
                    metavar=("TX_LO","TX_HI","TY_LO","TY_HI","TZ_LO","TZ_HI"))
    ap.add_argument("--aug_rot",   nargs=6, type=float,
                    default=[-15.0, 15.0, -10.0, 10.0, -10.0, 10.0],
                    metavar=("RX_LO","RX_HI","RY_LO","RY_HI","RZ_LO","RZ_HI"))
    ap.add_argument("--front_dist",type=float, default=2.0)
    ap.add_argument("--out_dir",   default="./visualize_augmentation")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.seq / args.cam
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cameras ──────────────────────────────────────────
    print("Loading cameras...")
    all_cam_raw = load_cameras(args.seq)

    # Build cam_params dicts for all cameras (for triangulation)
    cp_dict = {f"YOGI_Cam_{int(k.split('_')[1]):02d}": read_cam_params_dict(v)
               for k, v in all_cam_raw.items()}
    # Use ALL available cameras for triangulation
    all_cam_names = sorted(cp_dict.keys())
    cp_list = [cp_dict[c] for c in all_cam_names]

    # ── Load MMPose for ALL cameras (for triangulation) ───────
    print("Loading MMPose JSONs for all cameras...")
    mm_data_all = []
    cp_list_valid = []
    cam_names_valid = []
    for cam_name in all_cam_names:
        mmpose_path = find_mmpose(args.seq, cam_name)
        if mmpose_path is None:
            print(f"  [skip] {cam_name}: no mmpose JSON")
            continue
        with open(mmpose_path) as f:
            mm_data_all.append(json.load(f))
        cp_list_valid.append(cp_dict[cam_name])
        cam_names_valid.append(cam_name)
    print(f"  {len(mm_data_all)} cameras loaded for triangulation")

    # ── Load MMPose for the reference camera ──────────────────
    mmpose_path = find_mmpose(args.seq, args.cam)
    if mmpose_path is None:
        raise FileNotFoundError(f"No MMPose JSON for {args.seq} / {args.cam}")
    fid2kp, skel, kpt_c, lnk_c, instance_info = load_mmpose_json(mmpose_path)

    # ── Load images for reference camera ──────────────────────
    images = find_images(args.seq, args.cam)
    if not images:
        raise FileNotFoundError(f"No images found for {args.seq} / {args.cam}")

    # ── Load body models ──────────────────────────────────────
    print("Loading SMPL-X model...")
    smplx_path  = find_smplx(args.seq)
    smplx_poses = load_smplx_poses(smplx_path) if smplx_path else None
    smplx_model = build_smplx_model(smplx_poses["n_betas"] if smplx_poses else 10, args.device) if smplx_poses else None

    print("Loading MHR model...")
    mhr_path  = find_mhr(args.seq)
    mhr_poses = load_mhr_poses(mhr_path) if mhr_path else None
    mhr_model = build_mhr_model(args.device) if mhr_poses else None

    # ── Pick frames to visualise ──────────────────────────────
    n_total = len(images)
    if args.n_frames == -1 or args.n_frames >= n_total:
        frame_indices = list(range(n_total))
    else:
        frame_indices = [int(round(i * (n_total - 1) / (args.n_frames - 1)))
                         for i in range(args.n_frames)]

    # ── Triangulate only the needed frames ────────────────────
    # Also include frame 0 (T-pose) so front camera can be computed from it
    frames_to_tri = sorted(set(frame_indices) | {0})
    print(f"Triangulating {len(frames_to_tri)} frames from {len(mm_data_all)} cameras...")
    if len(mm_data_all) >= 2:
        kpts_3d_by_frame = triangulate_frames(frames_to_tri, mm_data_all, cp_list_valid)
        print(f"  Triangulated {len(kpts_3d_by_frame)} frames")
    else:
        kpts_3d_by_frame = {}
        print("  Not enough cameras for triangulation")

    # ── Compute front camera from T-pose (frame 0) ────────────
    print("Computing frontal camera from T-pose (frame 0)...")
    world_up = np.array([0., -1., 0.])
    front_T = front_R = None

    # Get reference focal + image height for auto-distance computation
    ref_cam_key  = cam_key(args.cam)
    ref_cam_json = all_cam_raw[ref_cam_key]
    _, _, _, focal_ref, _, _ = cam_params_for(ref_cam_json, intr_scale=INTR_SCALE)
    _tmp_img = cv2.imread(str(images[0]))
    H_img_ref = _tmp_img.shape[0] if _tmp_img is not None else 2056

    kpts_3d_frame0 = kpts_3d_by_frame.get(0)
    if kpts_3d_frame0 is not None:
        front_T, front_R = compute_front_camera(
            kpts_3d_frame0, world_up,
            f=focal_ref, H=H_img_ref, bottom_margin=0.10, top_margin=0.10,
        )
        if front_T is not None:
            print("  Front camera: OK (T-pose, auto-dist)")
        else:
            print("  Front camera: FAIL (insufficient joints in frame 0)")
    else:
        print("  Front camera: FAIL (frame 0 not triangulated)")

    mocap_fps = MOCAP_FPS
    cam_fps   = CAM_FPS

    # ── Process each frame ────────────────────────────────────
    for idx_in_list, fi in enumerate(frame_indices):
        img_path = images[fi]
        m = _FRAME_RE.search(img_path.stem)
        frame_number = int(m.group(1))
        pose_idx     = round(frame_number * mocap_fps / cam_fps)
        fid          = fi + 1          # 1-based sequential mmpose frame_id

        print(f"Frame {idx_in_list+1}/{len(frame_indices)}: img={fi:05d} "
              f"pose_idx={pose_idx:05d} mmpose_fid={fid}")

        panel = process_frame(
            frame_idx    = fi,
            img_path     = img_path,
            pose_idx     = pose_idx,
            fid          = fid,
            cam_json     = ref_cam_json,
            all_cam_jsons= all_cam_raw,
            smplx_model  = smplx_model,
            smplx_poses  = smplx_poses,
            mhr_model    = mhr_model,
            mhr_poses    = mhr_poses,
            fid2kp       = fid2kp,
            skel         = skel,
            kpt_c        = kpt_c,
            lnk_c        = lnk_c,
            front_T      = front_T,
            front_R      = front_R,
            world_up     = world_up,
            kpts_3d_by_frame = kpts_3d_by_frame,
            n_rows       = args.n_rows,
            aug_trans    = args.aug_trans,
            aug_rot      = args.aug_rot,
            front_dist   = args.front_dist,
            alpha        = args.alpha,
            thr          = args.score_thr,
            device       = args.device,
        )

        out_path = out_dir / f"{fi:05d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved {out_path}")

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
