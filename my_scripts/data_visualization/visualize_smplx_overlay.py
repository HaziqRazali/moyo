#!/usr/bin/env python3
"""
MOYO dataset visualizer: renders SMPL-X mesh overlay on camera images.

Usage:
    conda activate pytorch_env
    python visualize_smplx_overlay.py --seq 220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a --n_frames -1
        [--cam YOGI_Cam_01]       # if omitted, processes all cameras
        [--n_frames 50]           # frames to render (-1 = all)
        [--use_mosh]              # use mosh pkl instead of YOGI_2 npz
        [--alpha 0.7]             # mesh blend opacity
        [--no_video]              # skip MP4 generation

Data notes:
  - Camera positions are in mm; SMPL-X trans is in meters → divide cam pos by 1000
    - YOGI_2 @ 60 fps, images @ ~30 fps → frame mapping uses embedded frame number in filename
  - Camera model: single focal (fx=fy), cx/cy princpt, R + position (world center)
  - t = -R @ (position / 1000)   (position in mm → meters)
"""

import argparse
import json
import os
import pickle
import sys

import re

import cv2
import numpy as np
import torch
import trimesh
import pyrender
import smplx
from pathlib import Path

# Matches image stems ending in _XXXX (4-digit frame number)
_FRAME_RE = re.compile(r'_(\d{4})$')

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
ROOT = Path("/data/haziq/moyo/data")
SMPLX_MODEL_PATH = "/data/haziq/mocap/data/models_smplx_v1_1/models"

# Required for headless (no display) rendering
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def seq_to_date(seq_name: str) -> str:
    """220923_... → '20220923', 221004_... → '20221004'"""
    day = seq_name[:6]          # e.g. "220923"
    return "20" + day[:2] + day[2:]   # "20220923"


def cam_name_to_key(cam_name: str) -> str:
    """YOGI_Cam_01 → 'cam_1', YOGI_Cam_08 → 'cam_8'"""
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
                # Only return frame images (stem ends in _XXXX); skip composites like _full.jpg
                return sorted(f for f in p.glob("*.jpg") if _FRAME_RE.search(f.stem))
    return []


def find_seq_cameras(seq_name: str):
    """Find all available camera names for a sequence."""
    cameras = set()
    for split in ("20220923_20220926_with_hands", "20221004_with_com"):
        for tv in ("train", "val"):
            seq_dir = ROOT / "images" / split / tv / seq_name
            if seq_dir.exists():
                for cam_dir in seq_dir.iterdir():
                    if cam_dir.is_dir():
                        cameras.add(cam_dir.name)
    return sorted(cameras)


def find_pose_file(seq_name: str, prefer_yogi2: bool = True):
    """Returns (Path, 'yogi2' | 'mosh') or (None, None)."""
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


def load_poses(path: Path, pose_type: str) -> dict:
    """Returns a dict with standardized keys."""
    if pose_type == "yogi2":
        d = np.load(path, allow_pickle=True)
        n_betas = int(d["num_betas"])
        return {
            "root_orient":      d["root_orient"],          # (N, 3)
            "body_pose":        d["pose_body"],             # (N, 63)
            "left_hand_pose":   d["pose_hand"][:, :45],    # (N, 45)
            "right_hand_pose":  d["pose_hand"][:, 45:],    # (N, 45)
            "jaw_pose":         d["pose_jaw"],              # (N, 3)
            "leye_pose":        d["pose_eye"][:, :3],      # (N, 3)
            "reye_pose":        d["pose_eye"][:, 3:],      # (N, 3)
            "trans":            d["trans"],                 # (N, 3) in meters
            "betas":            d["betas"][:n_betas],      # (n_betas,)
            "n_betas":          n_betas,
            "fps":              float(d["mocap_frame_rate"]),
            "source":           "yogi2",
        }
    else:  # mosh pkl
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        fp = d["fullpose"]  # (N, 165)
        return {
            "root_orient":      fp[:, :3],
            "body_pose":        fp[:, 3:66],
            "jaw_pose":         fp[:, 66:69],
            "leye_pose":        fp[:, 69:72],
            "reye_pose":        fp[:, 72:75],
            "left_hand_pose":   fp[:, 75:120],
            "right_hand_pose":  fp[:, 120:165],
            "trans":            d["trans"],                 # (N, 3) in meters
            "betas":            d["betas"][:10],
            "n_betas":          10,
            "fps":              60.0,
            "source":           "mosh",
        }


def get_camera_params(cam: dict, intr_scale: float = 0.5):
    """
    Returns R (3×3), t (3,), focal, cx, cy.

    Camera calibration was done at native resolution (4112×3008).
    Images are stored at half resolution (2056×1504), so all
    intrinsics must be multiplied by intr_scale=0.5.

    cam["rotation"]: R  (world→camera, OpenCV convention)
    cam["position"]: camera center in world frame, IN MM
    t = -R @ (position / 1000)   (mm → meters, matching SMPL-X trans units)
    """
    R = np.array(cam["rotation"], dtype=np.float64)       # (3, 3)
    C_mm = np.array(cam["position"], dtype=np.float64)    # mm
    C_m = C_mm / 1000.0                                   # → meters
    t = -R @ C_m                                          # (3,)
    focal = float(cam["focal"]) * intr_scale
    cx    = float(cam["princpt"][0]) * intr_scale
    cy    = float(cam["princpt"][1]) * intr_scale
    return R, t, focal, cx, cy


def build_smplx_model(n_betas: int, device: str):
    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        num_betas=min(n_betas, 300),
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    )
    return model.to(device)


def get_smplx_verts(model, poses: dict, frame_idx: int, device: str):
    """Run one-frame smplx forward pass. Returns vertices (V, 3) numpy."""
    def to_tensor(key):
        return torch.tensor(
            poses[key][frame_idx : frame_idx + 1], dtype=torch.float32
        ).to(device)

    n_betas = model.num_betas
    betas_np = poses["betas"][:n_betas]
    betas_t = torch.tensor(betas_np, dtype=torch.float32).unsqueeze(0).to(device)

    output = model(
        global_orient=to_tensor("root_orient"),
        body_pose=to_tensor("body_pose"),
        left_hand_pose=to_tensor("left_hand_pose"),
        right_hand_pose=to_tensor("right_hand_pose"),
        jaw_pose=to_tensor("jaw_pose"),
        leye_pose=to_tensor("leye_pose"),
        reye_pose=to_tensor("reye_pose"),
        betas=betas_t,
        transl=to_tensor("trans"),
        return_verts=True,
    )
    return output.vertices[0].detach().cpu().numpy()  # (10475, 3)


def render_overlay(img_rgb: np.ndarray, verts: np.ndarray, faces: np.ndarray,
                   R: np.ndarray, t: np.ndarray,
                   focal: float, cx: float, cy: float,
                   alpha: float = 0.7) -> np.ndarray:
    """
    Render SMPL-X mesh and composite it onto img_rgb.
    Returns blended RGB image.
    """
    h, w = img_rgb.shape[:2]

    mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.6,
        baseColorFactor=(0.65, 0.74, 0.86, 1.0),   # soft blue
    )
    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri, material=material, smooth=True)

    scene = pyrender.Scene(ambient_light=(0.4, 0.4, 0.4), bg_color=(0, 0, 0, 0))
    scene.add(mesh_pr)

    # World-to-camera [R|t] → camera-to-world (c2w) for pyrender
    Rt44 = np.eye(4)
    Rt44[:3, :3] = R
    Rt44[:3, 3] = t
    c2w = np.linalg.inv(Rt44)

    # OpenCV → OpenGL: flip Y and Z columns
    c2w_gl = c2w.copy()
    c2w_gl[:, 1] *= -1
    c2w_gl[:, 2] *= -1

    camera = pyrender.IntrinsicsCamera(
        fx=focal, fy=focal, cx=cx, cy=cy, znear=0.05, zfar=50.0
    )
    scene.add(camera, pose=c2w_gl)

    # Key + fill lights in camera space
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=c2w_gl)
    fill_pose = c2w_gl @ np.diag([-1, 1, -1, 1]).astype(float)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    # Composite where mesh is visible
    mask = depth > 0
    out = img_rgb.copy()
    out[mask] = (
        alpha * color_rgba[mask, :3].astype(np.float32)
        + (1.0 - alpha) * img_rgb[mask].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return out


def process_camera(seq_name: str, cam_name: str, poses: dict, model, cam_data: dict,
                   out_base: Path, args) -> None:
    """Process one camera: render frames and optionally create video."""
    # ── Images ──────────────────────────────────────────────
    img_paths = find_seq_images(seq_name, cam_name)
    if not img_paths:
        print(f"  WARNING: No images found for cam={cam_name}")
        return
    print(f"[{cam_name}]    {len(img_paths)} frames  →  {img_paths[0].parent}")

    # ── Camera ──────────────────────────────────────────────
    cam_key = cam_name_to_key(cam_name)
    if cam_key not in cam_data:
        print(f"  WARNING: {cam_key} not found in cameras_param.json")
        return
    cam = cam_data[cam_key]
    R, t, focal, cx, cy = get_camera_params(cam, intr_scale=0.5)
    print(f"           focal={focal:.1f}px  cx={cx:.1f}  cy={cy:.1f}  t_z={t[2]:.3f}m")

    # ── Frame selection ──────────────────────────────────────
    n_imgs = len(img_paths)
    n_pose = len(poses["trans"])
    mocap_fps = poses["fps"]
    # Infer camera fps from mocap duration and image count (≈30fps for MOYO)
    mocap_duration = n_pose / mocap_fps
    cam_fps = round(n_imgs / mocap_duration) if mocap_duration > 0 else 30
    if cam_fps <= 0:
        cam_fps = 30
    print(f"           mocap={mocap_fps:.0f}fps  cam≈{cam_fps}fps  ratio={mocap_fps/cam_fps:.2f}x")

    if args.n_frames == -1 or args.n_frames >= n_imgs:
        img_indices = list(range(n_imgs))
    else:
        img_indices = np.linspace(0, n_imgs - 1, args.n_frames, dtype=int).tolist()
    print(f"           rendering {len(img_indices)} frames...\n")

    out_dir = out_base / seq_name / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    faces = model.faces
    for i, img_idx in enumerate(img_indices):
        img_path = img_paths[img_idx]
        # Use the frame number embedded in the image filename for precise temporal
        # alignment. Image filenames encode the synchronized capture frame index,
        # e.g. _0005.jpg = frame 5 at cam_fps → mocap frame round(5 * 60 / 30) = 10.
        m = _FRAME_RE.search(img_path.stem)
        if m:
            frame_number = int(m.group(1))
            pose_idx = int(round(frame_number * mocap_fps / cam_fps))
        else:
            # Fallback: proportional mapping
            pose_idx = int(round(img_idx * n_pose / n_imgs))
        pose_idx = min(max(pose_idx, 0), n_pose - 1)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: could not read {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        verts = get_smplx_verts(model, poses, pose_idx, args.device)
        overlay_rgb = render_overlay(img_rgb, verts, faces, R, t, focal, cx, cy, alpha=args.alpha)

        out_path = out_dir / f"frame_{img_idx:05d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(img_indices):
            print(f"  [{i+1:>4}/{len(img_indices)}]  img={img_idx:05d}  pose={pose_idx:05d}")

    print(f"  Saved {len(saved)} frames to {out_dir.name}/\n")

    # ── Video ─────────────────────────────────────────────────
    if not args.no_video and saved:
        try:
            first = cv2.imread(str(saved[0]))
            h, w = first.shape[:2]
            video_path = out_base / f"{seq_name}__{cam_name}__overlay.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25, (w, h)
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
        description="Render SMPL-X mesh overlay on MOYO images."
    )
    parser.add_argument("--seq", required=True,
                        help="Sequence name, e.g. 220923_yogi_body_hands_03596_Boat_Pose_...-a")
    parser.add_argument("--cam", default=None,
                        help="Camera name (default: all available)")
    parser.add_argument("--out_dir", default="./visualize_overlay",
                        help="Output directory for frames + video")
    parser.add_argument("--n_frames", type=int, default=30,
                        help="Number of frames to render evenly spaced (-1 = all)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Mesh opacity (0=transparent, 1=opaque)")
    parser.add_argument("--use_mosh", action="store_true",
                        help="Use mosh pkl instead of YOGI_2 npz")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip MP4 generation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_base = Path(args.out_dir)

    # ── Poses (loaded once, shared across cameras) ──────────
    pose_path, pose_type = find_pose_file(args.seq, prefer_yogi2=not args.use_mosh)
    if pose_path is None:
        print(f"ERROR: No pose file (YOGI_2 npz or mosh pkl) found for {args.seq}")
        sys.exit(1)
    poses = load_poses(pose_path, pose_type)
    print(f"[poses]    {len(poses['trans'])} frames @ {poses['fps']} fps  ({pose_type})")

    # ── SMPL-X model (loaded once, shared across cameras) ────
    print(f"[smplx]    loading  n_betas={min(poses['n_betas'], 300)}  device={args.device}")
    model = build_smplx_model(poses["n_betas"], args.device)

    # ── Determine cameras ───────────────────────────────────
    if args.cam is None:
        cameras = find_seq_cameras(args.seq)
        if not cameras:
            print(f"ERROR: No cameras found for seq={args.seq}")
            sys.exit(1)
        print(f"[cameras]  {len(cameras)}: {', '.join(cameras)}\n")
    else:
        cameras = [args.cam]

    # ── Load all camera parameters ───────────────────────────
    date = seq_to_date(args.seq)
    cam_data = load_cameras(date)

    # ── Process each camera ──────────────────────────────────
    for cam_name in cameras:
        process_camera(args.seq, cam_name, poses, model, cam_data, out_base, args)


if __name__ == "__main__":
    main()
