"""Dataset loading utilities for the windowed peg-in-hole data.

Each data point is one window w:
  image          : (3, H, W)  float32 — RGB frame captured at end of window w,
                               normalized with ImageNet mean/std (matches pretrained ResNet18)
  proprio_window : (k*12,)    float32 — z-scored joint pos/vel, flattened across k timesteps
  f_window       : (3,)       float32 — force direction unit vector (NOT z-scored)
  c_window       : int        0 or 1 — contact gate

Episode-level splits are done to avoid trajectory leakage across train/val.
"""

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_ROOT = "dataset"
SEED = 42

# Split fractions (mirrors utils.py from psets)
EVAL_FRAC = 0.20
TRAIN_NOISY_FRAC = 0.30

# ImageNet normalisation constants (for pretrained ResNet18)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Episode discovery ──────────────────────────────────────────────────────────

def collect_episodes(dataset_root: str = DATASET_ROOT) -> List[Dict]:
    """Return a list of episode dicts for every episode that has windowed data."""
    episodes = []
    for ep_name in sorted(os.listdir(dataset_root)):
        ep_path = os.path.join(dataset_root, ep_name)
        meta_path = os.path.join(ep_path, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)

        proprio_path = os.path.join(ep_path, "proprio_windows.npy")
        force_path   = os.path.join(ep_path, "force_directions.npy")
        c_path       = os.path.join(ep_path, "c_windows.npy")
        rgb_dir      = os.path.join(ep_path, "rgb")

        if not all(os.path.isfile(p) for p in [proprio_path, force_path, c_path]):
            continue
        if not os.path.isdir(rgb_dir):
            continue

        episodes.append({
            "path":         ep_path,
            "is_noisy":     meta["is_noisy"],
            "n_windows":    meta["n_windows"],
            "proprio_path": proprio_path,
            "force_path":   force_path,
            "c_path":       c_path,
            "rgb_dir":      rgb_dir,
        })

    n_clean = sum(not e["is_noisy"] for e in episodes)
    n_noisy = sum(e["is_noisy"] for e in episodes)
    print(f"Discovered {len(episodes)} episodes ({n_clean} clean, {n_noisy} noisy)")
    return episodes


# ── Global statistics (proprio only) ──────────────────────────────────────────

def compute_window_stats(episodes: List[Dict]) -> Dict[str, np.ndarray]:
    """Compute mean/std of proprio_windows over all episodes and all k timesteps.

    proprio_windows has shape (N, k, 12); we collapse across N and k.
    F_window (force_directions) is already a unit vector — not z-scored.
    """
    P_list = []
    for ep in tqdm(episodes, desc="Computing proprio stats"):
        pw = np.load(ep["proprio_path"]).astype(np.float32)  # (N, k, 12)
        P_list.append(pw.reshape(-1, pw.shape[-1]))            # (N*k, 12)

    P_all = np.concatenate(P_list, axis=0)   # (total_timesteps, 12)
    mean_p = P_all.mean(axis=0)              # (12,)
    std_p  = P_all.std(axis=0)              # (12,)
    std_p[std_p == 0] = 1.0                  # guard zero-variance dims

    print(f"Proprio — mean: {mean_p.round(3)}")
    print(f"          std : {std_p.round(3)}")
    return {"mean_p": mean_p, "std_p": std_p}


# ── Episode-level train / val / eval splits ────────────────────────────────────

def split_episodes(
    episodes: List[Dict],
    eval_frac: float = EVAL_FRAC,
    train_noisy_frac: float = TRAIN_NOISY_FRAC,
    val_frac_of_train: float = 0.20,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split episodes into train, val, and eval sets at the episode level.

    Eval is 100% noisy.  Train is split into an inner train/val at episode
    level (80/20 by default) to avoid trajectory leakage across windows from
    the same episode.

    Returns
    -------
    train_eps, val_eps, eval_eps
    """
    clean_eps = [e for e in episodes if not e["is_noisy"]]
    noisy_eps = [e for e in episodes if e["is_noisy"]]

    rng = random.Random(seed)
    rng.shuffle(clean_eps)
    rng.shuffle(noisy_eps)

    n_total = len(episodes)
    n_eval  = round(eval_frac * n_total)
    n_train_total = n_total - n_eval
    n_train_noisy = round(train_noisy_frac * n_train_total)
    n_train_clean = n_train_total - n_train_noisy

    eval_eps  = noisy_eps[:n_eval]
    candidate = clean_eps[:n_train_clean] + noisy_eps[n_eval:n_eval + n_train_noisy]

    # Episode-level val split within training candidates
    rng.shuffle(candidate)
    n_val = max(1, round(val_frac_of_train * len(candidate)))
    val_eps   = candidate[:n_val]
    train_eps = candidate[n_val:]

    print(
        f"\nSplit assignment (episode level):\n"
        f"  Train : {len(train_eps)} eps "
        f"({sum(not e['is_noisy'] for e in train_eps)} clean, "
        f"{sum(e['is_noisy'] for e in train_eps)} noisy)\n"
        f"  Val   : {len(val_eps)} eps "
        f"({sum(not e['is_noisy'] for e in val_eps)} clean, "
        f"{sum(e['is_noisy'] for e in val_eps)} noisy)\n"
        f"  Eval  : {len(eval_eps)} eps (100% noisy)"
    )
    return train_eps, val_eps, eval_eps


# ── Dataset ────────────────────────────────────────────────────────────────────

class WindowedRoboticsDataset(Dataset):
    """One sample = one window w of one episode.

    Returns a dict with:
        image          : (3, H, W)  float32, ImageNet-normalised
        proprio_window : (k*12,)    float32, z-scored
        f_window       : (3,)       float32, raw unit vector
        c_window       : int64      0 or 1
    """

    def __init__(self, episodes: List[Dict], stats: Dict[str, np.ndarray]):
        self._records: list = []  # (frame_path, proprio_flat, f_window, c_window)
        self.n_contact: int = 0   # number of contact-gated windows in this split

        mean_p = stats["mean_p"]  # (12,)
        std_p  = stats["std_p"]   # (12,)

        for ep in episodes:
            pw = np.load(ep["proprio_path"]).astype(np.float32)   # (N, k, 12)
            fd = np.load(ep["force_path"]).astype(np.float32)     # (N, 3)
            cw = np.load(ep["c_path"]).astype(np.int64)           # (N,)
            N  = pw.shape[0]
            k  = pw.shape[1]

            # z-score proprio: broadcast over (N, k, 12)
            pw_norm = (pw - mean_p) / std_p   # (N, k, 12)
            pw_flat = pw_norm.reshape(N, k * 12)  # (N, 60)

            for w in range(N):
                frame_path = os.path.join(ep["rgb_dir"], f"frame_{w:03d}.png")
                c = int(cw[w])
                self._records.append((
                    frame_path,
                    pw_flat[w].copy(),   # (60,)
                    fd[w].copy(),        # (3,)
                    c,
                ))
                self.n_contact += c

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_path, proprio_flat, f_window, c_window = self._records[idx]

        # Load RGB, convert to float32 in [0, 1], then ImageNet-normalise
        img = np.array(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD     # (H, W, 3)
        img_t = torch.from_numpy(img).permute(2, 0, 1) # (3, H, W)

        return {
            "image":          img_t,
            "proprio_window": torch.from_numpy(proprio_flat),
            "f_window":       torch.from_numpy(f_window),
            "c_window":       torch.tensor(c_window, dtype=torch.int64),
        }
