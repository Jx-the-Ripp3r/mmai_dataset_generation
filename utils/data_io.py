"""Save / load episode data in the canonical directory layout."""

import json
import os

import cv2
import numpy as np


def save_episode(
    base_dir: str,
    episode_id: int,
    rgb_frames: list[np.ndarray],
    proprio_windows: np.ndarray,
    force_directions: np.ndarray,
    c_windows: np.ndarray,
    metadata: dict,
) -> None:
    """Persist one episode to disk.

    Layout::

        base_dir/episode_XXXX/
            rgb/frame_000.png  …  frame_N-1.png  (one per window, end-of-window)
            proprio_windows.npy   [N, k, proprio_dim]  float32
            force_directions.npy  [N, 3]               float32
            c_windows.npy         [N,]                 int8
            metadata.json

    frame_i corresponds to window i: the image is captured at the last
    step of each k-step window, after the force/proprio data in that window.
    """
    ep_dir = os.path.join(base_dir, f"episode_{episode_id:04d}")
    rgb_dir = os.path.join(ep_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    for i, frame in enumerate(rgb_frames):
        path = os.path.join(rgb_dir, f"frame_{i:03d}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    np.save(os.path.join(ep_dir, "proprio_windows.npy"),
            proprio_windows.astype(np.float32))
    np.save(os.path.join(ep_dir, "force_directions.npy"),
            force_directions.astype(np.float32))
    np.save(os.path.join(ep_dir, "c_windows.npy"),
            c_windows.astype(np.int8))

    with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_episode(ep_dir: str) -> dict:
    """Load an episode directory back into memory."""
    proprio_windows = np.load(os.path.join(ep_dir, "proprio_windows.npy"))
    force_directions = np.load(os.path.join(ep_dir, "force_directions.npy"))
    c_windows = np.load(os.path.join(ep_dir, "c_windows.npy"))

    with open(os.path.join(ep_dir, "metadata.json")) as f:
        metadata = json.load(f)

    rgb_dir = os.path.join(ep_dir, "rgb")
    rgb_files = sorted(
        f for f in os.listdir(rgb_dir) if f.endswith(".png")
    )
    rgb_frames = [
        cv2.cvtColor(cv2.imread(os.path.join(rgb_dir, f)), cv2.COLOR_BGR2RGB)
        for f in rgb_files
    ]

    return {
        "proprio_windows": proprio_windows,
        "force_directions": force_directions,
        "c_windows": c_windows,
        "metadata": metadata,
        "rgb": rgb_frames,
    }
