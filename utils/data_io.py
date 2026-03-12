"""Save / load episode data in the canonical directory layout."""

import json
import os

import cv2
import numpy as np


def save_episode(
    base_dir: str,
    episode_id: int,
    rgb_frames: list[np.ndarray],
    proprio: np.ndarray,
    force: np.ndarray,
    metadata: dict,
) -> None:
    """Persist one episode to disk.

    Layout::

        base_dir/episode_XXXX/
            rgb/frame_000.png  …  frame_015.png
            proprio.npy   [T, 12]
            force.npy     [T, 6]
            metadata.json
    """
    ep_dir = os.path.join(base_dir, f"episode_{episode_id:04d}")
    rgb_dir = os.path.join(ep_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    for i, frame in enumerate(rgb_frames):
        path = os.path.join(rgb_dir, f"frame_{i:03d}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    np.save(os.path.join(ep_dir, "proprio.npy"), proprio.astype(np.float32))
    np.save(os.path.join(ep_dir, "force.npy"), force.astype(np.float32))

    with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_episode(ep_dir: str) -> dict:
    """Load an episode directory back into memory."""
    proprio = np.load(os.path.join(ep_dir, "proprio.npy"))
    force = np.load(os.path.join(ep_dir, "force.npy"))

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
        "proprio": proprio,
        "force": force,
        "metadata": metadata,
        "rgb": rgb_frames,
    }
