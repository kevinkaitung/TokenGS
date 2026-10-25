"""Per-view image dumps shared by TokenGS's evaluate.py and exps/eval_{insitunet,3dgs}.py.

Kept free of matplotlib / einops / py3.10-only syntax so the gs3d and insitunet envs can import it.
"""

import os

import imageio
import torch


def save_views(images: torch.Tensor, out_dir: str, names: list) -> None:
    """Save each view of a (v, c, h, w) tensor in ~[0, 1] as its own PNG, <out_dir>/<names[i]>.png."""
    assert images.ndim == 4 and len(names) == images.shape[0], f"{tuple(images.shape)} vs {len(names)} names"
    os.makedirs(out_dir, exist_ok=True)
    images = (images.detach().float().clamp(0, 1) * 255).round().to(torch.uint8)
    for name, image in zip(names, images.permute(0, 2, 3, 1).cpu().numpy()):
        imageio.imwrite(os.path.join(out_dir, f"{name}.png"), image)
