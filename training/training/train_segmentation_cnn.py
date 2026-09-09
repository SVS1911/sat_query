"""
training/train_segmentation_cnn.py
--------------------------------------
Trains models.cnn_segmentation's U-Net-lite on a labeled image+mask manifest.
This is the script to use once you provide a real labeled dataset (satellite
imagery + per-pixel land-cover masks, or object-detection boxes rasterized
into a mask — either works as long as each pixel has a class index).

Manifest format (JSONL, one sample per line):
    {"image": "path/to/scene_0001.tif", "mask": "path/to/scene_0001_mask.tif"}
    {"image": "path/to/scene_0002.tif", "mask": "path/to/scene_0002_mask.tif"}
  - image: any format utils.image_io.load_image() supports (GeoTIFF/TIFF/PNG/JPEG)
  - mask: single-band raster, same H x W as image, integer values indexing
    into --class_names (default: utils.spectral_indices.LAND_COVER_CLASSES)

Usage:
    pip install torch
    python training/train_segmentation_cnn.py \
        --manifest data/my_dataset/train_manifest.jsonl \
        --output_dir checkpoints/satquery_segmentation \
        --epochs 20 --tile_size 256

Then, in app.py:
    from models.cnn_segmentation import SegmentationCNNBackend
    seg_backend = SegmentationCNNBackend("checkpoints/satquery_segmentation/model.pt")
    vlm = RemoteSensingVLM(segmentation_backend=seg_backend)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils.image_io import load_image
from utils.spectral_indices import LAND_COVER_CLASSES


class SegmentationManifestDataset:
    def __init__(self, manifest_path: str, tile_size: int = 256, in_channels: int = 3):
        self.samples = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.tile_size = tile_size
        self.in_channels = in_channels

    def __len__(self):
        return len(self.samples)

    def _resize(self, arr, size, is_mask=False):
        h, w = arr.shape[:2]
        ys = np.linspace(0, h - 1, size).astype(int)
        xs = np.linspace(0, w - 1, size).astype(int)
        out = arr[ys][:, xs]
        return out

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = load_image(sample["image"]).array  # H,W,C float32 in [0,1]
        mask_img = load_image(sample["mask"]).array
        mask = mask_img[..., 0] if mask_img.ndim == 3 else mask_img
        # mask was normalized to [0,1] by load_image's percentile stretch, which
        # is wrong for label rasters — reload raw values directly instead.
        mask = _load_raw_mask(sample["mask"])

        img = self._resize(img, self.tile_size)
        mask = self._resize(mask, self.tile_size)

        if img.shape[-1] < self.in_channels:
            img = np.pad(img, ((0, 0), (0, 0), (0, self.in_channels - img.shape[-1])), mode="edge")
        elif img.shape[-1] > self.in_channels:
            img = img[..., :self.in_channels]

        return img.transpose(2, 0, 1).astype(np.float32), mask.astype(np.int64)


def _load_raw_mask(path: str) -> np.ndarray:
    """Load a label raster WITHOUT the percentile normalization load_image()
    applies (that's meant for imagery display, not integer class labels)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            return tifffile.imread(path).astype(np.int64)
        except Exception:
            pass
    from PIL import Image
    return np.array(Image.open(path)).astype(np.int64)


def train_loop(args):
    import torch
    from torch.utils.data import DataLoader
    from models.cnn_segmentation import build_unet_lite

    class_names = args.class_names.split(",") if args.class_names else LAND_COVER_CLASSES

    dataset = SegmentationManifestDataset(args.manifest, tile_size=args.tile_size,
                                           in_channels=args.in_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = build_unet_lite(args.in_channels, len(class_names))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (imgs, masks) in enumerate(loader):
            imgs = torch.as_tensor(imgs).to(device)
            masks = torch.as_tensor(masks).to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 10 == 0:
                print(f"[epoch {epoch}] step {step}/{len(loader)} loss={loss.item():.4f}")
        print(f"[epoch {epoch}] mean loss={running_loss / max(1, len(loader)):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "in_channels": args.in_channels,
        "tile_size": args.tile_size,
    }, ckpt_path)
    print(f"Saved segmentation checkpoint to {ckpt_path}")
    print("Use it via: SegmentationCNNBackend(" + repr(ckpt_path) + ")")


def main():
    p = argparse.ArgumentParser(description="Train a pixel-level land-cover segmentation CNN.")
    p.add_argument("--manifest", required=True, help="JSONL manifest of {image, mask} pairs.")
    p.add_argument("--output_dir", default="checkpoints/satquery_segmentation")
    p.add_argument("--class_names", default=None,
                    help="Comma-separated class list matching your mask's integer labels "
                         "(default: SatQuery's built-in water,vegetation,built_up,bare_soil,other).")
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--tile_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
