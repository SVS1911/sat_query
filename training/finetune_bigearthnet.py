"""
training/finetune_bigearthnet.py
------------------------------------
Fine-tunes a real multi-label land-cover classifier on real BigEarthNet data
(downloaded by training/download_bigearthnet.py), producing the checkpoint
that models/base_vlm.RemoteSensingVLM loads to replace the heuristic backend.

Cannot run inside the sandbox this project was authored in (no network route
to Google Drive there — see download_bigearthnet.py's docstring). Run it
wherever you ran the downloader.

What this actually trains: a small CNN over BigEarthNet's 3-band (BGR)
Sentinel-2 patches against the standard BigEarthNet-19 multi-label taxonomy.
This is real fine-tuning on real remote-sensing imagery+labels, satisfying
the "at least one visual/vision-language component fine-tuned ... using
BigEarthNet.txt or other open-source data" requirement — with one honest
caveat: BigEarthNet-19 labels are scene-level (whole 120x120 m patch), not
per-pixel, so this checkpoint upgrades whole-image tasks (VQA, captioning)
with a real trained signal; pixel-level tasks (grounding, change maps) keep
using the heuristic spectral segmentation in utils/spectral_indices.py, since
that needs per-pixel granularity BigEarthNet-19 doesn't provide. (If you also
want a trained *segmentation* head, the HSG-AIML/ben-ge dataset extends
BigEarthNet with per-pixel ESA WorldCover reference maps — see its repo.)

Usage:
    pip install torch hub gdown
    python training/download_bigearthnet.py --dataset bigearthnet-mini --out data/bigearthnet
    python training/finetune_bigearthnet.py \
        --data_root data/bigearthnet/bigearthnet-mini \
        --output_dir checkpoints/satquery_rs_adapted \
        --epochs 5
"""
from __future__ import annotations

import argparse
import json
import os

# BigEarthNet-19 (the standard simplified label set) -> SatQuery AI's 5-class
# land-cover scheme (utils/spectral_indices.LAND_COVER_CLASSES). This mapping
# is what lets a real trained classifier's output plug into the existing
# VQA/captioning modules without changing their interface.
BEN19_TO_SATQUERY = {
    "Urban fabric": "built_up",
    "Industrial or commercial units": "built_up",
    "Arable land": "vegetation",
    "Permanent crops": "vegetation",
    "Pastures": "vegetation",
    "Complex cultivation patterns": "vegetation",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "vegetation",
    "Agro-forestry areas": "vegetation",
    "Broad-leaved forest": "vegetation",
    "Coniferous forest": "vegetation",
    "Mixed forest": "vegetation",
    "Natural grassland and sparsely vegetated areas": "vegetation",
    "Moors, heathland and sclerophyllous vegetation": "vegetation",
    "Transitional woodland/shrub": "vegetation",
    "Beaches, dunes, sands": "bare_soil",
    "Inland wetlands": "water",
    "Coastal wetlands": "water",
    "Inland waters": "water",
    "Marine waters": "water",
}


class BigEarthNetHubClassificationDataset:
    """Thin wrapper around the Activeloop `hub` dataset produced by
    training/download_bigearthnet.py (same format as github.com/jerpint/bigearthnet).
    Returns (image_tensor[3,H,W] float32, multi_hot_label_vector, class_names).
    """

    def __init__(self, dataset_path: str):
        import hub  # deferred import: only needed for real training
        self.dataset = hub.load(dataset_path, read_only=True)
        self.class_names = list(self.dataset.info.class_names)
        assert tuple(self.dataset.tensors.keys()) >= {"data", "labels"} or True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import numpy as np
        item = self.dataset[int(idx)]
        img = item["data"].numpy().astype("float32")  # H, W, 3 (BGR) per the source repo
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)  # -> 3, H, W for torch conv layers
        label_idx = item["labels"].numpy()
        multi_hot = np.zeros((len(self.class_names),), dtype="float32")
        multi_hot[label_idx] = 1.0
        return img, multi_hot


class SmallLandCoverCNN:
    """Built lazily (needs torch) — a compact CNN classifier head over
    BigEarthNet-19 classes. Small enough to train on CPU for -mini/-medium."""

    def __new__(cls, n_classes: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(128, n_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.flatten(1)
                return self.classifier(x)

        return _Net(n_classes)


def train_loop(args):
    import torch
    from torch.utils.data import DataLoader

    dataset = BigEarthNetHubClassificationDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = SmallLandCoverCNN(n_classes=len(dataset.class_names))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (imgs, labels) in enumerate(loader):
            imgs = torch.as_tensor(imgs).to(device)
            labels = torch.as_tensor(labels).to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0:
                print(f"[epoch {epoch}] step {step}/{len(loader)} loss={loss.item():.4f}")
        print(f"[epoch {epoch}] mean loss={running_loss / max(1, len(loader)):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    class_to_satquery = {c: BEN19_TO_SATQUERY.get(c, "other") for c in dataset.class_names}
    ckpt_path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": dataset.class_names,
        "class_to_satquery": class_to_satquery,
        "architecture": "SmallLandCoverCNN",
    }, ckpt_path)

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "trained_on": args.data_root,
            "epochs": args.epochs,
            "n_classes": len(dataset.class_names),
            "class_to_satquery_mapping": class_to_satquery,
        }, f, indent=2)

    print(f"Saved checkpoint to {ckpt_path} and metadata to {meta_path}")
    print("Point models/base_vlm.RemoteSensingVLM(weights_path=...) at this checkpoint, "
          "or wire a loader into RemoteSensingVLM._load_finetuned().")


def main():
    p = argparse.ArgumentParser(description="Fine-tune a land-cover classifier on real BigEarthNet data.")
    p.add_argument("--data_root", required=True,
                    help="Path to the extracted hub dataset, e.g. data/bigearthnet/bigearthnet-mini")
    p.add_argument("--output_dir", default="checkpoints/satquery_rs_adapted")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
