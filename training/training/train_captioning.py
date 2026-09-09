"""Train the lightweight dataset-backed caption model.

Example:
    python training/train_captioning.py --dataset data/captioning --output models/captioning.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.captioning import CaptionModel, load_caption_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CPU-friendly image caption retrieval model")
    parser.add_argument("--dataset", required=True, help="Directory containing train.csv and images")
    parser.add_argument("--output", default="models/captioning.json", help="Checkpoint JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for a smoke run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_caption_records(args.dataset, split="train", limit=args.limit)
    model = CaptionModel.train(records, seed=args.seed)
    model.save(args.output)
    print(json.dumps({"records": len(records), "checkpoint": str(Path(args.output)), "backend": "dataset-nearest-neighbor"}))


if __name__ == "__main__":
    main()
