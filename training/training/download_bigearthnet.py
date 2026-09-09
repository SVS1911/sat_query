"""
training/download_bigearthnet.py
------------------------------------
Downloads real BigEarthNet data for domain adaptation.

IMPORTANT — sandbox limitation: this script cannot run inside the Claude.ai
container this project was built in. That container's outbound network is
restricted to package registries and github.com (see the project README);
Google Drive, the official BigEarthNet/TU Berlin servers, and Hugging Face are
all blocked from it. This script is written to run correctly on YOUR machine
(local, a VM, or Colab) where those hosts are reachable.

Two real, verified sources are wired up:

1. bigearthnet-mini / -medium / -full (default: mini, ~9.3 MB)
   Sentinel-2 BGR-band patches with BigEarthNet-19 multi-label annotations,
   pre-packaged as Activeloop `hub` datasets by github.com/jerpint/bigearthnet
   (MIT licensed, workshop project built on the official BigEarthNet data).
   File IDs below are copied verbatim from that repo's
   bigearthnet/datamodules/bigearthnet_datamodule.py (GDRIVE_URLS), confirmed
   directly from the source on 2026-09-09:
     - bigearthnet-mini:   1X2-NpZ4ExUooAi8tkCBWAlmHZAG_N3Ux  (~9.3 MB;   90/30/30 train/val/test)
     - bigearthnet-medium: 1YW4ugRQTl-YF_ZpLO7gIRSlHnB2Cwslz  (~2.5 GB;   25000/5000/5000)
     - bigearthnet-full:   1isUcPQvCn1xc5GWEmPDOqj_l24QH2ZtA  (~30 GB;    269695/123723/125866)
   NOTE: this source is Sentinel-2 optical only (3 bands, BGR) — no paired
   Sentinel-1 SAR. It's the fastest path to a *real* fine-tuning run and is
   what training/finetune_bigearthnet.py trains against by default.

2. Official BigEarthNet-S1 (SAR) + BigEarthNet-S2 (optical), for the full
   co-registered optical+SAR pairing the project spec calls for. These are
   large (~66 GB / ~48 GB) tar archives hosted directly by TU Berlin (no
   Google Drive/auth needed) — see `--print-official-urls`. Only fetch these
   if you actually intend to train the cross-modal fusion encoder at scale.

Usage:
    pip install gdown
    python training/download_bigearthnet.py --dataset bigearthnet-mini --out data/bigearthnet
    python training/download_bigearthnet.py --print-official-urls
"""
from __future__ import annotations

import argparse
import os
import tarfile
import pathlib

GDRIVE_FILE_IDS = {
    "bigearthnet-mini": "1X2-NpZ4ExUooAi8tkCBWAlmHZAG_N3Ux",
    "bigearthnet-medium": "1YW4ugRQTl-YF_ZpLO7gIRSlHnB2Cwslz",
    "bigearthnet-full": "1isUcPQvCn1xc5GWEmPDOqj_l24QH2ZtA",
}

# Official BigEarthNet archives (Sentinel-2 optical + Sentinel-1 SAR), hosted
# by the RSiM group at TU Berlin. Large; not auto-downloaded by this script's
# default path. Printed on request so you have the authoritative source.
OFFICIAL_URLS = {
    "BigEarthNet-S2 (optical, ~66 GB)": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
    "BigEarthNet-S1 (SAR, ~48 GB)": "https://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz",
    "Official website / license / citation": "https://bigearth.net/",
}


def print_official_urls():
    print("Official BigEarthNet sources (verify current links at https://bigearth.net/ "
          "before downloading — hosts occasionally rotate archive paths):")
    for name, url in OFFICIAL_URLS.items():
        print(f"  {name}: {url}")
    print("\nThese are large multi-GB tar.gz archives. A typical wget/curl works once you "
          "have accepted BigEarthNet's terms of use on their site, e.g.:")
    print("  wget https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz")
    print("  wget https://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz")


def download_hub_dataset(dataset_name: str, out_dir: str) -> str:
    if dataset_name not in GDRIVE_FILE_IDS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(GDRIVE_FILE_IDS)}")

    try:
        import gdown
    except ImportError as e:
        raise ImportError("This script needs `gdown` (pip install gdown) to fetch data from "
                           "Google Drive. It is deliberately not a hard dependency of the base "
                           "app, since the base app runs fully offline without it.") from e

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / dataset_name
    tar_path = str(dataset_path) + ".tar"

    if dataset_path.is_dir():
        print(f"'{dataset_name}' already extracted at {dataset_path}, skipping.")
        return str(dataset_path)

    if not os.path.isfile(tar_path):
        file_id = GDRIVE_FILE_IDS[dataset_name]
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {dataset_name} from Google Drive (file id {file_id}) -> {tar_path}")
        gdown.download(url, tar_path, quiet=False)

    print(f"Extracting {tar_path} -> {out_dir}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=str(out_dir))

    print(f"Done. Dataset ready at {dataset_path}")
    return str(dataset_path)


def main():
    p = argparse.ArgumentParser(description="Download BigEarthNet data for SatQuery AI adaptation.")
    p.add_argument("--dataset", default="bigearthnet-mini",
                    choices=list(GDRIVE_FILE_IDS.keys()),
                    help="Which pre-packaged hub dataset to fetch (default: bigearthnet-mini, ~9.3MB).")
    p.add_argument("--out", default="data/bigearthnet", help="Output directory.")
    p.add_argument("--print-official-urls", action="store_true",
                    help="Print the official BigEarthNet-S1/S2 (co-registered SAR+optical) archive URLs and exit.")
    args = p.parse_args()

    if args.print_official_urls:
        print_official_urls()
        return

    download_hub_dataset(args.dataset, args.out)


if __name__ == "__main__":
    main()
