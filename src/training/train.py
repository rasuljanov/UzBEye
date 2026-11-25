import argparse
from pathlib import Path

import torch
from src.common import config, read_json, seed_all
from src.models.grounding_dino import Grounder
from src.training.trainer import fit


def main(approach):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/"+("translate_train" if approach == "B" else "tahrirchi_encoder")+".yaml")
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--manifest", type=Path, help="Bilingual manifest path. Defaults to bilingual.json under --root.")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            p.error("--epochs must be positive")
        cfg["epochs"] = args.epochs
    if cfg["approach"] != approach:
        p.error("Wrong approach configuration")
    manifest = args.manifest or args.root/"bilingual.json"
    if not manifest.is_file():
        p.error(f"Missing {manifest}. Prepare the training translations first.")
    if Path(args.out).exists():
        p.error(f"Output directory already exists: {args.out}. Choose a new directory.")
    rows = read_json(manifest)
    device = torch.device(args.device)
    print(f"Loading Model {approach} on {device}...", flush=True)
    seed_all(cfg["seed"])
    try:
        model = Grounder(cfg, approach, device, training=True)
        fit(model, args.root, rows, args.out, cfg, approach)
    except torch.OutOfMemoryError as exc:
        raise SystemExit("CUDA ran out of memory. Lower image_size in a copy of the config and retry in a new run folder.") from exc
