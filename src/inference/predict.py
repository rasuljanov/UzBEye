import argparse
from pathlib import Path

import torch
from PIL import Image

from src.common import config, peak_mib, read_json, seed_all
from src.inference.translation_backend import backend
from src.models.grounding_dino import Grounder, load_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("prompt")
    p.add_argument("--approach", choices=list("ABCD"), default="D")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--run", type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    adapted = args.approach in ("B","C")
    if adapted and not args.run:
        p.error("B/C require --run with a trained delta")
    cfg = read_json(args.run/"training.json")["config"] if adapted else config(args.config)
    seed_all(cfg["seed"])
    device = torch.device(args.device)
    model = Grounder(cfg, args.approach, device)
    if adapted:
        load_delta(model, args.run/"best.pt", args.approach)
    prompt = args.prompt
    if args.approach == "A":
        prompt = backend(cfg, device).translate(prompt, "uz", "en")
    with Image.open(args.image) as im:
        boxes, scores = model.predict(im.convert("RGB"), prompt)
    import json
    print(json.dumps(dict(prompt=prompt, normalized_xyxy=boxes, scores=scores,
                          peak_allocated_mib=peak_mib(device)), ensure_ascii=False))


if __name__ == "__main__":
    main()
