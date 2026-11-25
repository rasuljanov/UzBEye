import argparse
from pathlib import Path
from src.data.dataset import prepare


def main():
    p = argparse.ArgumentParser(description="Download UMD annotations and just the selected COCO images")
    p.add_argument("--root", type=Path, default=Path("data/refcocog"))
    p.add_argument("--limits", nargs=3, type=int, default=[4096, 256, 128], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    print(f"Prepared {len(prepare(args.root, args.limits, args.seed, args.workers))} expressions")


if __name__ == "__main__":
    main()
