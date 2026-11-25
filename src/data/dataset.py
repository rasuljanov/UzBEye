"""One UMD RefCOCOg expression per row. Coordinates stay in original pixels."""
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

REVISION = "55319436ad54b9480cefda6b9d64397de92456dd"
SHARDS = {
    "train": "train-00000-of-00001-4fe3e6340cfb69ed.parquet",
    "val": "validation-00000-of-00001-15168dfe7b5961e5.parquet",
    "test": "test-00000-of-00001-2316f36b19cd7f72.parquet",
}


def fetch(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response, tmp.open("wb") as output:
                shutil.copyfileobj(response, output)
            tmp.replace(path)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == 2 or isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404):
                raise
            sleep(2 ** attempt)


def flatten(refs, split):
    for ref in refs:
        info = json.loads(ref["raw_image_info"])
        ann = json.loads(ref["raw_anns"])
        iid = ref["image_id"]
        if (ref["split"] != split or info["id"] != iid or ann["image_id"] != iid
                or ann["id"] != ref["ann_id"]):
            raise ValueError("Inconsistent RefCOCOg join or split")
        x, y, w, h = ann["bbox"]
        if min(w, h) <= 0 or x < 0 or y < 0 or x+w > info["width"]+1 or y+h > info["height"]+1:
            raise ValueError(f"Invalid COCO box for annotation {ann['id']}")
        for sent in ref["sentences"]:
            yield dict(id=sent["sent_id"], ref_id=ref["ref_id"], ann_id=ann["id"], image_id=iid,
                       image=f"COCO_train2014_{iid:012d}.jpg", split=split, en=sent["raw"].strip(),
                       bbox=[x,y,w,h], width=info["width"], height=info["height"],
                       license_id=info.get("license"))


def subset(rows, seed, limit):
    if limit < 0:
        raise ValueError("Use 0 for the full split")
    ranked = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}/{r['id']}".encode()).digest())
    return ranked[:limit] if limit else ranked


def prepare(root, limits, seed, workers=8):
    import pyarrow.parquet as pq
    from PIL import Image
    from src.common import digest, write_json
    image_splits, ids = {}, set()
    selected = []
    split_stats = {}
    for (split, shard), limit in zip(SHARDS.items(), limits, strict=True):
        path = root / "raw" / shard
        fetch(f"https://huggingface.co/datasets/jxu124/refcocog/resolve/{REVISION}/data/{shard}", path)
        rows = list(flatten(pq.read_table(path).to_pylist(), split))
        for row in rows:
            if image_splits.setdefault(row["image_id"], split) != split or row["id"] in ids:
                raise ValueError("Image leakage or duplicate expression")
            ids.add(row["id"])
        sample = subset(rows, seed, limit)
        with path.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        split_stats[split] = dict(source_expressions=len(rows), selected=len(sample),
                                 selected_images=len({r["image_id"] for r in sample}), shard_sha256=sha)
        selected.extend(sample)
    def get_image(r):
        path = root / "images" / r["image"]
        fetch("https://s3.amazonaws.com/images.cocodataset.org/train2014/" + r["image"], path)
        with Image.open(path) as image:
            if image.size != (r["width"], r["height"]):
                raise ValueError(f"Image metadata mismatch: {path}")
            image.verify()

    by_image = {r["image_id"]: r for r in selected}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(get_image, by_image.values()):
            pass
    write_json(root / "english.json", selected)
    write_json(root / "selection.json", dict(dataset="RefCOCOg UMD", revision=REVISION, seed=seed,
                                             limits=limits, expressions=len(selected), splits=split_stats,
                                             english_digest=digest(selected)))
    return selected


def open_image(root, row):
    from PIL import Image
    with Image.open(Path(root) / "images" / row["image"]) as image:
        return image.convert("RGB")


def training_input_hash(rows):
    from src.common import digest

    fields = ("id", "image", "bbox", "width", "height", "uz")
    return digest([
        [row[field] for field in fields] + [split]
        for split in ("train", "val") for row in rows if row["split"] == split
    ])
