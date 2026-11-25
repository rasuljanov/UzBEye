"""Prepare a reviewable pilot, then build a bilingual manifest with stable IDs."""
import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import torch
from src.common import config, digest, read_json, sync, write_json
from src.inference.translation_backend import Cache, InvalidTranslation, backend
from src.data.preprocessing import normalize


def repeated_phrase(text):
    words = re.findall(r"\w+(?:'\w+)*", text.casefold())
    for length in range(1, 7):
        for start in range(len(words) - 5*length + 1):
            phrase = words[start:start+length]
            if phrase * 5 == words[start:start+5*length]:
                return True
    return False


def apply_reviews(rows, reviews):
    by_id = {r["id"]: r for r in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("Duplicate review IDs")
    known = {r["id"] for r in rows}
    if by_id.keys() - known:
        raise ValueError("Review contains unknown expressions")
    for r in rows:
        if r["id"] in by_id:
            v = by_id[r["id"]]
            if any(r[k] != v[k] for k in ("image_id", "en", "mt", "bbox", "split")):
                raise ValueError("Review source differs from manifest")
            if v["status"] == "accepted":
                if not v["reviewer"].strip() or not normalize(v["uz"]):
                    raise ValueError("Accepted reviews require a reviewer and Uzbek text")
                corrected = normalize(v["uz"])
                r.update(uz=corrected if r["split"] == "test" else r["mt"], review_text=corrected, review_status="accepted",
                         reviewed=True, reviewer=v["reviewer"], tags=v.get("tags", []))
            elif v["status"] in ("pending", "rejected"):
                if v["status"] == "rejected" and not (v["reviewer"].strip() and v.get("reason", "").strip()):
                    raise ValueError("A rejected reference needs a reviewer and reason")
                r.update(reviewed=False, uz=r["mt"], review_status=v["status"])
                if v["status"] == "rejected":
                    r.update(reviewer=v["reviewer"], reason=v["reason"])
            else:
                raise ValueError("Unknown review status")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", choices=["pilot", "training", "build", "review"], required=True,
                   help="Use training to translate only train/val after inspecting pilot quality")
    p.add_argument("--reviews", type=Path)
    p.add_argument("--batch-size", type=int, default=1, help="Translation batch size for local NLLB")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    root = args.root
    if args.stage == "review":
        rows = apply_reviews(read_json(root/"bilingual.json"), read_json(args.reviews or root/"review.json"))
        write_json(root/"bilingual.json", rows)
        return
    rows = read_json(root/"english.json")
    pilot_ids = {r["id"] for split in ("train", "val", "test")
                 for r in [x for x in rows if x["split"] == split][:8]}
    if args.stage == "pilot":
        rows = [r for r in rows if r["id"] in pilot_ids]
    elif args.stage == "training":
        rows = [r for r in rows if r["split"] in ("train", "val")]
    elif len(rows) > len(pilot_ids):
        reviews = read_json(args.reviews or root/"review.json")
        if not pilot_ids <= {r["id"] for r in reviews if r["status"] in ("accepted","rejected") and r["reviewer"].strip()}:
            raise ValueError("Review the pilot against its images before scaling translation")
    device = torch.device(args.device)
    model = backend(config(args.config), device)
    if args.batch_size > 1:
        if not callable(getattr(model, "translate_many", None)):
            p.error("This translation backend only supports --batch-size 1")
        model.signature = json.dumps([model.signature, {"batch_size": args.batch_size}])
    cache = Cache(root/"translations.sqlite")
    costs = read_json(root/"translation_times.json") if (root/"translation_times.json").exists() else {}
    failures = []
    for offset in range(0, len(rows), args.batch_size):
        group = rows[offset:offset+args.batch_size]
        texts = [r["en"] for r in group]
        keys = {text: digest([model.signature, text]) for text in texts}
        refresh = {text for text, key in keys.items() if key not in costs}
        sync(device)
        start = perf_counter()
        # A cached lookup after an interrupted run cannot establish cold preprocessing cost.
        if args.batch_size == 1:
            text, = texts
            try:
                translations = {text: cache.get(model, text, "en", "uz", refresh=bool(refresh))}
            except InvalidTranslation as exc:
                translations = {text: exc}
        else:
            translations = cache.get_many(model, texts, "en", "uz", refresh=refresh)
        sync(device)
        seconds = (perf_counter()-start) / max(1, len(refresh))
        for r in group:
            mt = translations[r["en"]]
            if isinstance(mt, InvalidTranslation):
                if args.stage != "training":
                    write_json(root/"translation_times.json", costs)
                    raise mt
                failures.append(dict(id=r["id"], split=r["split"], en=r["en"], reason=str(mt)))
                continue
            key = keys[r["en"]]
            costs.setdefault(key, seconds)
            if args.stage == "training" and repeated_phrase(mt):
                failures.append(dict(id=r["id"], split=r["split"], en=r["en"], reason="Repeated decoder phrase"))
                continue
            r.update(uz=mt, mt=mt, reviewed=False, translation_key=key, translation_model=model.signature)
        if offset // args.batch_size % max(1, 50 // args.batch_size) == 0:
            write_json(root/"translation_times.json", costs)
    write_json(root/"translation_times.json", costs)
    if args.stage == "training":
        path = root/"training_bilingual.json"
        failed_ids = {r["id"] for r in failures}
        rows = [r for r in rows if r["id"] not in failed_ids]
        write_json(root/"training_translation_failures.json", failures)
        write_json(path, rows)
        print(f"{path}: {len(rows)} machine-translated training and validation expressions, "
              f"{len(failures)} excluded (see training_translation_failures.json)")
        return
    if args.stage == "build":
        rows = apply_reviews(rows, read_json(args.reviews or root/"review.json"))
    out = "pilot" if args.stage == "pilot" and len(rows) < len(read_json(root/"english.json")) else "bilingual"
    write_json(root/(out+".json"), rows)
    # Regeneration must never overwrite a reviewer's edits.
    template = root/("evaluation_review.json" if args.stage == "build" else "review.json")
    if not template.exists():
        write_json(template, [dict(r, reviewer="", status="pending", tags=[]) for r in rows
                              if args.stage == "pilot" or r["split"] == "test"])
    print(f"{out}: {len(rows)} expressions; review form: {template}")


if __name__ == "__main__":
    main()
