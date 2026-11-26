"""Expand the existing corrected RefCOCOg training split with audited Uzbek prompts."""

import argparse
import gc
import re
from collections import defaultdict
from pathlib import Path

from src.common import read_json, write_json
from src.data.dataset import SHARDS, flatten, subset

BASE_MANIFEST = "training_bilingual_corrected.json"
EXPANSION = "expansion_8k"
DISTINCTION = re.compile(
    r"\b(?:left|right|middle|front|back|behind|next to|closest|furthest|first|second|"
    r"black|white|red|blue|green|yellow|gray|brown|wearing|holding|sitting|standing)\b",
    re.IGNORECASE,
)


def text_key(text):
    return " ".join(re.findall(r"\w+(?:'\w+)*", text.casefold()))


def heldout_images(root):
    import pyarrow.parquet as pq

    return {image_id for split in ("val", "test")
            for image_id in pq.read_table(root / "raw" / SHARDS[split], columns=["image_id"])[
                "image_id"].to_pylist()}


def select_candidates(root, inherited, limit):
    import pyarrow.parquet as pq

    old_ids = {row["id"] for row in inherited}
    old_images = {row["image_id"] for row in inherited}
    heldout = heldout_images(root)
    old_annotations = {row["ann_id"] for row in inherited}
    old_text = {(row["image_id"], text_key(row["en"])) for row in inherited}
    old_groups = defaultdict(set)
    eligible = []
    for batch in pq.ParquetFile(root / "raw" / SHARDS["train"]).iter_batches(batch_size=2048):
        for ref in batch.to_pylist():
            group = (ref["image_id"], ref["category_id"])
            if ref["ann_id"] in old_annotations:
                old_groups[group].add(ref["ann_id"])
                continue
            if ref["image_id"] in heldout:
                continue
            for row in flatten([ref], "train"):
                key = text_key(row["en"])
                if (row["id"] not in old_ids and (row["image_id"], key) not in old_text
                        and 3 <= len(key.split()) <= 28):
                    eligible.append(row | {"category_id": ref["category_id"]})

    # Prefer one clear caption per target to avoid training on near-duplicate phrases.
    by_annotation = {}
    for row in subset(eligible, 42, 0):
        old = by_annotation.get(row["ann_id"])
        if old is None or (DISTINCTION.search(row["en"]) and not DISTINCTION.search(old["en"])):
            by_annotation[row["ann_id"]] = row
    by_group = defaultdict(list)
    for row in by_annotation.values():
        by_group[(row["image_id"], row["category_id"])].append(row)
    selected = []
    selected_images = set()
    for group, rows in by_group.items():
        if len(selected) >= min(550, limit // 10):
            break
        image, _ = group
        if image in selected_images or group not in old_groups:
            continue
        partner = next((r for r in rows if DISTINCTION.search(r["en"])), None)
        if partner:
            selected.append(partner | {"selection": "same_category_partner"})
            selected_images.add(image)
    pairs = 0
    for group, rows in by_group.items():
        if pairs >= min(600, limit // 9):
            break
        image, _ = group
        if image in old_images or image in selected_images:
            continue
        contrasting = [row for row in rows if DISTINCTION.search(row["en"])]
        if len(contrasting) < 2:
            continue
        first = contrasting[0]
        second = next((r for r in contrasting[1:] if text_key(r["en"]) != text_key(first["en"])), None)
        if second:
            selected.extend(row | {"selection": "same_category_pair"} for row in (first, second))
            selected_images.add(image)
            pairs += 1
    for row in by_annotation.values():
        if row["image_id"] not in old_images and row["image_id"] not in selected_images:
            selected.append(row | {"selection": "new_image"})
            selected_images.add(row["image_id"])
            if len(selected) == limit:
                return selected
    raise ValueError(f"Only {len(selected)} eligible candidates for a pool of {limit}")


def make_candidates(root, limit, device_name, batch_size):
    import torch
    from scripts.audit_translations import OVOZIFY, LlamaTranslator, generate
    from src.inference.translation_backend import Cache, NLLB

    work = root / EXPANSION
    pool_path = work / "candidate_sources.json"
    pool = read_json(pool_path) if pool_path.is_file() else []
    if len(pool) != limit:
        inherited = [row for row in read_json(root / BASE_MANIFEST) if row["split"] == "train"]
        pool = select_candidates(root, inherited, limit)
        write_json(pool_path, pool)
    print(f"{len(pool)} annotated training candidates", flush=True)
    # Similar source lengths reduce padding. The SQLite cache preserves completed batches on retry.
    english = sorted({row["en"] for row in pool}, key=lambda text: (len(text), text))
    device = torch.device(device_name)
    cache = Cache(work / "candidates.sqlite")
    try:
        llama = LlamaTranslator(device)
        llama.signature += f"/batch-{batch_size}"
        llama_text = generate(cache, llama, english, "en", "uz", batch_size, "Llama forward")
        llama_signature = llama.signature
        del llama
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ovo = NLLB(dict(translator=OVOZIFY[0], translator_revision=OVOZIFY[1]), device)
        ovo.signature += f"/batch-{batch_size}"
        ovo_text = generate(cache, ovo, english, "en", "uz", batch_size, "Ovozify forward")
        uzbek = sorted(set(llama_text.values()) | set(ovo_text.values()), key=lambda text: (len(text), text))
        reverse = generate(cache, ovo, uzbek, "uz", "en", batch_size, "English round trip")
        write_json(work / "translations.json", {
            "models": {"llama": llama_signature, "ovozify": ovo.signature},
            "llama": llama_text, "ovozify": ovo_text, "back_en": reverse,
        })
    finally:
        cache.db.close()


# Direct checks complement the English round trip, which can mask a missing relation.
EXTRA_CUES = [
    (r"\b(?:far left|left[ -]?most|extreme left)\b", r"\b(?:eng chap|chap (?:chet|chekka))\w*", "far left"),
    (r"\b(?:far right|right[ -]?most|extreme right)\b", r"\b(?:eng o'ng|o'ng (?:chet|chekka))\w*", "far right"),
    (r"\b(?:below|underneath|beneath|under)\b(?! [0-9])", r"\b(?:ost|tag|past|quyi|ostki)\w*", "below"),
    (r"\b(?:above|overhead|on top of)\b", r"\b(?:ust|tepa|yuqori)\w*", "above"),
    (r"\b(?:behind|in back of)\b", r"\b(?:orqa|ort)\w*", "behind"),
    (r"\bin front of\b", r"\b(?:old|ro'para)\w*", "in front of"),
    (r"\bbetween\b", r"\b(?:ora|o'rta)\w*", "between"),
    (r"\b(?:in the middle|in the cent(?:er|re))\b", r"\b(?:o'rta|markaz)\w*", "middle"),
    (r"\bwithout\b", r"\b(?:\w+siz|yo'q|emas|taqmagan|kiymagan|ushlamagan)\b", "without"),
    (r"\btwo\b|\b2\b", r"\b(?:ikki\w*|2)\b", "two"),
    (r"\bthree\b|\b3\b", r"\b(?:uch\w*|3)\b", "three"),
    (r"\bfour\b|\b4\b", r"\b(?:to'rt\w*|4)\b", "four"),
    (r"\bfirst\b", r"\b(?:birinchi|1-chi|1)\b", "first"),
    (r"\bsecond\b", r"\b(?:ikkinchi|2-chi|2)\b", "second"),
    (r"\bthird\b", r"\b(?:uchinchi|3-chi|3)\b", "third"),
    (r"\bnavy (?:blue|uniform|shirt|jacket|coat|shorts|pants)\b", r"\bto'q (?:ko'k|moviy)\w*", "navy blue"),
    (r"\bscar(?:f|ves)\b", r"\b(?:sharf|ro'mol|durr?a)\w*", "scarf"),
    (r"\b(?:baseball bat|swinging (?:a |the )?bat|batting|how to bat)\b",
     r"\b(?:tayoq|bita)\w*|\b(?:to'p|beysbol).{0,25}\bur(?:ish|moq|ayotgan)", "baseball bat"),
]


def cue_flags(english, uzbek):
    flags = [f"cue:{label}" for source, target, label in EXTRA_CUES
             if re.search(source, english, re.I) and not re.search(target, uzbek, re.I)]
    numbers = set(re.findall(r"\b\d{2,}\b", english)) - set(re.findall(r"\b\d{2,}\b", uzbek))
    flags.extend(f"cue:number:{number}" for number in sorted(numbers))
    if re.search(r"\bjinsiy\b", uzbek, re.I) and not re.search(r"\bsex(?:ual)?\b", english, re.I):
        flags.append("unrelated_decoder_content")
    return flags


def assess_candidates(rows, translations):
    from scripts.audit_translations import score
    from src.data.preprocessing import normalize

    records = []
    for row in rows:
        en = row["en"]
        candidates = {}
        for method in ("llama", "ovozify"):
            uz = normalize(translations[method].get(en, ""))
            back = translations["back_en"].get(uz, "")
            merit, flags = score(en, uz, back)
            if uz:
                flags.extend(cue_flags(en, uz))
                if text_key(uz) == text_key(en):
                    flags.append("unchanged_english")
                if re.match(r"^(?:translation|tarjima|uzbek(?:cha)?)\s*:", uz, re.I):
                    flags.append("decoder_header")
            candidates[method] = {"uz": uz, "back_en": back, "score": merit, "flags": flags}
        records.append({"id": row["id"], "image_id": row["image_id"], "en": en,
                        "selection": row["selection"], "candidates": candidates})
    return records


def audit_candidates(root, device, batch_size):
    import torch
    from scripts.audit_translations import generate, score
    from src.common import config
    from src.inference.translation_backend import Cache, NLLB

    work = root / EXPANSION
    rows = read_json(work / "candidate_sources.json")
    translations = read_json(work / "translations.json")
    records = assess_candidates(rows, translations)
    eligible = sorted({c["uz"] for row in records for c in row["candidates"].values()
                       if c["score"] >= .65 and not c["flags"]}, key=lambda text: (len(text), text))
    cache = Cache(work / "candidates.sqlite")
    try:
        independent = NLLB(config("configs/baseline.yaml"), torch.device(device))
        independent.signature += f"/batch-{batch_size}"
        reverse = generate(cache, independent, eligible, "uz", "en", batch_size, "Independent English round trip")
    finally:
        cache.db.close()
    for row in records:
        for candidate in row["candidates"].values():
            if candidate["score"] < .65 or candidate["flags"]:
                continue
            back = reverse.get(candidate["uz"], "")
            merit, flags = score(row["en"], candidate["uz"], back)
            candidate.update(independent_back_en=back, independent_score=merit, independent_flags=flags)
    write_json(work / "comparison.json", records)
    write_json(work / "models.json", translations["models"] | {"independent_reverse": independent.signature})
    print(f"Compared {len(records)} English sources. "
          f"{sum(best_candidate(r) is not None for r in records)} pass both machine checks.", flush=True)


def best_candidate(record):
    eligible = [(method, candidate) for method, candidate in record["candidates"].items()
                if not candidate["flags"] and candidate["score"] >= .65
                and candidate.get("independent_score", 0.) >= .50
                and candidate["score"] + candidate["independent_score"] >= 1.32]
    return max(eligible, default=None, key=lambda item: (
        item[1]["score"] + item[1]["independent_score"],
        item[1]["independent_score"],
    ))


def accepted_rows(root, inherited, target):
    from scripts.audit_translations import english_words
    from src.data.preprocessing import normalize
    from src.data.translate import repeated_phrase

    work = root / EXPANSION
    source_path = work / "candidate_sources.json"
    sources = {r["id"]: r for r in read_json(source_path)}
    audit = read_json(work / "comparison.json")
    review_path = work / "direct_review.json"
    direct = read_json(review_path) if review_path.is_file() else {}
    if set(direct) - {str(r["id"]) for r in audit}:
        raise ValueError("Direct review contains unknown expression IDs")
    seen = {(r["image_id"], text_key(r["uz"])) for r in inherited}
    source_keys = {(r["image_id"], tuple(english_words(r["en"]))) for r in inherited}
    additions, decisions = [], []
    needed = target - len(inherited)
    if needed < 1:
        raise ValueError("Target must exceed the inherited training count")
    for record in sorted(audit, key=lambda r: str(r["id"]) not in direct):
        row = sources[record["id"]]
        reviewed = direct.get(str(row["id"]))
        if reviewed is not None:
            if reviewed["en"] != row["en"]:
                raise ValueError(f"Review source changed for {row['id']}")
            if not reviewed["uz"]:
                decisions.append({"id": row["id"], "status": "excluded_after_direct_review",
                                  "reason": reviewed["reason"]})
                continue
            uz = normalize(reviewed["uz"])
            method = "direct_review"
            if not text_key(uz) or repeated_phrase(uz) or re.search(r"[\u0400-\u04ff]", uz):
                raise ValueError(f"Invalid direct correction for {row['id']}")
        else:
            chosen = best_candidate(record)
            if chosen is None:
                decisions.append({"id": row["id"], "status": "excluded_by_machine_checks"})
                continue
            method, evidence = chosen
            uz = evidence["uz"]
        key = (row["image_id"], text_key(uz))
        source_key = (row["image_id"], tuple(english_words(row["en"])))
        if key in seen or source_key in source_keys:
            decisions.append({"id": row["id"], "status": "duplicate_prompt_on_image"})
            continue
        if len(additions) >= needed:
            decisions.append({"id": row["id"], "status": "qualified_reserve", "method": method})
            continue
        seen.add(key)
        source_keys.add(source_key)
        chosen = best_candidate(record)
        original = record["candidates"]["llama"]["uz"] or record["candidates"]["ovozify"]["uz"] or uz
        additions.append(row | {"uz": uz, "mt": original, "reviewed": False,
                                "audit_method": method, "audit_needs_review": False})
        decisions.append({"id": row["id"], "status": "selected", "method": method,
                          "direct_review": reviewed, "uz": uz,
                          "machine_choice": chosen[0] if chosen else None})
    write_json(work / "selection_decisions.json", decisions)
    if len(additions) != needed:
        raise ValueError(f"Only {len(additions)} of {needed} additions passed quality checks. "
                         "Increase --pool-size and run the candidates and audit stages again.")
    return additions, decisions


def fetch_images(root, rows, workers):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from src.data.dataset import fetch

    def get(row):
        path = root / "images" / row["image"]
        fetch("https://s3.amazonaws.com/images.cocodataset.org/train2014/" + row["image"], path)
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"Image metadata mismatch: {path}")
            image.load()

    unique = {row["image_id"]: row for row in rows}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(get, unique.values()), 1):
            if i % 500 == 0 or i == len(unique):
                print(f"Verified {i}/{len(unique)} images", flush=True)


def finalize(root, target, workers):
    from collections import Counter
    from pyarrow.parquet import read_table
    from transformers import AutoTokenizer
    from src.common import config, digest
    from src.data.dataset import REVISION, training_input_hash
    from src.models.assets import source

    base = read_json(root / BASE_MANIFEST)
    inherited = [r for r in base if r["split"] == "train"]
    validation = [r for r in base if r["split"] == "val"]
    additions, decisions = accepted_rows(root, inherited, target)
    combined = inherited + additions + validation
    held_out = heldout_images(root)
    train_images = {r["image_id"] for r in inherited + additions}
    if train_images & held_out or len({r["id"] for r in combined}) != len(combined):
        raise ValueError("Split image leakage or duplicate expression IDs")
    for settings in (config("configs/translate_train.yaml"), config("configs/tahrirchi_encoder.yaml")):
        model = "encoder" if "encoder" in settings else "detector"
        revision = settings[f"{model}_revision"]
        tokenizer = AutoTokenizer.from_pretrained(source(settings[model], revision), revision=revision)
        # The detector lowercases and appends a period before tokenization.
        captions = [r["uz"].lower().rstrip(" .?!") + "." for r in combined]
        for start in range(0, len(captions), 256):
            tokens = tokenizer(captions[start:start + 256], truncation=False)["input_ids"]
            if any(len(ids) > settings["sequence_length"] for ids in tokens):
                raise ValueError(f"Caption exceeds the token budget for {model}")
    train_rows = inherited + additions
    labels = read_table(root / "raw" / SHARDS["train"], columns=["ann_id", "category_id"]).to_pylist()
    categories = {label["ann_id"]: label["category_id"] for label in labels}
    targets = defaultdict(set)
    for row in train_rows:
        targets[(row["image_id"], categories[row["ann_id"]])].add(row["ann_id"])
    paired = {(r["image_id"], r["category_id"]) for r in additions
              if len(targets[(r["image_id"], r["category_id"])]) > 1}
    fetch_images(root, combined, workers)
    work = root / EXPANSION
    summary = {
        "dataset": "RefCOCOg UMD",
        "source_revision": REVISION,
        "inherited_expressions": len(inherited),
        "inherited_audit_methods": dict(Counter(r["audit_method"] for r in inherited)),
        "new_expressions": len(additions),
        "train_expressions": target,
        "train_images": len(train_images),
        "new_train_images": len(train_images - {r["image_id"] for r in inherited}),
        "validation_expressions": len(validation),
        "validation_digest": digest(validation),
        "base_training_input_hash": training_input_hash(base),
        "training_input_hash": training_input_hash(combined),
        "new_selection": dict(Counter(r["selection"] for r in additions)),
        "images_with_added_same_category_targets": len({image_id for image_id, _ in paired}),
        "new_translation_methods": dict(Counter(r["audit_method"] for r in additions)),
        "candidate_decisions": dict(Counter(r["status"] for r in decisions)),
        "models": read_json(work / "models.json"),
    }
    write_json(work / "summary.json", summary)
    write_json(root / "training_bilingual_8k.json", combined)
    print(summary, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("candidates", "audit", "finalize"), default="candidates")
    p.add_argument("--target", type=int, default=8000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--root", type=Path, default=Path("data/refcocog"))
    p.add_argument("--pool-size", type=int, default=7200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=48)
    args = p.parse_args()
    if min(args.pool_size, args.batch_size, args.target, args.workers) < 1:
        p.error("Counts must be positive")
    if args.stage == "candidates":
        make_candidates(args.root, args.pool_size, args.device, args.batch_size)
    elif args.stage == "audit":
        audit_candidates(args.root, args.device, args.batch_size)
    else:
        finalize(args.root, args.target, args.workers)


if __name__ == "__main__":
    main()

