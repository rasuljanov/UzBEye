import gc
import shutil
from pathlib import Path
import torch

from src.common import digest, environment, read_json, seed_all, write_json
from src.evaluation.benchmark import benchmark
from src.inference.translation_backend import Cache, backend
from src.models.grounding_dino import Grounder, load_delta
from src.models.assets import checkpoint_bytes


def evaluate(tag, root, rows, cfg, run, out, device, warmups):
    approach = tag[0]
    mode = tag.split("-")[-1]
    train = {}
    if run:
        train = read_json(run/"training.json")
        cfg = train["config"]
    seed_all(cfg["seed"])
    model = Grounder(cfg, approach, device)
    if run:
        load_delta(model, run/"best.pt", approach)
    translator = backend(cfg, device) if approach == "A" else None
    cache = Cache(root/"runtime.sqlite") if mode == "cached" else None
    stats, records = benchmark(model, translator, cache, root, rows, warmups, mode)
    detector_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    files = ["detector"]
    if approach == "C":
        files.append("encoder")
    if approach == "A" and cfg["translation_backend"] == "nllb":
        files.append("translator")
    stats["checkpoint_bytes"] = sum(checkpoint_bytes(cfg[k], cfg[k+"_revision"]) for k in files) + train.get("delta_bytes", 0)
    stats.update(approach=tag, train_seconds=train.get("train_seconds", 0), trainable=train.get("trainable", 0),
                 delta_bytes=train.get("delta_bytes",0), detector_parameter_bytes=detector_bytes,
                 translator_parameter_bytes=sum(p.numel()*p.element_size() for p in translator.model.parameters())
                 if hasattr(translator, "model") else 0)
    write_json(out/"metrics"/(tag+".json"), dict(stats=stats, config=cfg, environment=environment(device),
               evaluation_hash=digest(rows), warmups=warmups, training=train))
    if run:
        shutil.copyfile(run/"training.jsonl", out/"metrics"/(tag+"-training.jsonl"))
    write_json(out/"benchmarks"/(tag+".json"), records)
    if cache:
        cache.db.close()
    del model, translator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats, records
