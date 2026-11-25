import subprocess
from statistics import mean
from threading import Event, Thread
from time import perf_counter

import numpy as np
import torch

from src.common import peak_mib, sync
from src.data.dataset import open_image
from src.evaluation.metrics import overlaps, summarize


class Utilization:
    """Device-wide samples; other desktop workloads also contribute."""
    def __init__(self, device):
        self.samples = []
        self.stop = Event()
        self.index = device.index or 0
        self.thread = Thread(target=self.sample, daemon=True) if device.type == "cuda" else None

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "-i", str(self.index), "--query-gpu=utilization.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                        check=True, timeout=3, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                self.samples.append(float(result.stdout.strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self.stop.wait(.5)


def benchmark(model, translator, cache, root, rows, warmups, mode):
    device = model.device

    def query(row):
        sync(device)
        start = perf_counter()
        prompt = row["uz"]
        translation_ms = 0.
        if translator is not None:
            prompt = (cache.get(translator, prompt, "uz", "en") if mode == "cached" else
                      translator.translate(prompt, "uz", "en"))
            sync(device)
            translation_ms = (perf_counter()-start)*1000
        image = open_image(root, row)
        sync(device)
        t = perf_counter()
        boxes, scores = model.predict(image, prompt)
        sync(device)
        end = perf_counter()
        return dict(id=row["id"], image_id=row["image_id"], prompt=prompt, boxes=boxes,
                    scores=scores, iou=overlaps(row, boxes), translation_ms=translation_ms,
                    inference_ms=(end-t)*1000, total_ms=(end-start)*1000)

    if mode == "cached":
        for row in rows:
            cache.get(translator, row["uz"], "uz", "en")
    for i in range(warmups):
        query(rows[i % len(rows)])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    util = Utilization(device)
    if util.thread:
        util.thread.start()
    try:
        records = [query(row) for row in rows]
    finally:
        util.stop.set()
        if util.thread:
            util.thread.join()
    stats = summarize(records)
    for key in ["total_ms", "inference_ms", "translation_ms"]:
        times = [r[key] for r in records]
        for percentile in (50, 95):
            stats[f"{key}_p{percentile}"] = float(np.percentile(times, percentile))
    seconds = mean(r["total_ms"] for r in records)/1000
    stats.update(n=len(rows), mean_seconds=seconds, queries_per_s=1/seconds,
                 peak_allocated_mib=peak_mib(device), gpu_util_percent=mean(util.samples) if util.samples else None,
                 gpu_util_samples=len(util.samples))
    return stats, records
