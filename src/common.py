import hashlib
import json
import os
import platform
import random
from importlib.metadata import version
from pathlib import Path

import yaml


def config(path):
    path = Path(path)
    values = yaml.safe_load(path.read_text(encoding="utf8"))
    parent = values.pop("base", None)
    return (config(path.parent / parent) if parent else {}) | values


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def environment(device):
    import torch
    return dict(os=platform.platform(), python=platform.python_version(), device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cuda=torch.version.cuda, packages={s: version(s) for s in
                ("torch", "torchvision", "transformers", "tokenizers", "huggingface-hub", "numpy", "Pillow")})


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_mib(device):
    import torch
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
