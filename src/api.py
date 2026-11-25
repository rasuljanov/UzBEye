"""Loopback API for the Vue model tester."""
from functools import lru_cache
from io import BytesIO
from os import environ
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from src.common import config, read_json
from src.inference.translation_backend import InvalidTranslation, backend
from src.models.grounding_dino import Grounder, load_delta

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = Path(environ.get("GROUNDING_CHECKPOINTS", ROOT / "runs" / "corrected")).resolve()
MAX_BYTES = 12 * 1024**2
MAX_PIXELS = 24_000_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ModelID = Literal["A", "B", "C", "D"]

app = FastAPI(title="UzBEye", docs_url=None, redoc_url=None, openapi_url=None)
lock = Lock()


def ready(approach):
    folder = CHECKPOINTS / approach
    return approach in ("A", "D") or all((folder / file).is_file() for file in ("training.json", "best.pt"))


@app.get("/api/models")
def models():
    return {"device": torch.cuda.get_device_name() if DEVICE.type == "cuda" else "CPU",
            "models": {tag: ready(tag) for tag in "ABCD"}}


@lru_cache(maxsize=3)
def load_detector(approach):
    if approach == "D":
        settings = config(ROOT / "configs" / "baseline.yaml")
        return Grounder(settings, approach, DEVICE)
    folder = CHECKPOINTS / approach
    settings = read_json(folder / "training.json")["config"]
    model = Grounder(settings, approach, DEVICE)
    load_delta(model, folder / "best.pt", approach)
    return model


@lru_cache(maxsize=1)
def translator():
    return backend(config(ROOT / "configs" / "baseline.yaml"), DEVICE)


def decode_upload(upload):
    raw = upload.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Choose an image under 12 MB.")
    try:
        with Image.open(BytesIO(raw), formats=("JPEG", "PNG", "WEBP")) as image:
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Choose an image under 24 megapixels.")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(422, "Choose a valid JPG, PNG or WebP image.") from exc


@app.post("/api/predict")
def predict(approach: Annotated[ModelID, Form()], prompt: Annotated[str, Form(max_length=500)],
            image: Annotated[UploadFile, File()]):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(422, "Enter an Uzbek prompt.")
    if not ready(approach):
        raise HTTPException(409, f"Model {approach} needs a checkpoint and training.json in its run folder.")
    picture = decode_upload(image)
    try:
        with lock:
            model = load_detector("D" if approach in ("A", "D") else approach)
            translation = translator() if approach == "A" else None
            start = perf_counter()
            grounded_prompt = translation.translate(prompt, "uz", "en") if translation else prompt
            boxes, scores = model.predict(picture, grounded_prompt)
            duration_ms = (perf_counter() - start) * 1000
    except InvalidTranslation as exc:
        raise HTTPException(422, "Translation failed. Try a shorter Uzbek prompt.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except torch.OutOfMemoryError as exc:
        raise HTTPException(503, "The GPU is out of memory. Free GPU memory and restart the local server.") from exc
    return {"approach": approach, "grounded_prompt": grounded_prompt,
            "width": picture.width, "height": picture.height, "duration_ms": duration_ms,
            "predictions": [{"box": box, "score": score} for box, score in zip(boxes, scores, strict=True)]}


dist = ROOT / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True))
