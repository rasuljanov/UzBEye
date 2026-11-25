import unicodedata

import torch


def normalize(text):
    text = unicodedata.normalize("NFC", text).translate(str.maketrans({c: "'" for c in "ʻʼ‘’`´"}))
    return " ".join(text.split())


def positive_tokens(offsets, phrase):
    # A complete referring expression denotes one region, including its modifiers.
    return [b > a and any(c.isalnum() for c in phrase[a:min(b, len(phrase))])
            for a, b in offsets]


def encode(tokenizer, processor, image, text, cfg, device):
    phrase = normalize(text).lower().rstrip(" .?!")
    ids = tokenizer(phrase + ".", return_offsets_mapping=True, return_tensors="pt")
    offsets = ids.pop("offset_mapping")[0].tolist()
    if len(offsets) > cfg["sequence_length"]:
        raise ValueError("Expression exceeds the shared token budget; revise the budget for every approach")
    positive = torch.tensor(positive_tokens(offsets, phrase), dtype=torch.bool, device=device)
    if not positive.any():
        raise ValueError("Expression contains no groundable tokens")
    edge = cfg["image_size"]
    pixels = processor(images=image, size={"shortest_edge": edge, "longest_edge": edge}, return_tensors="pt")
    return {k: v.to(device) for k, v in {**pixels, **ids}.items()}, positive


def normalized_box(row):
    x, y, w, h = row["bbox"]
    return [(x+w/2)/row["width"], (y+h/2)/row["height"], w/row["width"], h/row["height"]]


def corners(boxes):
    center, size = boxes.split(2, dim=-1)
    return torch.cat([center - size/2, center + size/2], dim=-1)
