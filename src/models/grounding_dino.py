import torch
from torch import nn
from transformers import AutoImageProcessor, AutoTokenizer, GroundingDinoForObjectDetection

from src.data.preprocessing import corners, encode
from src.models.assets import source
from src.models.uzbek_encoder import replace_encoder
from src.models.adapters import select_trainable


class Grounder(nn.Module):
    def __init__(self, cfg, approach, device, training=False):
        super().__init__()
        self.cfg, self.device = cfg, device
        rev = cfg["detector_revision"]
        path = source(cfg["detector"], rev)
        self.detector = GroundingDinoForObjectDetection.from_pretrained(path, revision=rev,
                                                                       disable_custom_kernels=True)
        if not 3 <= cfg["sequence_length"] <= self.detector.config.max_text_len:
            raise ValueError("Token budget must fit the detector's contrastive head")
        self.processor = AutoImageProcessor.from_pretrained(path, revision=rev, use_fast=False)
        if approach == "C":
            self.tokenizer, perm = replace_encoder(self.detector.model, cfg)
            self.register_buffer("token_map", perm, persistent=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, revision=rev, use_fast=True)
            self.token_map = None
        self.detector.requires_grad_(False)
        if training:
            select_trainable(self.detector.model, cfg["gradient_checkpointing"])
        self.to(device)
        self.eval()

    def forward(self, image, phrase):
        inputs, positive = encode(self.tokenizer, self.processor, image, phrase, self.cfg, self.device)
        if self.token_map is not None:
            inputs["input_ids"] = self.token_map[inputs["input_ids"]]
        with torch.autocast(self.device.type, dtype=torch.float16,
                            enabled=self.device.type == "cuda" and self.cfg["precision"] == "fp16"):
            out = self.detector(**inputs)
        logits = out.logits[0, :, :len(positive)].float()[:, positive]
        boxes = out.pred_boxes[0].float()
        if not (torch.isfinite(logits).all() and torch.isfinite(boxes).all()):
            raise FloatingPointError("Non-finite detector output")
        return logits, boxes

    @torch.inference_mode()
    def predict(self, image, phrase):
        logits, boxes = self(image, phrase)
        scores = logits.sigmoid().mean(-1)
        top = scores.topk(min(5, len(scores))).indices
        return corners(boxes[top]).clamp(0, 1).cpu().tolist(), scores[top].cpu().tolist()


def load_delta(model, path, approach):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["approach"] != approach or ck["config"] != model.cfg:
        raise ValueError("Checkpoint configuration or approach mismatch")
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in ck["weights"].items():
            own[k].copy_(v)
    return ck
