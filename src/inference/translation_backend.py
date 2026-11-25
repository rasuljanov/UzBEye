import json
import sqlite3
from pathlib import Path
from typing import Protocol

from src.data.preprocessing import normalize
from src.models.assets import source


class Backend(Protocol):
    signature: str

    def translate(self, text: str, src: str, dst: str) -> str: ...


class InvalidTranslation(ValueError):
    """The decoder returned no usable translation for this expression."""


class NLLB:
    codes = {"en": "eng_Latn", "uz": "uzn_Latn"}

    def __init__(self, cfg, device):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        repo, rev = cfg["translator"], cfg["translator_revision"]
        path = source(repo, rev)
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=rev)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, revision=rev,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).eval().to(device)
        self.device = device
        self.signature = json.dumps([repo, rev, "greedy", 128, str(self.model.dtype), "latin-v1"])

    def translate(self, text, src, dst):
        result, = self.translate_many([text], src, dst)
        if isinstance(result, InvalidTranslation):
            raise result
        return result

    def translate_many(self, texts, src, dst):
        import torch
        self.tokenizer.src_lang = self.codes[src]
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.codes[dst]),
                                      max_new_tokens=128, do_sample=False, num_beams=1)
        tokens = ids.tolist()
        decoded = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        results = []
        for text, seq, result in zip(texts, tokens, decoded, strict=True):
            if sum(token != self.tokenizer.pad_token_id for token in seq[1:]) >= 128:
                results.append(InvalidTranslation(f"Translation reached length limit: {text}"))
                continue
            normalized = normalize(result)
            results.append(normalized or InvalidTranslation(f"Empty translation: {text}"))
        return results


def backend(cfg, device):
    # A custom backend can expose the same two-member interface.
    if cfg["translation_backend"] == "nllb":
        return NLLB(cfg, device)
    from importlib import import_module
    module, factory = cfg["translation_backend"].split(":")
    return getattr(import_module(module), factory)(cfg, device)


class Cache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS translations (key TEXT PRIMARY KEY, text TEXT NOT NULL)")

    def get(self, model, text, src, dst, *, refresh=False):
        key = json.dumps([model.signature, src, dst, text], ensure_ascii=False)
        hit = self.db.execute("SELECT text FROM translations WHERE key=?", (key,)).fetchone()
        if hit and not refresh:
            return hit[0]
        result = model.translate(text, src, dst)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO translations VALUES (?,?)", (key, result))
        return result

    def get_many(self, model, texts, src, dst, *, refresh):
        results, pending = {}, {}
        for text in dict.fromkeys(texts):
            key = json.dumps([model.signature, src, dst, text], ensure_ascii=False)
            hit = self.db.execute("SELECT text FROM translations WHERE key=?", (key,)).fetchone()
            if hit and text not in refresh:
                results[text] = hit[0]
            else:
                pending[text] = key
        if pending:
            translations = model.translate_many(list(pending), src, dst)
            results.update(zip(pending, translations, strict=True))
            with self.db:
                self.db.executemany("INSERT OR REPLACE INTO translations VALUES (?,?)",
                                    [(key, results[text]) for text, key in pending.items()
                                     if isinstance(results[text], str)])
        return results
