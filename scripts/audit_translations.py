"""Compare each Uzbek prompt with its English source and save a versioned repair."""

import argparse
import gc
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from src.common import read_json, write_json
from src.data.preprocessing import normalize
from src.data.translate import repeated_phrase
from src.inference.translation_backend import Cache, InvalidTranslation, NLLB
from src.models.assets import source

OVOZIFY = ("OvozifyLabs/nllb-200-uz-en-v1", "56eb41e3270c93999f75b0a6b631369e3cfe734c")
LLAMA = ("Saidakmal/Llama-3.2-1B-uz-en-translator", "fbe5c1be522a0aa0d2e1fb6b5134e46f7dbea1a1")
PILOT_IDS = {62288, 22496, 79369, 65822, 21271, 70869, 53708, 28495, 22290, 78911,
             72200, 91031, 22027, 21444, 90381, 11840, 51173, 4441, 32694, 69888}
STOPWORDS = {"a", "an", "the", "is", "that", "its", "it", "to", "in", "on", "of", "with", "and", "one"}
SYNONYMS = {"grey": "gray", "colour": "color", "blond": "blonde", "persons": "person", "people": "person",
            "women": "woman", "men": "man", "children": "child", "trousers": "pants", "centre": "middle",
            "center": "middle", "underneath": "below", "under": "below", "beneath": "below", "beside": "next"}
CLUES = {"left", "right", "middle", "behind", "below", "above", "foreground", "background",
         "black", "white", "red", "blue", "green", "yellow", "gray", "pink", "brown", "purple", "orange",
         "first", "second", "third", "fourth", "two", "three", "four", "without", "not", "blonde"}
TERM_CHECKS = [
    (r"\bumbrellas?\b", r"\b(?:soyabon|zont|shamsiya)", "umbrella"),
    (r"\bkites?\b", r"\bvarrak", "kite"),
    (r"\bbench(?:es)?\b", r"\b(?:skameyka|o'rin)", "bench"),
    (r"\bbackpacks?\b", r"\b(?:ryukzak|orqa.{0,20}sumka)", "backpack"),
    (r"\b(?:stuffed|teddy) bears?\b", r"\bayiq", "toy bear"),
    (r"\b(?:couch|sofa)\b", r"\b(?:divan|sofa)", "couch"),
    (r"\bblond(?:e)?\b", r"\b(?:malla|sariq|sarig'|sarg'ish|blond)", "blonde"),
    (r"\bgiraffes?\b", r"\b(?:jiraf|giraf|ziraf)", "giraffe"),
    (r"\bzebras?\b", r"\bzebr", "zebra"),
    (r"\btoothbrush(?:es)?\b", r"\btish\b.{0,35}\b(?:cho'tka|chochka|shchetka)", "toothbrush"),
    (r"\bdonuts?\b|\bdoughnuts?\b", r"\b(?:donat|donut|ponchik)", "doughnut"),
    (r"\bsandwi(?:ch(?:es)?|c)\b", r"\b(?:sendvi|sandvi|buterbrod)", "sandwich"),
    (r"\bskis?\b", r"\b(?:chang'i|changi|ski)", "ski"),
    (r"\b(?:checkered|plaid|tartan)\b", r"\b(?:katak|kletka)", "plaid"),
    (r"\bstrip(?:ed|es)\b", r"\b(?:chiziq|yo'l|tasma)", "striped"),
    (r"\btan\b",
     r"\b(?:och (?:jigarrang|qo'ng'ir)|bej|malla)", "tan"),
    (r"\borange (?:colou?red|soup|jacket|cones?|ticket|hat|shirt|car|shorts|dress|chair|lettering|tie)\b",
     r"\b(?:to'q sariq|zarg'aldoq|noranj)", "orange color"),
    (r"\bleft(?:most|-hand)?\b(?! over)", r"\bchap", "left"),
    (r"\bright(?:most|-hand)?\b(?! next)", r"\bo'ng", "right"),
    (r"\bwhite\b", r"\boq", "white"),
    (r"\bblack\b", r"\bqora", "black"),
    (r"\bgray\b|\bgrey\b", r"\bkul\s?rang", "gray"),
    (r"\bbrown\b", r"\b(?:jigarrang|qo'ng'ir)", "brown"),
    (r"\bblue\b", r"\b(?:ko'k|moviy|havorang)", "blue"),
    (r"\bgreen\b", r"\byashil", "green"),
    (r"\bred\b", r"\bqizil", "red"),
    (r"\byellow\b", r"\bsariq", "yellow"),
    (r"\bpink\b", r"\bpushti", "pink"),
    (r"\b(?:purple|violet)\b", r"\bbinafsha", "purple"),
]


class LlamaTranslator:
    def __init__(self, device):
        repo, revision = LLAMA
        path = source(repo, revision)
        files = {}
        for name in ("tokenizer.json", "chat_template.jinja"):
            local = Path(path)/name
            files[name] = local if local.is_file() else Path(hf_hub_download(repo, name, revision=revision))
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(files["tokenizer.json"]),
                                                bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                                pad_token="<|eot_id|>", padding_side="left")
        self.tokenizer.chat_template = files["chat_template.jinja"].read_text(encoding="utf8")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(path, revision=revision, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.signature = f"{repo}@{revision}/greedy-96/{dtype}"

    @torch.inference_mode()
    def translate_many(self, texts, src, dst):
        languages = {"en": "English", "uz": "Uzbek"}
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Translate the following text from {languages[src]} to "
                 f"{languages[dst]}. Output only the translation.\n\n{text}"}],
                tokenize=False, add_generation_prompt=True,
            )
            for text in texts
        ]
        encoded = self.tokenizer(prompts, add_special_tokens=False, return_token_type_ids=False,
                                 return_tensors="pt", padding=True).to(self.device)
        generated = self.model.generate(**encoded, do_sample=False, max_new_tokens=96,
                                        temperature=1.0, top_p=1.0, pad_token_id=self.tokenizer.pad_token_id)
        output = generated[:, encoded.input_ids.shape[1]:].tolist()
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return [normalize(text) or InvalidTranslation("Empty translation") for text in decoded]


def generate(cache, translator, texts, src, dst, batch_size, label):
    texts = list(dict.fromkeys(text for text in texts if text))
    output = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        translated = cache.get_many(translator, batch, src, dst, refresh=set())
        output.update({text: value for text, value in translated.items() if isinstance(value, str)})
        if start % (batch_size * 20) == 0 or start + len(batch) == len(texts):
            print(f"{label}: {start + len(batch)}/{len(texts)} unique prompts processed", flush=True)
    return output


def english_words(text):
    standard = re.sub(r"\bright next\b", "next", text.casefold())
    tokens = re.findall(r"[a-z]+|[0-9]+", standard)
    return [SYNONYMS.get(word, word) for word in tokens if word not in STOPWORDS]


def score(source_text, candidate, back):
    """Use round-trip evidence and check cues that change the referred object."""
    if not candidate or not re.search(r"[a-zA-Z]", candidate) or repeated_phrase(candidate):
        return 0.0, ["missing_or_invalid"]
    src, rt = english_words(source_text), english_words(back)
    common = sum((Counter(src) & Counter(rt)).values())
    overlap = 2 * common / max(1, len(src) + len(rt))
    order = SequenceMatcher(None, src, rt, autojunk=False).ratio()
    missing_terms = [name for en, uz, name in TERM_CHECKS
                     if re.search(en, source_text, re.I) and not re.search(uz, candidate, re.I)]
    missing_clues = sorted((set(src) & CLUES) - set(rt))
    extra_clues = sorted((set(rt) & CLUES) - set(src))
    issues = [f"term:{term}" for term in missing_terms]
    issues.extend(f"missing:{clue}" for clue in missing_clues)
    issues.extend(f"extra:{clue}" for clue in extra_clues)
    if not back:
        issues.append("missing_round_trip")
    if re.search(r"[\u0400-\u04ff]", candidate):
        issues.append("cyrillic")
    if len(candidate.split()) > max(14, 3 * len(source_text.split())):
        issues.append("unusually_long")
    penalty = .21 * len(missing_terms) + .12 * len(missing_clues) + .10 * len(extra_clues)
    penalty += .15 * int("cyrillic" in issues or "unusually_long" in issues)
    return round(max(0.0, .7 * overlap + .3 * order - penalty), 4), issues


def choose(row, candidates, reverse):
    assessments = {}
    for name, text in candidates.items():
        back = reverse.get(text, "")
        merit, flags = score(row["en"], text, back)
        assessments[name] = dict(uz=text, back_en=back, score=merit, flags=flags)
    best = max(assessments, key=lambda tag: assessments[tag]["score"])
    old = assessments["original"]
    proposed = assessments[best]
    if best != "original":
        fixes_term = (any(flag.startswith("term:") for flag in old["flags"])
                      and not any(flag.startswith("term:") for flag in proposed["flags"]))
        if proposed["score"] - old["score"] < .09 and not fixes_term:
            best = "original"
    return best, assessments


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path("data/refcocog"))
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--batch-size", type=int, default=32)
    cli.add_argument("--pilot", action="store_true", help="Inspect known difficult examples first.")
    args = cli.parse_args()
    if args.batch_size < 1:
        cli.error("--batch-size must be positive")
    root = args.root
    rows = read_json(root/"training_bilingual.json")
    if args.pilot:
        rows = [row for row in rows if row["id"] in PILOT_IDS]
    work = root/"translation_audit"
    cache = Cache(work/"candidates.sqlite")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(8)
    sources = [r["en"] for r in rows]
    llama = LlamaTranslator(device)
    llama.signature += f"/batch-{args.batch_size}"
    llama_uz = generate(cache, llama, sources, "en", "uz", args.batch_size, "Llama English → Uzbek")
    llama_signature = llama.signature
    del llama
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ovo = NLLB(dict(translator=OVOZIFY[0], translator_revision=OVOZIFY[1]), device)
    ovo.signature += f"/batch-{args.batch_size}"
    ovo_uz = generate(cache, ovo, sources, "en", "uz", args.batch_size, "Ovozify English → Uzbek")
    reverse = generate(cache, ovo, [r["uz"] for r in rows] + list(llama_uz.values()) + list(ovo_uz.values()),
                       "uz", "en", args.batch_size, "Uzbek → English cross-check")
    cache.db.close()
    manual_path = work/"corrections.json"
    manual = read_json(manual_path) if manual_path.exists() else {}
    final, audit = [], []
    for row in rows:
        en = row["en"]
        candidates = {"original": row["uz"], "llama": llama_uz.get(en, ""), "ovozify": ovo_uz.get(en, "")}
        choice, assessments = choose(row, candidates, reverse)
        selected = assessments[choice]
        reviewed = str(row["id"]) in manual
        if reviewed:
            corrected = normalize(manual[str(row["id"])] or row["uz"])
        else:
            corrected = normalize(selected["uz"])
        if not corrected:
            raise ValueError(f"Empty selected translation for expression {row['id']}")
        needs_review = not reviewed and (selected["score"] < .60 or bool(selected["flags"]))
        method = "assistant" if reviewed else choice
        audit.append(dict(id=row["id"], split=row["split"], image_id=row["image_id"], en=en,
                          original=row["uz"], corrected=corrected, decision=method, needs_review=needs_review,
                          candidates=assessments))
        final.append(row | dict(uz=corrected, audit_method=method, audit_needs_review=needs_review))
    if args.pilot:
        write_json(work/"pilot.json", audit)
        return
    for tag in ("train", "val"):
        part = [r for r in audit if r["split"] == tag]
        print(tag, dict(total=len(part), changed=sum(r["original"] != r["corrected"] for r in part),
                        flagged=sum(r["needs_review"] for r in part),
                        methods=dict(Counter(r["decision"] for r in part))), flush=True)
    write_json(work/"comparison.json", audit)
    write_json(work/"needs_review.json", [r for r in audit if r["needs_review"]])
    write_json(work/"provenance.json", dict(method="two local candidates + English round trip and cue checks",
               llama=llama_signature, ovozify=ovo.signature, compared=len(audit), manual=len(manual)))
    write_json(root/"training_bilingual_corrected.json", final)


if __name__ == "__main__":
    main()
