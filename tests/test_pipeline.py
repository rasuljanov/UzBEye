import json
import torch
from PIL import Image
from transformers import BertTokenizerFast, GroundingDinoImageProcessor

from scripts.audit_translations import choose
from scripts.expand_dataset import assess_candidates, best_candidate
from scripts.polish_dataset import clean_english, select_reviewed
from src.common import read_json, write_json
from src.data import translate
from src.data.dataset import flatten, subset, training_input_hash
from src.data.preprocessing import corners, encode, normalized_box, normalize
from src.data.translate import apply_reviews
from src.evaluation.metrics import crossover, iou, overlaps, summarize
from src.inference.translation_backend import Cache, InvalidTranslation
from src.models.uzbek_encoder import vocabulary_map
from src.training.trainer import grounding_loss
import pytest


def example():
    return dict(image_id=7, ref_id=8, ann_id=9, split="train",
                raw_image_info=json.dumps(dict(id=7,width=100,height=50)),
                raw_anns=json.dumps(dict(id=9,image_id=7,bbox=[10,5,30,20])),
                sentences=[dict(raw="the man", sent_id=123)])


def test_parse_sample_and_bbox():
    row, = flatten([example()], "train")
    assert row["id"] == 123 and row["image"] == "COCO_train2014_000000000007.jpg"
    box = corners(torch.tensor(normalized_box(row))).tolist()
    assert box == pytest.approx([.1,.1,.4,.5])
    assert overlaps(row, [box]) == pytest.approx([1])
    rows = [dict(id=i) for i in range(12)]
    assert subset(rows,42,5) == subset(list(reversed(rows)),42,5)
    with pytest.raises(ValueError):
        list(flatten([example()], "test"))


def test_translation_cache(tmp_path):
    class Translator:
        signature = "model-a"
        calls = 0

        def translate(self, text, src, dst):
            self.calls += 1
            return str(self.calls)

    t = Translator()
    path = tmp_path/"cache.db"
    cache = Cache(path)
    assert cache.get(t,"a","en","uz") == "1"
    cache.db.close()
    cache = Cache(path)
    assert cache.get(t,"a","en","uz") == "1"
    assert cache.get(t,"a","uz","en") == "2"
    t.signature = "model-b"
    assert cache.get(t,"a","en","uz") == "3"


def test_translation_audit_checks_the_object_and_its_attributes():
    row = dict(en="A person in a brown jacket and a checkered hat")
    original = "Jigarrang kurtka va bosh kiyim kiygan odam."
    corrected = "Jigarrang kurtka va katakli bosh kiyim kiygan odam."
    choice, assessments = choose(row, {"original": original, "candidate": corrected},
                                 {original: row["en"], corrected: row["en"]})
    assert assessments["original"]["flags"] == ["term:plaid"]
    assert choice == "candidate"


def test_expansion_checks_spatial_cues_and_independent_evidence():
    en = "The red bag behind the chair"
    right = "Stulning orqasidagi qizil sumka."
    wrong = "Stulning oldidagi qizil sumka."
    row = dict(id=1, image_id=1, en=en, selection="new_image")
    translations = dict(llama={en: right}, ovozify={en: wrong}, back_en={right: en, wrong: en})
    record, = assess_candidates([row], translations)
    assert record["candidates"]["ovozify"]["flags"] == ["cue:behind"]
    assert best_candidate(record) is None
    candidate = record["candidates"]["llama"]
    candidate.update(score=.67, independent_score=.55)
    assert best_candidate(record) is None
    candidate.update(independent_score=.8)
    assert best_candidate(record)[0] == "llama"


def test_polish_preserves_grounding_metadata_and_rejects_a_lost_cue():
    row = dict(id=21, image_id=11, en="Red bag on the left", uz="Chapdagi qizil sumka.",
               split="train", bbox=[1, 2, 3, 4])
    evidence = {21: dict(review=True, old_score=.92, old_flags=[])}
    selected, report = select_reviewed([row], {21: "Qizil sumka."}, evidence, {})
    assert selected[0]["uz"] == row["uz"]
    assert report[0]["decision"] == "retained_prior_after_cross_check"
    direct = {21: dict(en=row["en"], suggested_uz="Chap tomondagi qizil sumka.")}
    selected, report = select_reviewed([row], {21: "?"}, evidence, direct)
    assert selected[0]["bbox"] == row["bbox"]
    assert selected[0]["uz"] == direct[21]["suggested_uz"]
    assert report[0]["decision"] == "direct_image_review"
    assert clean_english("The red bag on the left", "Red bag on the right") == "The red bag on the left"
    assert clean_english("Pizza in the right hand picture", "?") == "Pizza in the picture on the right"
    validation = row | {"split": "val"}
    preserved, _ = select_reviewed([validation], {21: "?"}, evidence, direct)
    assert preserved == [validation]


def test_preprocessing_and_alignment(tmp_path):
    vocab = tmp_path/"vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nqiz\nchap\n.\n?\n", encoding="utf8")
    tok = BertTokenizerFast(vocab_file=str(vocab))
    image = Image.new("RGB",(120,60))
    cfg = dict(sequence_length=16,image_size=64)
    inputs, pos = encode(tok,GroundingDinoImageProcessor(),image,"  chap qiz! ",cfg,torch.device("cpu"))
    assert inputs["pixel_values"].shape == (1,3,32,64)
    assert pos.tolist() == [False,True,True,False,False]
    assert normalize("  o‘ng  qoʻl ") == "o'ng qo'l"
    perm = vocabulary_map(tok, 2048)
    assert perm.unique().numel() == 2048
    assert perm[[tok.cls_token_id,tok.sep_token_id,7,8]].tolist() == [101,102,1012,1029]
    with pytest.raises(ValueError,match="budget"):
        encode(tok,GroundingDinoImageProcessor(),image,"qiz "*20,cfg,torch.device("cpu"))


def test_objective_and_metrics():
    boxes = torch.tensor([[.5,.5,.2,.2],[.2,.2,.1,.1]],requires_grad=True)
    logits = torch.zeros(2,3,requires_grad=True)
    grounding_loss(logits,boxes,[.51,.5,.18,.18]).backward()
    assert logits.grad[0].max() < 0 < logits.grad[1].min()
    assert torch.isfinite(boxes.grad).all()
    assert iou([0,0,1,1],[.5,0,1,1]) == .5
    s = summarize([dict(iou=[.2,.9]),dict(iou=[.5,.4])])
    assert s == dict(accuracy_50=.5,recall_1=.5,recall_5=1,mean_iou=.35)
    assert crossover(50,2,1) == 50 and crossover(50,1,2) is None


def test_ranking_loss_prefers_the_best_overlap_over_the_highest_scoring_wrong_box():
    target = [.5, .5, .2, .2]
    boxes = torch.tensor([target, [.5,.5,.18,.18], [.2,.2,.1,.1], [.8,.8,.1,.1]])
    logits = torch.tensor([[-2., -2.], [3., 3.], [1., 1.], [0., 0.]], requires_grad=True)
    extra = grounding_loss(logits, boxes, target, ranking_loss=True) - grounding_loss(logits, boxes, target)
    gradients, = torch.autograd.grad(extra, logits)
    assert extra > 0
    assert (gradients[0] < 0).all() and (gradients[2] > 0).all()
    torch.testing.assert_close(gradients[[1, 3]], torch.zeros_like(gradients[[1, 3]]))

    for subset in (slice(0, 2), slice(2, 4)):
        default = grounding_loss(logits[subset], boxes[subset], target)
        ranked = grounding_loss(logits[subset], boxes[subset], target, ranking_loss=True)
        torch.testing.assert_close(ranked, default)


def test_review_source_and_identity():
    row = dict(id=1,image_id=2,en="red",mt="qizil",bbox=[0,0,2,2],split="test")
    review = dict(row,uz="qizil",reviewer="reader",status="accepted")
    assert apply_reviews([row.copy()],[review])[0]["reviewed"]
    with pytest.raises(ValueError):
        apply_reviews([row.copy()],[dict(review,en="blue")])
    with pytest.raises(ValueError):
        apply_reviews([row.copy()],[dict(review,reviewer="")])


def test_training_hash_ignores_review_metadata_and_test_rows():
    row, = flatten([example()], "train")
    row["uz"] = "erkak"
    reviewed = dict(row, reviewed=True, review_text="bir erkak")
    test = dict(row, id=124, split="test", uz="ayol")
    assert training_input_hash([row]) == training_input_hash([reviewed, test])
    assert training_input_hash([row]) != training_input_hash([dict(row, uz="ayol")])


def test_training_translation_records_decoder_failures(tmp_path, monkeypatch):
    class Translator:
        signature = "mock"

        def translate(self, text, src, dst):
            if text == "bad":
                raise InvalidTranslation("decoder length limit")
            if text == "repetitive":
                return "O'z navbatida, " * 5
            return "erkak"

    rows = [dict(id=1, split="train", en="bad"), dict(id=2, split="val", en="the man"),
            dict(id=3, split="train", en="repetitive")]
    write_json(tmp_path/"english.json", rows)
    monkeypatch.setattr(translate, "backend", lambda cfg, device: Translator())
    monkeypatch.setattr("sys.argv", ["translate", "--root", str(tmp_path), "--stage", "training", "--device", "cpu"])
    translate.main()
    assert [r["id"] for r in read_json(tmp_path/"training_bilingual.json")] == [2]
    assert read_json(tmp_path/"training_translation_failures.json") == [
        dict(id=1, split="train", en="bad", reason="decoder length limit"),
        dict(id=3, split="train", en="repetitive", reason="Repeated decoder phrase"),
    ]


def test_batched_cache_deduplicates_and_retries_failed_rows(tmp_path):
    class Translator:
        signature = "batch"
        calls = []

        def translate_many(self, texts, src, dst):
            self.calls.append(texts)
            return [InvalidTranslation("decoder length limit") if text == "bad" else "erkak" for text in texts]

    model = Translator()
    cache = Cache(tmp_path/"batch.sqlite")
    result = cache.get_many(model, ["the man", "bad", "the man"], "en", "uz", refresh=set())
    assert result["the man"] == "erkak"
    assert isinstance(result["bad"], InvalidTranslation)
    cache.get_many(model, ["the man", "bad"], "en", "uz", refresh=set())
    assert model.calls == [["the man", "bad"], ["bad"]]
    cache.db.close()
