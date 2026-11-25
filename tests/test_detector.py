"""Offline forward pass through the actual architecture with a tiny random config."""
import torch
from transformers import GroundingDinoConfig, GroundingDinoForObjectDetection, SwinConfig


def test_detector_output_dimensions():
    backbone = SwinConfig(image_size=64, embed_dim=16, depths=[1]*4, num_heads=[1,2,4,8],
                          window_size=2, out_indices=[2,3,4])
    cfg = GroundingDinoConfig(backbone_config=backbone.to_dict(), text_config=dict(model_type="bert",
        vocab_size=1100,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4),
        d_model=64,encoder_layers=1,decoder_layers=1,encoder_attention_heads=4,decoder_attention_heads=4,
        encoder_ffn_dim=64,decoder_ffn_dim=64,num_queries=8,max_text_len=12,disable_custom_kernels=True)
    model = GroundingDinoForObjectDetection(cfg).eval()
    with torch.no_grad():
        out = model(pixel_values=torch.zeros(1,3,64,64),input_ids=torch.tensor([[101,104,1012,102]]))
    assert out.logits.shape == (1,8,12)
    assert out.pred_boxes.shape == (1,8,4)
    assert torch.isfinite(out.logits[:,:,:4]).all()
    assert torch.all((0 <= out.pred_boxes) & (out.pred_boxes <= 1))

def test_counted_prompts_preserve_the_target_and_ranked_fallback():
    from src.inference.instances import multiple_instances

    assert multiple_instances("Ikkita qizil soyabon.") == ("qizil soyabon", 2)
    assert multiple_instances("2 ta odam") == ("odam", 2)
    assert multiple_instances("To‘rtta mashina") == ("mashina", 4)
    assert multiple_instances("ikki odam yonidagi televizor") is None
    assert multiple_instances("O'ng tomondagi qizil kiyimli ayol.") is None
    assert multiple_instances("both people") == ("person", 2)
    assert multiple_instances("barcha odamlar") == ("odam", None)
    assert multiple_instances("odamlar") == ("odam", None)


def test_instance_filter_removes_duplicates_before_applying_count():
    from src.models.grounding_dino import distinct_instances

    proposals = torch.tensor([
        [.0, .0, .4, .9],
        [.02, .02, .39, .88],
        [.6, .0, 1., .9],
        [.4, .05, .5, .2],
    ])
    scores = torch.tensor([.9, .85, .8, .2])
    boxes, confidences = distinct_instances(proposals, scores, limit=2)
    assert boxes == proposals[[0, 2]].tolist()
    assert confidences == scores[[0, 2]].tolist()
    assert distinct_instances(proposals, scores * .01, limit=2) == ([], [])
