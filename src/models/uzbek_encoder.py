"""Adapt the tokenizer vocabulary to the delimiter IDs used by Transformers 4.51.3."""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from .assets import source


def vocabulary_map(tokenizer, size):
    native = [tokenizer.cls_token_id, tokenizer.sep_token_id]
    native += [tokenizer.convert_tokens_to_ids(p) for p in (".", "?")]
    if len(set(native)) != 4 or tokenizer.unk_token_id in native:
        raise ValueError("Tokenizer needs four distinct grounding delimiters")
    # A permutation keeps ordinary Uzbek tokens with IDs 101/102/1012/1029 unambiguous.
    perm = list(range(size))
    for original, wanted in zip(native, [101, 102, 1012, 1029]):
        other = perm.index(wanted)
        perm[original], perm[other] = perm[other], perm[original]
    return torch.tensor(perm)


def replace_encoder(core, cfg):
    rev = cfg["encoder_revision"]
    path = source(cfg["encoder"], rev)
    tokenizer = AutoTokenizer.from_pretrained(path, revision=rev, use_fast=True)
    bert = AutoModel.from_pretrained(path, revision=rev, add_pooling_layer=False, attn_implementation="eager")
    embeddings = bert.get_input_embeddings()
    perm = vocabulary_map(tokenizer, embeddings.num_embeddings)
    # Reindex the embedding rows once so the same permutation serves masking and lookup.
    with torch.no_grad():
        embeddings.weight.copy_(embeddings.weight[perm.argsort()])
    embeddings.padding_idx = int(perm[tokenizer.pad_token_id])
    core.text_backbone = bert
    core.text_projection = nn.Linear(bert.config.hidden_size, core.config.d_model)
    return tokenizer, perm
