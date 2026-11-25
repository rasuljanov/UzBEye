from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def source(repo, revision):
    local = PROJECT_ROOT / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if local.exists() and (local / "tokenizer_config.json").exists() and any(
        (local / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return str(local)
    return repo


def checkpoint_bytes(repo, revision):
    from huggingface_hub import try_to_load_from_cache
    local = Path(source(repo, revision))
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = local / name
        if path.is_file():
            return path.stat().st_size
        cached = try_to_load_from_cache(repo, name, revision=revision)
        if isinstance(cached, str):
            return Path(cached).stat().st_size
    raise FileNotFoundError(f"Missing weight file for {repo}@{revision}")
