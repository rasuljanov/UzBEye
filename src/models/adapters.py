def select_trainable(core, checkpointing):
    # Identical selected modules in B and C. The pretrained box heads remain differentiable.
    core.requires_grad_(False)
    core.text_projection.requires_grad_(True)
    core.text_backbone.encoder.layer[-1].requires_grad_(True)
    core.encoder.layers[-1].fusion_layer.requires_grad_(True)
    core.decoder.layers[-1].encoder_attn_text.requires_grad_(True)
    if checkpointing:
        core.text_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
