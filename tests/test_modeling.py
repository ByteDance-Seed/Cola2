# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from cola2 import (
    Cola2Config,
    Cola2ForConditionalGeneration,
    Cola2GenerationConfig,
)


def tiny_model() -> Cola2ForConditionalGeneration:
    torch.manual_seed(7)
    config = Cola2Config.tiny(
        encoder_embedding_size=16,
        encoder_hidden_size=16,
        encoder_intermediate_size=32,
        encoder_num_attention_heads=2,
        pool_num_hidden_layers=1,
        latent_encoder_num_hidden_layers=1,
        latent_dim=4,
        latent_block_size=2,
        dit_hidden_size=16,
        dit_intermediate_size=32,
        dit_num_hidden_layers=1,
        dit_num_attention_heads=2,
        dit_head_dim=8,
        dit_rope_dim=8,
        dit_block_size=2,
        decoder_hidden_size=16,
        decoder_intermediate_size=32,
        decoder_num_hidden_layers=1,
        decoder_num_attention_heads=2,
        decoder_num_key_value_heads=1,
        decoder_head_dim=8,
        decoder_condition_num_hidden_layers=1,
        duration_hidden_size=16,
        local_position_dim=4,
        max_span_length=2,
    )
    return Cola2ForConditionalGeneration(config).cpu().eval()


def test_tiny_forward_computes_real_span_causal_logits_and_loss():
    model = tiny_model()
    input_ids = torch.tensor(
        [[1, 4, 5, 6, 7], [1, 8, 9, 10, 0]], dtype=torch.long
    )
    attention_mask = input_ids.ne(0)

    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    assert output.logits.shape == (2, 5, model.config.vocab_size)
    assert output.latents.shape[:2] == output.span_lengths.shape
    assert torch.isfinite(output.logits[attention_mask]).all()
    assert output.loss is not None and torch.isfinite(output.loss)
    assert not torch.equal(
        output.logits[0, 1], output.logits[1, 1]
    ), "valid logits must be computed from the actual input"


def test_tiny_generate_runs_every_full_dit_stage_on_cpu():
    model = tiny_model()
    prompt = torch.tensor([[1, 11, 12, 13]], dtype=torch.long)
    generation_config = Cola2GenerationConfig(
        max_new_tokens=3,
        dit_num_steps=2,
        dit_guidance_scale=2.0,
        decoder_guidance_scale=-0.9,
        span_causal_chunk_size=1,
        dynamic_duration=True,
        latent_feedback=True,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
    )

    output = model.generate(
        prompt,
        generation_config=generation_config,
        generator=torch.Generator(device="cpu").manual_seed(123),
    )

    assert output.shape == (1, prompt.shape[1] + 3)
    assert torch.equal(output[:, : prompt.shape[1]], prompt)
    stats = model.last_generation_stats
    assert stats["prompt_encodes"] == 1
    assert stats["dit_denoise_steps"] >= 2
    assert stats["duration_predictions"] >= 1
    assert stats["decoder_chunks"] >= 1
    assert stats["feedback_rounds"] == stats["generation_rounds"] >= 1


def test_empty_prefix_cfg_gate_matches_cfg_one():
    model = tiny_model()
    block = torch.randn(model.config.dit_block_size, model.config.latent_dim)
    prefix = block.new_zeros((0, model.config.latent_dim))

    def run(config):
        model.last_generation_stats = {"dit_denoise_steps": 0}
        return model._denoise_block(prefix, block, 0, config)

    gated = run(
        Cola2GenerationConfig(
            dit_num_steps=2,
            dit_guidance_scale=10.0,
            dit_cfg_disable_empty_prefix=True,
        )
    )
    cfg_one = run(
        Cola2GenerationConfig(
            dit_num_steps=2,
            dit_guidance_scale=1.0,
        )
    )
    assert torch.equal(gated, cfg_one)


def test_strict_state_dict_loader_accepts_wrappers_and_rejects_holes():
    model = tiny_model()
    wrapped = {
        f"module.{key}": value.detach().clone()
        for key, value in model.state_dict().items()
    }
    restored = tiny_model()
    restored.load_strict_state_dict(wrapped)

    for key, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], expected)

    broken = dict(wrapped)
    broken.pop(next(iter(broken)))
    with pytest.raises(RuntimeError, match="missing="):
        restored.load_strict_state_dict(broken)


def test_save_pretrained_loads_through_auto_model(tmp_path: Path):
    model = tiny_model()
    model.save_pretrained(tmp_path, safe_serialization=True)

    restored = AutoModel.from_pretrained(tmp_path)
    strict_restored = tiny_model().load_strict_state_dict(tmp_path)

    assert isinstance(restored, Cola2ForConditionalGeneration)
    assert isinstance(
        strict_restored.generation_config, Cola2GenerationConfig
    )
    assert restored.config.model_type == "cola2"
    assert isinstance(restored.generation_config, Cola2GenerationConfig)
