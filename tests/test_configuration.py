# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoModel

from cola2 import (
    Cola2Config,
    Cola2ForConditionalGeneration,
    Cola2GenerationConfig,
)
from cola2.cli_chat import build_parser


def test_generation_defaults_match_full_dit_release_contract():
    config = Cola2GenerationConfig()

    assert config.dit_num_steps == config.dit_steps == 8
    assert config.dit_timestep_schedule == "low_noise"
    assert config.dit_timestep_power is None
    assert config.dit_cfg_disable_empty_prefix is False
    assert config.dit_guidance_scale == config.dit_cfg_scale == 10.0
    assert config.decoder_guidance_scale == config.decoder_cfg_scale == -0.9
    assert config.span_causal_chunk_size == config.chunk_spans == 1
    assert config.dynamic_duration is True
    assert config.latent_feedback is True


def test_config_roundtrip_and_auto_registration():
    config = Cola2Config.tiny()
    restored = Cola2Config.from_dict(config.to_dict())

    assert restored.to_dict()["model_type"] == "cola2"
    assert restored.auto_map["AutoConfig"].endswith("Cola2Config")
    assert restored.auto_map["AutoModel"].endswith(
        "Cola2ForConditionalGeneration"
    )
    assert isinstance(AutoConfig.for_model("cola2"), Cola2Config)
    assert AutoModel._model_mapping[Cola2Config] is Cola2ForConditionalGeneration


def test_chat_cli_uses_release_preset():
    args = build_parser().parse_args(["--model", "dummy"])

    assert args.dit_steps == 8
    assert args.dit_timestep_schedule == "low_noise"
    assert args.dit_cfg == 10.0
    assert args.decoder_cfg == -0.9
    assert args.decoder_chunk_spans == 1
    assert args.repetition_penalty == 1.0
    assert args.max_new_tokens == 32


def test_generation_config_accepts_timestep_schedule_variants():
    assert (
        Cola2GenerationConfig(
            dit_timestep_schedule="high_noise"
        ).dit_timestep_schedule
        == "high_noise"
    )
    config = Cola2GenerationConfig(
        dit_timestep_schedule="power",
        dit_timestep_power=0.75,
    )
    assert config.dit_timestep_power == 0.75
