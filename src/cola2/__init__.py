# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public API and Transformers auto-class registration for Cola2."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_cola2 import Cola2Config, Cola2GenerationConfig
from .modeling_cola2 import (
    Cola2ConditionalGenerationOutput,
    Cola2EncoderOutput,
    Cola2ForConditionalGeneration,
    build_export_components,
)

__version__ = "0.1.0"


def register_cola2_auto_classes() -> None:
    """Register local classes without requiring ``trust_remote_code``."""

    AutoConfig.register(Cola2Config.model_type, Cola2Config, exist_ok=True)
    AutoModel.register(
        Cola2Config, Cola2ForConditionalGeneration, exist_ok=True
    )
    AutoModelForCausalLM.register(
        Cola2Config, Cola2ForConditionalGeneration, exist_ok=True
    )


register_cola2_auto_classes()

# Tell save_pretrained() which source modules to copy into a standalone HF repo.
Cola2Config.register_for_auto_class("AutoConfig")
Cola2ForConditionalGeneration.register_for_auto_class("AutoModel")

__all__ = [
    "Cola2Config",
    "Cola2GenerationConfig",
    "Cola2EncoderOutput",
    "Cola2ConditionalGenerationOutput",
    "Cola2ForConditionalGeneration",
    "build_export_components",
    "register_cola2_auto_classes",
]
