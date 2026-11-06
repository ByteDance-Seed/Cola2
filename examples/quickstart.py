# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Minimal Cola2 Transformers example."""

import os

import torch
from transformers import AutoModel, AutoTokenizer


model_path = os.environ["COLA2_MODEL_PATH"]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda").eval()

prompt = "Question: What is the capital of France? Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs)

print(
    tokenizer.decode(
        output[0, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )
)
