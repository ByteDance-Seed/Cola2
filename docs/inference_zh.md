# 推理

[English](inference.md) · [文档索引](README_zh.md)

## 运行要求

Checkpoint 推理需要托管在
[ByteDance-Seed/Cola2](https://huggingface.co/ByteDance-Seed/Cola2) 的模型
snapshot。推荐使用支持 bfloat16 的 CUDA 环境。CPU 适合运行 tiny 测试配置，
但不适合作为发布 checkpoint 的实际推理目标。

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "/path/to/cola2"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda").eval()
```

`trust_remote_code=True` 会执行 checkpoint 中的 Python。加载不可信或镜像
snapshot 时应先审查这些代码。安装本仓库后也会在本地注册相同 class。

## 官方 preset

Checkpoint 的 `generation_config.json` 保存论文设置：

```json
{
  "do_sample": false,
  "max_new_tokens": 32,
  "dit_num_steps": 8,
  "dit_timestep_schedule": "low_noise",
  "dit_guidance_scale": 10.0,
  "decoder_guidance_scale": -0.9,
  "span_causal_chunk_size": 1,
  "dynamic_duration": true,
  "latent_feedback": true
}
```

Low-noise 调度使用 \(t_i=T(1-i/8)^2\)，在低噪声区域放置更多 Euler node。
直接调用 `model.generate(**inputs)` 会使用 checkpoint 的完整 preset。

## 确定性比较

`do_sample=false` 只控制 token selection。每个 latent block 仍从高斯噪声开始，
因此不传 generator 时调用仍具有随机性。

```python
prompt = "Question: What is the capital of France? Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generator = torch.Generator(device=model.device).manual_seed(42)

with torch.inference_mode():
    output = model.generate(**inputs, generator=generator)
```

需要完全 replay 时，应在每次调用前以同一 seed 新建 generator。复用同一个
generator 会推进其状态，从而有意产生不同 latent sample。

不同 device、PyTorch 版本或 kernel 之间不保证逐 bit 一致。Benchmark 会为每条
record 派生稳定 seed，使 resume 和 sharding 不改变已定义的公开 run。

## 生成流程

每个 block 中，Cola2：

1. 编码 prompt 与已经提交的上下文；
2. 使用 block-causal DiT 对 8 个 latent 位置去噪；
3. 为每个 latent 预测动态 token duration；
4. 每个 decoder chunk 解码 1 个完整 latent span；
5. 提交选中的 token；
6. 将完成的 span 重新编码为 latent feedback。

Decoder CFG -0.9 会组合 conditional 与 unconditional decoder logits。负值在
保留有用 latent 语义的同时，允许 causal decoder 补充压缩 latent 未表示的
token-level 信息。

## 参数覆盖

生成字段可直接传入，也可写入复制后的 `Cola2GenerationConfig`：

```python
output = model.generate(
    **inputs,
    max_new_tokens=48,
    dit_num_steps=8,
    dit_timestep_schedule="low_noise",
    dit_guidance_scale=10.0,
    decoder_guidance_scale=-0.9,
    span_causal_chunk_size=1,
    dynamic_duration=True,
    latent_feedback=True,
    repetition_penalty=1.0,
    do_sample=False,
    generator=torch.Generator(device=model.device).manual_seed(42),
)
```

DiT schedule 支持 `uniform`、`high_noise`、`low_noise` 和 `power`。使用
`power` 时还必须提供正数 `dit_timestep_power`。`temperature`、`top_k` 和
`top_p` 仅在 `do_sample=True` 时影响 token decoding。

发布版只支持一个 return sequence，不支持 beam search；block-parallel
span-causal generation 没有定义 beam search。

## 命令行

安装包后可运行：

```bash
cola2-chat \
  --model /path/to/cola2 \
  --device cuda \
  --dtype bfloat16 \
  --seed 42
```

仓库 helper 的等价调用为：

```bash
COLA2_MODEL_PATH=/path/to/cola2 bash scripts/run_chat.sh
```

常用选项包括用于单次生成的 `--prompt`、`--max-new-tokens`、
`--max-context-tokens`、`--dit-steps`、`--dit-timestep-schedule`、
`--dit-cfg`、`--decoder-cfg`、`--decoder-chunk-spans` 和
`--repetition-penalty`。

交互 wrapper 只是在本地把历史格式化为 `User:` / `Assistant:` 文本。Cola2 是
base model，并未作为聊天助手进行 instruction tuning。

## 故障排查

- **找不到模型：** 设置 `--model` 或 `COLA2_MODEL_PATH`。
- **显存不足：** 缩短 prompt 或输出、关闭其他 GPU 进程，或选择支持的低精度
  dtype；不要静默修改模型架构字段。
- **相同 prompt 输出不同：** 传入固定 seed 的 generator，并核对 device 与包版本。
- **Unused generation argument：** 使用[配置参考](configuration_zh.md)中的
  canonical name；未知字段会直接报错而不会被忽略。
- **Remote-code 错误：** 安装本仓库，或确认 snapshot 包含声明的 remote-code
  文件。
