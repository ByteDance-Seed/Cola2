# 架构

[English](architecture.md) · [文档索引](README_zh.md)

Cola2 将语言生成拆分为连续语义表示、latent prior 和离散 token realization。
发布代码仅包含推理实现。

## 系统视图

```text
prompt tokens
    │
    ▼
动态压缩 VAE ──► 已提交的 latent history
                         │
                         ▼
高斯噪声 ──► block-causal DiT ──► 新 latent block
                         │
                         ▼
                 duration + span-causal decoder
                         │
                         ▼
                    已提交文本 span
                         │
                         └── 重新编码为 latent feedback
```

## 动态压缩 VAE

`Cola2CompressionVAE` 将连续 token span 映射为 continuous latent。发布配置：

- 动态选择 1 至 8 个 token 的 span；
- 目标平均压缩率为 0.5；
- 每个 span 使用 64 维 variational latent；
- 使用 2560 维 token embedding、span pooling layer 和 latent encoder layer；
- 将 latent 展开为供 decoder 使用的 token-aligned condition。

Boundary score 用于识别语义边界。负反馈控制器调节有效 threshold，在不向语言
建模目标添加 rate loss 的情况下控制压缩率。

## Latent DiT

论文 checkpoint 中的 `Cola2DiT` 约为 4B 参数。公开配置包含 34 层、hidden
size 2560、20 个 attention head、64 维 latent，以及每 block 8 个 latent
位置。

DiT 使用 block-causal attention：已经提交的 latent block 构成 clean prefix，
新 block 内的全部位置联合去噪。官方 solver 使用 8 步 low-noise-biased Euler
调度：

\[
t_i = T(1-i/8)^2,\qquad i=0,\ldots,8.
\]

Classifier-free guidance 以 scale 10 组合 conditional 和 unconditional velocity
prediction。

## Token realization

Token decoder 使用 Qwen3-4B-shaped causal backbone：hidden size 2560、
36 层、32 个 query head 和 8 个 key/value head。生成 latent 按预测的 span
duration 展开，并通过 conditional layer 输入 decoder。

发布路径是 span-causal 的：模型从左到右提交完整 token span。官方 preset 每个
decoder chunk realization 一个 latent span。Decoder CFG 使用 -0.9，在 latent
重建与 decoder 自主语言建模能力之间进行平衡。

## Block 生成循环

每个输出 block 中，`Cola2ForConditionalGeneration.generate()`：

1. 编码 prompt 与已经提交的文本；
2. 为 8-latent block 中未填充的位置采样高斯噪声；
3. 使用 DiT 对 block 去噪；
4. 为每个新 latent 预测 token duration；
5. 解码并提交完整 token span；
6. 遇到 EOS 或达到 `max_new_tokens` 时停止；
7. 在生成下一个 block 前，按照固定 layout 重新编码已提交 span。

最后一步即 **latent feedback**，它避免后续 block 继续依赖与 decoder 实际选中
文本不一致的 latent trajectory。

## 论文与发布代码术语

论文在受控 token-decoder 实验中使用 masked-diffusion 术语，并在多个层级讨论
block-causal generation。在公开实现中：

- **DiT** 或 **Full-DiT** 指连续 latent denoiser；
- **span-causal decoder** 指发布的离散 realization 路径；
- **block-causal** 指 latent visibility pattern。

这些术语不表示所有实验性 token-decoder factorization 都由发布 checkpoint
提供。软件 API 应以公开 class name 和配置字段为准。

## Transformers 接口

`Cola2ForConditionalGeneration` 继承 `PreTrainedModel`，内部包含
`Cola2CompressionVAE` 与 `Cola2DiT`。导入 `cola2` 时会注册：

- `Cola2Config` 到 `AutoConfig`；
- 模型到 `AutoModel`；
- 同一个 conditional-generation 模型到 `AutoModelForCausalLM`。

主要公开操作包括：

- `encode()`：动态语义压缩；
- `forward()`：重建 logits；
- `generate()`：latent 与 token continuation；
- `from_pretrained()` / `save_pretrained()`：Transformers checkpoint 管理。

发布 checkpoint 也包含 `auto_map` 和 remote-code 模块，因此可通过
`trust_remote_code=True` 加载而无需单独安装本包。在安全敏感环境中，启用该
选项前应先审查 remote code。

## 范围

本仓库只记录并测试推理行为，不包含三阶段训练实现，也不具备训练复现所需的全部
产物。训练摘要见[论文概览](paper_zh.md)，公开默认值见
[配置参考](configuration_zh.md)。
