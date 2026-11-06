# 模型卡：Cola2

[English](model_card.md) · [文档索引](README_zh.md)

## 模型简介

Cola2 是用于层次化文本生成的 base continuous-latent language model，包含：

- 将 1–8 token 语义 span 映射为 64 维 latent 的动态压缩 VAE；
- 约 4B 参数、34 层的 block-causal latent DiT；
- 36 层 Qwen3-4B-shaped span-causal token decoder。

模型使用 dynamic duration prediction，并把已经提交的文本重新编码为 latent
feedback。Checkpoint 与 tokenizer 将在 Hugging Face 单独发布，链接目前为
**即将发布**。本仓库包含推理实现与评测契约。

## 论文与项目

- **论文：** *Cola2: Co-Adaptive Hierarchical Language Modeling with
  Continuous Semantic Compression and Balanced Discrete Generation*
- **项目主页：** <https://hongcanguo.github.io/Cola2>
- **论文/arXiv：** 即将发布
- **模型仓库：** 即将发布
- **架构：** [docs/architecture_zh.md](architecture_zh.md)

## 训练摘要

论文 checkpoint 使用经 Qwen3-4B tokenizer 重新分词的 OLMo 2 预训练文本。
Decoder 与 VAE encoder embedding 从 Qwen3-4B-Base 初始化。

| 阶段 | 训练组件 | 新增 source tokens |
|---|---|---:|
| I | 动态压缩 VAE | 52.43B |
| II | VAE + 约 4B latent DiT，detached DiT target | 125.83B |
| III | DiT，冻结 Stage-II EMA VAE | 167.77B |
| **总计** |  | **346.03B** |

公开仓库不包含训练代码、训练语料、optimizer state 或足以复现训练的完整产物。

## 输入与输出

输入为发布版 Qwen3-compatible tokenizer 接受的 tokenized text。输出遵循
decoder-only Transformers `generate()` 约定：原 prompt 后拼接生成 token ID。

实际 context 与输出上限取决于显存和 checkpoint 配置。公开 benchmark 将 prompt
截断到 2048 token，最多生成 32 个新 token；这是评测设置，不代表通用质量保证。

## 适用范围

- continuous-latent 与 block-parallel 语言生成研究；
- 语义压缩和 latent/token 协同适配分析；
- 短文本补全、cloze、多项选择和抽取式回答实验；
- 复现公开推理与 scoring 协议。

任何下游应用都应独立验证模型行为。

## 超出范围的用途

- 将输出当作可靠事实或专业建议；
- 作为 instruction-following 或 safety-aligned assistant 部署；
- 对人做自主高影响决策；
- 违法、欺骗、滥用、侵犯隐私或其他侵权用途。

Cola2 没有经过 instruction tuning、preference tuning，也没有声明经过可部署级
安全对齐。

## 官方推理设置

- Low-noise 8-step Euler DiT，\(t_i=T(1-i/8)^2\)；
- DiT CFG 10；
- decoder CFG -0.9；
- 每个 decoder chunk 一个 latent span；
- 开启 dynamic duration 与 latent feedback；
- greedy token selection，`max_new_tokens=32`。

即使使用 greedy token selection，latent generation 仍具有随机性。Replay 时应
传入固定 seed 的 `torch.Generator`，详见[推理](inference_zh.md)。

## 评测

论文报告结果见
[eval_output/full_test_results.md](../eval_output/full_test_results.md)。
Cola2 成绩如下：

| 任务 | 准确率（%） |
|---|---:|
| RACE | 69.91 |
| NarrativeQA | 32.70 |
| HellaSwag | 41.75 |
| Story Cloze | 80.42 |
| LAMBADA | 63.00 |
| ARC-Easy | 87.62 |
| TruthfulQA | 54.10 |
| MMLU | 51.32 |
| **等权平均** | **60.10** |

每项 Cola2 数值是三次独立 latent noise 评测的均值。Baseline 与 Cola2 的训练
预算不是 compute-matched，各模型也使用不同 tokenizer 与数据统计方式。

归一化 benchmark JSONL 将随模型外部分发。Prompt template、scorer、预期行数和
hash 保留在本仓库，详见 [Benchmark](benchmark_zh.md) 与
[复现](reproduction_zh.md)。

## 局限与风险

- 总体 benchmark 质量仍低于表中更强的 baseline。
- 长文本可能重复、语法错误、失去连贯性或偏离主题。
- 公开评测仅包含英文任务，不能证明多语言、代码、数学、长上下文、事实、鲁棒性
  或安全能力。
- 精确输出取决于 sampled latent noise，并可能随软件与硬件环境变化。
- 生成文本可能复现训练数据中的事实错误、刻板印象、有害内容、版权文本、个人
  信息或其他不良模式。
- Benchmark 覆盖范围有限，也可能继承上游数据集的污染或标注问题。

使用前应执行面向实际应用的测试、输出验证、访问控制和安全审核。

## 隐私与安全

除非部署已经为此明确设计并通过审核，否则不要输入 secret 或敏感个人数据。漏洞
应通过 [SECURITY.md](../SECURITY.md) 中的私密渠道报告，不应创建公开 issue。

## 许可证

仓库代码采用 [Apache License 2.0](../LICENSE)。模型权重、tokenizer、
归一化 benchmark 文件和上游数据集仍遵循随相应产物发布的条款。详见
[NOTICE](../NOTICE)。
