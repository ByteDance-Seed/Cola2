# 评测数据

[English](README.md) · [Benchmark 文档](../docs/benchmark_zh.md)

归一化 benchmark JSONL 与 Cola2 模型一起托管在
[ByteDance-Seed/Cola2](https://huggingface.co/ByteDance-Seed/Cola2)；大体积
评测文件不会提交到本源码仓库。

## 预期文件

将以下文件下载到本目录，或放到其他目录并设置
`GENERATE_TASK_DATA_DIR`：

```text
arc_easy.jsonl
hellaswag.jsonl
lambada.jsonl
mmlu.jsonl
narrativeqa.jsonl
race.jsonl
story_cloze.jsonl
truthfulqa.jsonl
```

随代码版本管理的 `manifest.json` 记录每个文件的名称、行数和 SHA-256。
Benchmark 会拒绝缺失文件与错误行数，prepared-data manifest 还会记录实际 hash
以便审计。

## Record 契约

Source row 包含非空 `id`、`question`、位于 `ground_truth` 或 `answer` 中的答案，
以及 `context`、`choices`、`type` 等任务字段。准备阶段会加入固定 prompt 和
canonical `task` 字段，不修改 ground truth 或 choices。

不要重排或去重。NarrativeQA 有意包含重复 source ID；准备过程会保留每一行，并
根据 source position 分配无歧义 benchmark ID。

## 来源

Record 来自以下公开 evaluation split：

- RACE
- HellaSwag
- LAMBADA
- ARC-Easy
- Story Cloze
- NarrativeQA
- TruthfulQA
- MMLU

每个上游数据集仍遵循其原始许可证、条款和引用要求。归一化文件只用于评测，不应
被视为模型训练数据。重新分发前请查阅 Hugging Face 发布页和上游 dataset card。
