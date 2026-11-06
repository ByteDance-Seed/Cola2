# Cola2 文档

[English](README.md) · [仓库首页](../README_zh.md)

本文档描述发布版推理实现与公开评测契约。模型权重、tokenizer 和归一化
benchmark JSONL 将通过同一个 Hugging Face 仓库分发，链接目前为
**即将发布**。

## 入门

- [论文概览](paper_zh.md)与
  [arXiv:2610.31305](https://arxiv.org/abs/2610.31305)：论文动机、方法、
  训练阶段和主要结论。
- [架构](architecture_zh.md)：发布模块与 block 生成循环。
- [推理](inference_zh.md)：checkpoint 加载、固定 seed replay、CLI 和故障排查。
- [配置](configuration_zh.md)：模型字段、生成参数、别名和环境变量。

## 评测

- [Benchmark](benchmark_zh.md)：任务、prompt、scoring、数据契约和产物目录。
- [复现协议](reproduction_zh.md)：quick、完整和论文式三 seed 运行。
- [评测数据](../eval_data/README_zh.md)：外部分发与 manifest 校验。
- [论文结果](../eval_output/full_test_results.md)：随论文报告的结果表。

## 发布与治理

- [模型卡](model_card_zh.md)
- [贡献指南](../CONTRIBUTING.md)
- [安全政策](../SECURITY.md)
- [引用](../CITATION.md)
- [Apache License 2.0](../LICENSE)
- [第三方声明](../NOTICE)

## 仓库边界

这是一个推理与评测仓库，有意不包含训练代码、checkpoint、tokenizer、
benchmark JSONL、论文 LaTeX 和图片源码。论文概览会总结与理解模型有关的训练
事实，但不代表本仓库可以复现训练过程。
