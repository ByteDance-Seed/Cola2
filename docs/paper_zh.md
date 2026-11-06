# 论文概览

[English](paper.md) · [文档索引](README_zh.md)

## 基本信息

**标题：** *Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous
Semantic Compression and Balanced Discrete Generation*

**作者：** Hongcan Guo、Yian Zhao、Feng Wang、Shen Nie、Guoqiang Wei、
Yan Zeng、Hengshuang Zhao。

**单位：** 香港大学、ByteDance Seed、北京大学、中国人民大学。

[项目主页](https://hongcanguo.github.io/Cola2) ·
论文与 arXiv 信息：**即将发布**

## 动机

层次化语言模型将语义规划与 token 实现分开。模型效果取决于两个相互耦合的
问题：压缩后的连续表示应该保留什么信息，以及离散 decoder 应如何将这些信息与
自身语言建模能力结合。Cola2 联合研究这两个问题，而不是把 latent encoder 与
token decoder 当作彼此独立的模块。

## 方法

Cola2 包含三个协同适配的部分：

1. **动态语义压缩。** 文本被划分为变长 span；无需辅助 loss 的负反馈机制控制
   平均压缩率，每个 span 被映射为紧凑的连续 latent。
2. **Latent 生成。** Block-causal DiT 学习 latent block 上的 flow-matching
   prior。Flow-matching 目标在 latent target 处 stop-gradient，避免 generator
   为了更容易预测而牺牲 token 语义。
3. **平衡的离散生成。** Token decoder 使用 clean、noisy 和 dropped latent
   condition 训练；decoder classifier-free guidance 调节遵循 latent 与自主补全
   token 细节之间的平衡。

推理时，模型按照已经提交的 span layout 重新编码生成文本。Latent feedback
使后续 latent history 与实际输出的文本保持一致。

## 最终 checkpoint 训练

论文报告的发布 checkpoint 使用经 Qwen3-4B tokenizer 重新分词的 OLMo 2
预训练文本。Decoder 和 VAE encoder embedding 从 Qwen3-4B-Base 初始化。

- **Stage I：** 使用 52.43B source tokens 训练动态压缩 VAE。
- **Stage II：** 使用额外 125.83B source tokens 联合更新 VAE 与约 4B 参数的
  latent DiT；DiT 目标使用 detached VAE latent。
- **Stage III：** 冻结 Stage-II EMA VAE，使用额外 167.77B source tokens
  继续训练 DiT。

总计为 **346.03B source tokens**。本仓库不包含训练实现、optimizer state 或
训练语料。

## 论文结论

组件实验支持三点结论：

- 动态分段与生成式监督有助于学习语义压缩表示；
- decoder 在条件重建与自主生成之间保持平衡时表现最好；
- latent 表示和 token generator 能从协同适配中获益。

最终 checkpoint 在八项任务上报告 **60.10** 的等权平均成绩。其 HellaSwag 和
Story Cloze 结果高于论文所列 Llama-3-8B，但总体平均仍低于更强的 baseline。
各模型的数据、tokenizer、目标函数和 token 统计方式不同，因此该表用于说明
checkpoint 所处水平，并非 compute-matched 对比。

每项 Cola2 论文成绩是三次独立 latent 噪声评测的均值。Token decoding 使用
greedy，随机性来自 DiT 初始高斯噪声。详见[复现协议](reproduction_zh.md)。

## 与本仓库的对应关系

论文讨论了更广泛的层次化框架，以及 autoregressive 与 masked-diffusion token
decoder 的受控实验。发布的 `Cola2ForConditionalGeneration` checkpoint 提供
[架构说明](architecture_zh.md)中的 Full-DiT latent generator 与 span-causal
token realization 路径。如果实验术语不能与公开 API 一一对应，应以发布代码的
文档为准。

## 引用状态

正式发表信息可用前，请使用 [CITATION.md](../CITATION.md) 中的软件引用。后续会
补充 arXiv 标识，不改变软件接口。
