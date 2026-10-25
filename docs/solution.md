# Jigsaw 2025 银牌方案说明

本项目对应 Kaggle **Jigsaw - Agile Community Rules Classification**，获奖者 **Chen Minxi（opennlp）**，成绩为 **银牌，41 / 2,445，约前 1.68%**，可通过[官方证书](https://www.kaggle.com/certification/competitions/opennlp/jigsaw-agile-community-rules)核对。

方案把生成式判断、语义度量和判别式分类结合起来。以下细节按仓库中的代码整理；随附[赛后讲解](Jigsaw2025赛后讲解.pdf)提供了整体思路。

## 数据和监督信号

输入包括 `body`、`rule`、`subreddit`、`positive_example_1/2`、`negative_example_1/2`。这里 `positive_example` 是**违规**示例，`negative_example` 是**合规**示例。训练标签为 `rule_violation`，1 表示违规，0 表示合规。

三个分支使用的样本构造并不完全相同：

- **Qwen：** 使用训练集原始标签，并按规则从测试行抽取 5%，将这些行中已经提供的正负示例转为额外监督样本。构造提示时，作为待判断评论的示例不会同时用作同侧演示。
- **GTE：** 从测试输入给出的示例构造三元组，并在同一规则下随机重新配对以增强数据。
- **DeBERTa：** 合并训练集原始样本、训练集示例和测试输入中的示例，再按评论与规则去重。

这是对公开输入中示例标签的利用。代码没有把未知测试评论直接赋予模型生成的伪标签，也没有使用其隐藏真值。

## Qwen3-4B

文件：[训练](../scripts/train_qwen4b.py)、[推理](../scripts/inference_qwen4b.py)、[分布式配置](../configs/accelerate_config.yaml)。

提示包含系统指令、subreddit、规则、一条违规示例、一条合规示例和待判断评论。监督目标为单词 `Yes` 或 `No`。

训练采用 NF4 4-bit 量化与 QLoRA，LoRA 的 `r=32`、`alpha=64`，作用于注意力投影和 MLP 投影。主要参数为 1 epoch、学习率 `2e-4`、每设备 batch size 4、梯度累积 4、余弦学习率调度与梯度检查点。配置使用两进程 DeepSpeed ZeRO-2。

推理使用 vLLM 动态加载 LoRA，以 `MultipleChoiceLogitsProcessor` 将生成限制为 Yes/No；使用 Yes 的分数作为违规信号。两个 GPU 分别处理测试集的一半，最后合并结果，输出 `sub_llm.csv`。

## GTE 三元组度量学习

文件：[train_and_infer_triplet_gte.py](../scripts/train_and_infer_triplet_gte.py)。

三元组定义为：

```text
anchor   = 规则文本
positive = 合规评论（negative_example）
negative = 违规评论（positive_example）
```

三元组中的 positive/negative 表示“接近/远离 anchor”的训练角色，和比赛字段中的同名概念不同。代码使用 `TripletLoss`，margin 为 0.25，实际主流程微调 1 epoch、batch size 32、学习率 `2e-5`，最大序列长度 256。

模型对评论和示例生成归一化嵌入。对每条规则分别汇总违规示例和合规示例，求均值后再次归一化，形成两类中心。预测分数为：

```text
score = distance(comment, compliant_centroid)
      - distance(comment, violating_centroid)
```

评论越靠近违规中心、越远离合规中心，分数越高。输出为 `sub_gte.csv`。代码中部分日志和输出目录仍沿用 `bge` 命名，实际加载的模型路径为 GTE。

## DeBERTa-v3 分类器

文件：[train_and_infer_cls_deberta_v3.py](../scripts/train_and_infer_cls_deberta_v3.py)。

输入为规则文本、`[SEP]` 和评论，并在评论后追加从 URL 域名和路径提取的关键词。模型取最后两层隐藏状态，按代码中的 `(2.2 × last + previous) / 2` 组合后，通过注意力池化和线性分类头输出两个类别的 logits。

分类头包含五种 Dropout 比率，训练时对五路 logits 求平均；标准评估模式下 Dropout 关闭，因此不构成推理阶段的随机 Monte Carlo 采样。交叉熵使用 0.10 标签平滑。

自定义 Trainer 还包含嵌入对抗扰动与 EMA：EMA 从训练进度的 80% 开始更新，衰减系数为 0.99，并在预测时交换权重。主流程训练 3 epochs，学习率 `2e-5`，batch size 8，最大序列长度 512；输出 `sub_deberta_v3.csv`。

## 加权排名融合

三个分支的原始分数尺度不同：LLM 的 Yes 分数、GTE 的距离差、DeBERTa 的 softmax 分数不能直接等同。原 Notebook 按 `row_id` 排序，然后对**整份预测集**计算平均秩：

```python
r = scores.rank(method="average") / (N + 1)
blend = 0.7 * r_llm + 0.5 * r_deberta + 0.3 * r_gte
```

此实现没有按规则分组计算秩。权重之和为 1.5，最终值可能大于 1；它是排序分数，不代表校准后的违规概率。除以 1.5 可改变数值范围而保持排序，但仓库中的原 Notebook 与独立融合工具均保留原始公式。

新增的[独立融合工具](../scripts/blend_predictions.py)校验三份文件是否具有相同的 `row_id`，避免按位置相加时静默混入缺行或错行。

## 成绩与复现边界

官方证书证明获奖者、银牌和最终名次。原材料未提供本 Notebook 对应提交的 Public/Private AUC 数值，也未提供消融实验，所以本仓库不补写这些数据。

仓库整理保留了原 Notebook 的所有代码单元；四个模型脚本与 YAML 从其 `%%writefile` 单元提取。清除了 Notebook 的输出及运行时间元数据，便于公开浏览。此次整理检查了代码语法和融合工具；没有重新训练三套 GPU 模型，不能据此承诺逐位复现原比赛成绩。

## English overview

This silver medal project addresses rule-conditioned Reddit comment classification. It combines Qwen3-4B QLoRA fine-tuning and constrained Yes/No inference, GTE triplet learning with rule-specific example centroids, and a DeBERTa-v3 classifier with attention pooling. The three prediction streams are aligned by row ID and blended using global normalized ranks, with weights of 0.7, 0.3, and 0.5 respectively.

The official Kaggle certificate records **41st of 2,445 teams** for **Chen Minxi (`opennlp`)**. The archived notebook expects the original Kaggle GPU environment and attached resources; the complete training pipeline has not been rerun as part of this repository preparation.
