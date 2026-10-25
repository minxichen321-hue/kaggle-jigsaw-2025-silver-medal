# Jigsaw 2025 Silver Medal Solution

[![Kaggle Silver Medal](https://img.shields.io/badge/Kaggle-Silver%20Medal-silver)](https://www.kaggle.com/certification/competitions/opennlp/jigsaw-agile-community-rules)
[![Rank 41 of 2445](https://img.shields.io/badge/Rank-41%20%2F%202445-blue)](https://www.kaggle.com/certification/competitions/opennlp/jigsaw-agile-community-rules)
![Top 1.68 percent](https://img.shields.io/badge/Top-1.68%25-success)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)

**Kaggle 银牌 · 第 41 / 2,445 名 · 全球前 1.68%**

2025 年 [Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) 竞赛方案。结合 **Qwen3-4B QLoRA、GTE 三元组度量学习与 DeBERTa-v3 分类器**，通过加权排名融合预测 Reddit 评论是否违反给定社区规则。

**Silver medal solution for Kaggle's Jigsaw - Agile Community Rules Classification (2025), placing 41st out of 2,445 teams, approximately top 1.68%.** The ensemble combines a QLoRA-tuned Qwen3-4B language model, GTE triplet embeddings, and a DeBERTa-v3 classifier.

## 比赛与成绩

| 项目 | 信息 |
| --- | --- |
| 竞赛 | [Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) |
| 平台 / 年份 | Kaggle / 2025 |
| 获奖者 | Chen Minxi |
| Kaggle 用户 | [opennlp](https://www.kaggle.com/opennlp) |
| 奖项 | **Competition Silver Medal（竞赛银牌）** |
| 最终名次 | **41 / 2,445 teams** |
| 排名比例 | **Top 1.68%**，按 `41 / 2445 × 100%` 计算 |
| 成绩证明 | [Kaggle 官方银牌证书](https://www.kaggle.com/certification/competitions/opennlp/jigsaw-agile-community-rules) |
| 评价指标 | Column-averaged AUC，见[官方赛题说明](https://www.kaggle.com/c/jigsaw-agile-community-rules) |

## 任务与方法

模型接收评论 `body`、社区规则 `rule`、社区名称 `subreddit`，以及各两条违规/合规示例，输出违规分数。训练集只覆盖部分规则，模型还需要处理测试阶段出现的新规则。

| 分支 | 方法 | 融合权重 |
| --- | --- | --- |
| **Qwen3-4B-Base** | Few-shot 提示；4-bit NF4 QLoRA；LoRA rank 32；DeepSpeed ZeRO-2 训练；vLLM 双 GPU 推理；约束输出 Yes/No | **0.7** |
| **GTE-base-en-v1.5** | 用规则、合规评论、违规评论构造三元组；比较评论到两类示例中心的欧氏距离 | **0.3** |
| **DeBERTa-v3-base** | 规则与评论拼接；URL 语义特征；最后两层隐藏状态与注意力池化；多 Dropout、对抗扰动、EMA、标签平滑 | **0.5** |

训练流程还利用了数据中明确给出的 `positive_example_1/2` 和 `negative_example_1/2`：它们分别表示违规和合规示例，可以构造额外监督样本。代码没有读取测试评论的隐藏 `rule_violation` 标签。

每个模型的预测先按 `row_id` 对齐，再计算整个预测集上的归一化排名：

```python
rank_score = prediction.rank(method="average") / (N + 1)
blend = 0.7 * rank_llm + 0.5 * rank_deberta + 0.3 * rank_gte
```

这是原 Notebook 中的**全局排名融合**。权重之和为 1.5，输出是用于排序的分数，可能大于 1，并非校准概率。详见[方案说明](docs/solution.md)。

## 仓库内容

```text
.
├── notebooks/jigsaw-ensemble-code.ipynb   # 原代码单元，清空输出并增加章节说明
├── scripts/
│   ├── train_qwen4b.py                   # 从 Notebook 提取
│   ├── inference_qwen4b.py               # 从 Notebook 提取
│   ├── train_and_infer_triplet_gte.py     # 从 Notebook 提取
│   ├── train_and_infer_cls_deberta_v3.py  # 从 Notebook 提取
│   ├── install_kaggle_dependencies.sh    # 原离线安装命令
│   └── blend_predictions.py              # 融合工具，增加输入与 row_id 校验
├── configs/accelerate_config.yaml        # 原双 GPU DeepSpeed 配置
├── docs/
│   ├── solution.md                      # 算法说明与代码细节
│   ├── reproduction.md                  # Kaggle 输入与运行步骤
│   └── Jigsaw2025赛后讲解.pdf             # 随附赛后讲解，保留原署名
├── tests/test_blend_predictions.py
└── requirements.txt                     # 仅用于融合工具及其测试
```

## 运行

完整训练和推理以 **Kaggle Linux / CUDA / 双 GPU** 为目标环境。原 Notebook 记录的 Python 版本为 3.11.13，启用了 T4 GPU，并关闭互联网。比赛数据、预训练权重和离线依赖包需要单独添加，仓库不包含这些大文件。

**推荐使用 Notebook：** 将 [notebooks/jigsaw-ensemble-code.ipynb](notebooks/jigsaw-ensemble-code.ipynb) 导入 Kaggle，按[复现指南](docs/reproduction.md)添加输入并依次执行。原代码在交互运行时会对 Qwen 训练数据取前 64 条；完整训练模式、路径和依赖说明均在指南中列出。

如已有三个模型的预测 CSV，可以在本地仅运行融合：

```bash
python -m pip install -r requirements.txt
python scripts/blend_predictions.py \
  --deberta sub_deberta_v3.csv \
  --gte sub_gte.csv \
  --llm sub_llm.csv \
  --output submission.csv
```

输入文件须包含 `row_id,rule_violation`，三份文件的 ID 集合必须相同。工具会检查重复 ID、缺失行和无效分数。运行相关检查：

```bash
python -m unittest discover -s tests -v
```

## 资料与致谢

- [Kaggle 官方竞赛](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)与[官方成绩证书](https://www.kaggle.com/certification/competitions/opennlp/jigsaw-agile-community-rules)。
- 原 Notebook 引用的模型、数据包与工具来源见[复现指南](docs/reproduction.md#输入资源)。
- 算法文字根据随附项目介绍与实际代码整理；[赛后讲解 PDF](docs/Jigsaw2025赛后讲解.pdf)保留原文标注的主讲人 **H 老师**。

本仓库保留原方案代码以供阅读与复现；完整 GPU 流程尚未在本次仓库整理中重跑。数据、模型及第三方资料的使用条件以各自来源为准。
