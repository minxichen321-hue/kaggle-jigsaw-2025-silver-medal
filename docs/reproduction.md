# 运行与复现指南

原方案依赖 Kaggle 的 Linux/CUDA 环境、双 GPU 和预先挂载的输入。建议先使用 [Notebook](../notebooks/jigsaw-ensemble-code.ipynb)，再按需要使用提取的脚本。

## 输入资源

以下链接来自原 Notebook 首个代码单元，其对应目录被代码直接引用：

| 资源 | 来源 | 预期路径 |
| --- | --- | --- |
| 比赛数据 | [Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) | `/kaggle/input/jigsaw-agile-community-rules/`，含 `train.csv`、`test.csv` |
| Qwen3-4B-Base | [Qwen-3 / Transformers / 4b-base / 1](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/4b-base/1) | `/kaggle/input/qwen-3/transformers/4b-base/1` |
| DeBERTa-v3-base | [Hugging Face DeBERTa variants](https://www.kaggle.com/datasets/verracodeguacas/huggingfacedebertav3variants) | `/kaggle/input/huggingfacedebertav3variants/deberta-v3-base` |
| GTE-base-en-v1.5 | [GTE dataset](https://www.kaggle.com/datasets/yeoyunsianggeremie/gte-base-en-v1-5) | `/kaggle/input/gte-base-en-v1-5` |
| 离线依赖 wheel 包 | [jigsaw-packages2](https://www.kaggle.com/code/hiranorm/jigsaw-packages2) | `/kaggle/input/jigsaw-packages2/whls/` |
| 随机性辅助包 | [deterministic](https://www.kaggle.com/code/wasupandceacar/deterministic) | `kagglehub.package_import('wasupandceacar/deterministic')` |

比赛数据需要通过 Kaggle 获取，并遵循数据访问条件。仓库只发布代码与说明，不打包比赛数据、模型权重或训练产物。

## 环境

原 Notebook 元数据记录 Python 3.11.13、T4 GPU、互联网关闭。Qwen 推理显式启动 GPU 0/1 两个进程；Accelerate 配置同样使用两个进程，因此单 GPU 或 macOS CPU 环境不能原样执行完整流程。

原安装单元使用离线 wheel 包，明确指定的版本包括：

```text
trl==0.21.0
optimum==1.27.0
auto-gptq==0.7.1
bitsandbytes==0.46.1
logits-processor-zoo==0.2.1
vllm==0.10.0
deepspeed==0.17.4
triton==3.2.0
```

`clean-text`、`peft`、`accelerate`、`datasets` 没有在原安装单元中固定版本；其余库继承 Kaggle 镜像环境。依赖还包括 PyTorch、Transformers、Sentence Transformers、scikit-learn、FAISS、KaggleHub、NumPy、pandas 和 tqdm。原材料没有完整的环境锁文件。

使用 Notebook 的原安装单元，或执行以下保留原安装顺序的脚本：

```bash
bash scripts/install_kaggle_dependencies.sh
```

根目录的 `requirements.txt` **只覆盖独立融合工具和测试**。它不能安装完整 GPU 训练环境。

## Notebook 执行

1. 将 `notebooks/jigsaw-ensemble-code.ipynb` 导入 Kaggle。
2. 挂载上表中的比赛数据、模型和离线包，选择双 GPU；确认实际目录与代码一致。
3. 依次运行依赖安装、Qwen、GTE、DeBERTa 和融合单元。
4. 从 `/kaggle/working/submission.csv` 获取最终输出。

**训练模式：** Qwen 的 `build_dataset()` 通过环境变量 `KAGGLE_IS_COMPETITION_RERUN` 判断是否为比赛提交重跑。该变量为空时只使用前 64 条训练样本；变量非空时才使用构造后的完整样本。需要在非提交模式进行完整实验时，可在训练前显式设置：

```python
import os
os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
```

设置非空字符串（包括 `"0"`）都会被原代码当作真值。这个变量只控制该分支的数据截断，不会赋予比赛参赛或提交资格。DeBERTa 在 `test.csv` 只有 10 行时，也会将训练样本截断到前 128 条。

## 使用提取的脚本

脚本保留原模型路径、算法和超参数。以下示例假设仓库位于 `/kaggle/working/kaggle-jigsaw-2025-silver-medal`，并已完成依赖和输入准备。顺序执行可释放上一阶段占用的 GPU 显存：

```bash
cd /kaggle/working
JIGSAW_REPO=/kaggle/working/kaggle-jigsaw-2025-silver-medal

KAGGLE_IS_COMPETITION_RERUN=1 accelerate launch \
  --config_file "$JIGSAW_REPO/configs/accelerate_config.yaml" \
  "$JIGSAW_REPO/scripts/train_qwen4b.py"

python "$JIGSAW_REPO/scripts/inference_qwen4b.py"
python "$JIGSAW_REPO/scripts/train_and_infer_triplet_gte.py"
python "$JIGSAW_REPO/scripts/train_and_infer_cls_deberta_v3.py"
python "$JIGSAW_REPO/scripts/blend_predictions.py" --output submission.csv
```

预期产物：

| 文件 | 内容 |
| --- | --- |
| `/kaggle/working/pseudo_lora/` | Qwen LoRA 权重 |
| `/kaggle/working/sub_llm.csv` | LLM 分支的 Yes 分数 |
| `/kaggle/working/sub_gte.csv` | GTE 距离差分数 |
| `/kaggle/working/sub_deberta_v3.csv` | DeBERTa 分类分数 |
| `/kaggle/working/submission.csv` | 三模型排名融合结果 |

## 本地融合与检查

独立融合工具不需要 GPU。它读取三份含 `row_id,rule_violation` 的 CSV，检查 ID 集合、重复行和非有限数值，再使用原公式计算融合结果：

```bash
python -m pip install -r requirements.txt
python scripts/blend_predictions.py \
  --deberta /path/to/sub_deberta_v3.csv \
  --gte /path/to/sub_gte.csv \
  --llm /path/to/sub_llm.csv \
  --output /path/to/submission.csv

python -m unittest discover -s tests -v
```

检查覆盖乱序行对齐、同分的平均秩，以及缺失预测、重复 ID、空输入、缺失列和无效分数等错误。完整训练与最终比赛分数仍需在原 GPU 环境和所需数据上单独复现。
