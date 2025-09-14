## 项目索引（快速上手与组件关系图）

本项目是一个多模态VQA训练与评测框架，核心由三部分协同：
- CFRF（基于 BAN 与 AoA 的细粒度视觉-文本交互与分类器）
- XVLM 预训练多模态编码-解码器（用于粗粒度答案打分/重排序）
- ISubGVQA 场景图推理分支（图结构理解与答案概率）

训练入口位于 `main.py`，评测与预测入口位于 `evaluate.py`。

## 快速开始

运行训练（单机单卡，示例参数）：

```bash
python main.py \
  --dataset GQA \
  --checkpoint pretrained/xvlm.ckpt \
  --config ./xvlm/configs/VQA_480.yaml \
  --output saved_models/GQA/run1 \
  --batch_size 32 --epochs 20 --xvlm_lr 5e-5
```

分布式多卡（建议使用 torchrun；本项目已内置 `xvlm.utils.init_distributed_mode`）：

```bash
torchrun --nproc_per_node=4 --master_port=29500 main.py \
  --dataset GQA \
  --checkpoint pretrained/xvlm.ckpt \
  --config ./xvlm/configs/VQA_480.yaml \
  --output saved_models/GQA/run_ddp \
  --batch_size 32 --epochs 20 --xvlm_lr 5e-5 --distributed
```

评测/预测：

```bash
python evaluate.py \
  --dataset GQA \
  --checkpoint pretrained/xvlm.ckpt \
  --config ./xvlm/configs/VQA_480.yaml \
  --input saved_models/GQA/run1 \
  --epoch 12 --split test --batch_size 64
```

## 目录结构（关键文件）

- `main.py`: 训练入口；构建数据集与 `DataLoader`、搭建模型、优化器与训练循环调用。
- `evaluate.py`: 评测与预测入口；加载权重、构建数据、运行 `evaluate/predict` 并导出 JSON 结果。
- `src/FFOE/`
  - `dataset.py`: `GQAFeatureDataset` / `VQAFeatureDataset`，加载 HDF5 特征、标注与图像路径；返回 VQA 三元组与场景图。
  - `base_model.py`: 模型组装：`CFRF_Model`（BAN+AoA）+ `XVLM` + `ISubGVQA`，并行前向，`adapted_w` 自适应融合概率。
  - `train.py`: 训练/评测核心循环；优化器（`Adamax` + `BertAdam`）、梯度累积、混合精度（可选）。
  - `optimized_collate.py`: 优化版 `collate_fn`，动态 pad、`torch_geometric` 场景图批处理、可选性能监控。
  - 其它：`trainer.py`（若存在于本地，封装单步训练）、工具函数位于 `src/utils.py`。
- `xvlm/`
  - `model_vqa.py`: `XVLM` 封装（视觉编码 + 文本编解码）；训练/推理两种路径（loss / logits）。
  - `dataset/`: XVLM 原生数据构造与 `collate`（本项目主要复用自定义 `GQA/VQA` 数据集）。
  - `utils/__init__.py`: 分布式初始化、度量与工具集合。
  - `configs/VQA_480.yaml`: XVLM 侧训练/推理超参（图像尺寸、编码器、batch 配置等）。
- `ISubGVQA/`
  - `models/build.py`: `build_model(args)` 返回 `ISubGVQA`，支持 IMLE、MGAT、掩码等开关。
  - `datasets/build.py`: ISubGVQA 原生 GQA 数据集构建（本工程中从 `src/FFOE/dataset` 取 batch，同时单独加载场景图）。
  - `sampling/methods/aimle.py`: I-MLE 实现（隐式最大似然，离散结构可导近似）。
- `src/`
  - `attention.py`, `bc.py`, `classifier.py`, `counting.py`, `activation.py` 等：BAN/AoA 相关模块与分类器等基础组件。
  - `language_model.py`: 词嵌入与问题编码（LSTM/GRU/BERT）。

## 数据到训练的端到端流程

```mermaid
flowchart TD
  A[配置与CLI
  - main.py args
  - xvlm/config.yaml
  - optimization_config.yaml 可选] --> B[数据集构建
  - GQA/VQAFeatureDataset
  - HDF5特征、答案、实体
  - 图像读取与transform
  - 场景图 GQA/VQA SceneGraphs]
  B --> C[DataLoader
  - optimized_collate
  - 动态pad/Batch图]
  C --> D[CFRF_Model 组装]
  D --> D1[BAN + AoA
  - 细粒度交互
  - SimpleClassifier→ban_probs]
  D --> D2[XVLM
  - 视觉编码+文本解码
  - compute_logits→xvlm_probs]
  D --> D3[ISubGVQA
  - 图神经推理
  - softmax→isub_probs]
  D1 --> E[自适应融合 adapted_w
  prob_add=sum(w_i * probs_i)]
  D2 --> E
  D3 --> E
  E --> F[Loss/Score
  - 交叉熵或VQA得分]
  F --> G[train.py 训练循环
  - Adamax + BertAdam(XVLM)
  - 混合精度/梯度累积
  - 断点与日志]
```

## 模型拼装与前向（核心要点）

- 细粒度（BAN+AoA）
  - 问题/统计词/实体经 `WordEmbedding` 与 `QuestionEmbedding` 编码；
  - `build_ban_fusion` 构建 `BiAttention + BCNet + FCNet (+Counter)`，多次 AoA 增强；
  - `SimpleClassifier` 输出 `ban_logits → softmax → ban_probs`。
- 粗粒度（XVLM）
  - 视觉编码 + 文本编码，按答案表重排序/打分：`compute_logits` 生成 `xvlm_probs`（完整答案空间概率）。
- 场景图（ISubGVQA）
  - 以 `torch_geometric` 的 `Data/Batch` 输入；输出 `isub_probs`。
- 自适应融合
  - `adapted_w ∈ R^{3×num_ans}`（可学习），对 `[isub, xvlm, ban]` 概率逐类归一化后加权求和，得到 `fusion_probs`。

## 训练循环与优化

- 优化器
  - XVLM 参数：`BertAdam(lr=args.xvlm_lr)`；其余参数：`Adamax(lr=args.lr)`。
  - 注意：`train.py` 中默认按 `model.module.XVLM_model` 取参数，未分布式时需改为 `model.XVLM_model`。
- 训练技巧
  - 梯度累积：`--update_freq` 与 `--gradient_accumulation_steps`；
  - 混合精度：`--mixed_precision`；
  - 动态 batch：`--dynamic_batch_size --max_memory_usage`；
  - 梯度检查点：`--gradient_checkpointing`（在 `CFRF_Model.enable_gradient_checkpointing` 中透传）。
- DataLoader
  - `optimized_collate_fn` 动态 pad；`persistent_workers`、`prefetch_factor` 可配置。

## 配置总览

- CLI（见 `main.py`）
  - 数据与通用：`--dataset {GQA,VQA}`，`--output`，`--epochs`，`--batch_size`，`--update_freq`，`--gpu/--device`。
  - 模型：`--AOA_layers`，`--num_hid`，`--gamma`，`--omega_q`，`--omega_v`，`--dropout`，`--activation`。
  - XVLM：`--checkpoint`，`--config`，`--xvlm_lr`，`--bs`（覆盖 YAML 的 `batch_size_train`）。
  - ISubGVQA：`--mgat_layers`，`--mgat_masks`，`--use_masking`，`--use_instruction`，`--clip_url` 等。
  - 性能优化：`--dynamic_batch_size`，`--mixed_precision`，`--memory_efficient_attention`，`--gradient_checkpointing`。
- XVLM YAML（见 `xvlm/configs/VQA_480.yaml`）
  - 图像尺寸与编码器、文本编码器、答案表、batch、优化器/调度器等。
- 额外优化配置（可选 `optimization_config.yaml`）
  - 可覆盖内存/训练/模型层面的默认参数（在 `main.py` 中 `apply_optimization_config`）。

## 评测与导出

- `train.evaluate(...)`：逐 batch 计算 VQA 指标，返回 `cfrf/fg/coarse/ens/mid/bound` 得分。
- `evaluate.py` 中 `predict(...)`：导出多种预测 JSON（融合/细粒度/粗粒度/场景图）。

## 多卡并行要点（重塑提示）

- 分布式初始化：已由 `xvlm.utils.init_distributed_mode(args)` 完成（读取 `RANK/LOCAL_RANK/WORLD_SIZE`）。
- 模型封装：`main.py` 在 `--distributed` 时使用 `DistributedDataParallel(find_unused_parameters=True)`。
- 优化器参数划分：当前 `train.py` 默认访问 `model.module.XVLM_model`，若单卡（未 DDP）需改为 `model.XVLM_model` 或使用 `getattr(model, 'module', model)` 兼容写法。
- 数据采样：在分布式下，通过 `xvlm.dataset.create_sampler` 为 train/val 传入 `DistributedSampler`。
- 启动方式：建议 `torchrun --nproc_per_node=N ... --distributed`。

## 外部资源与数据要求

- 预训练权重：`--checkpoint` 指向 XVLM 权重；
- 文本编码器：`config['text_encoder']`（如 `data/bert-base-uncased`）；
- 答案列表与标注：`data/finetune/answer_list.json`、`{gqa|vqa}/cache/*.pkl`；
- 图像根目录：`config['gqa_root'|'vqa_root'|'vg_root']`；
- 场景图：`ISubGVQA.datasets.scene_graph` 相关缓存；
- CLIP Tokenizer 缓存：`--clip_url` 指向本地 HuggingFace 缓存路径。

## 常见坑位（快速排查）

- 单卡训练时报 `model.module` 属性错误：将 `train.py` 中对 `model.module.XVLM_model` 的访问改为兼容写法。
- RoBERTa 模式：`xvlm/model_vqa.py` 中对 RoBERTa 有 TODO 提示，默认关闭 `use_roberta`。
- AoA 依赖：`from aoa_pytorch import AoA` 需在环境中可用。
- HDF5 / 路径：确保 `data/{gqa|vqa}` 的 HDF5、cache 与图像路径与配置一致。

---

如果你计划把该项目系统性重构为“多卡并行 + 更稳定可移植”的版本，优先建议：
- 将 `train.py` 中的 `.module` 访问统一改为 `m = getattr(model, 'module', model)`；
- 统一使用 `torchrun` 启动并强制 `--distributed`；
- 在 `DataLoader` 上使用 `DistributedSampler` 并在每个 epoch 调用 `sampler.set_epoch(epoch)`；
- 按需开启 `--mixed_precision` 与 `--gradient_checkpointing`，同时监控显存以调参 `--batch_size/--update_freq`；
- 将可变的路径/缓存与 batch 配置固化在一个 `experiment.yaml`，通过 `optimization_config.yaml` 进行增量覆盖。


