# 🎯 超参数调优指南

## 📊 超参数分类

### 🚀 训练基础参数

| 参数 | 默认值 | 类型 | 说明 | 调优建议 |
|------|--------|------|------|----------|
| `--epochs` | 20 | int | 训练轮数 | 10-50，根据收敛情况调整 |
| `--lr` | 2.8e-4 | float | 主学习率 | 1e-4 到 5e-4 |
| `--xvlm_lr` | 5e-5 | float | XVLM学习率 | 1e-5 到 1e-4 |
| `--batch_size` | 32 | int | 批处理大小 | 16-64，根据GPU内存调整 |
| `--update_freq` | 4 | int | 参数更新频率 | 2-8，影响有效批大小 |
| `--seed` | 1204 | int | 随机种子 | 确保实验可重现 |

### 🧠 模型架构参数

#### BAN (Bilinear Attention Networks)
| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--gamma` | 2 | glimpse数量 | 1-4，影响注意力机制 |
| `--use_counter` | False | 是否使用计数器模块 | 根据任务需要开启 |
| `--counter_act` | 'zhang' | 计数器激活函数 | 通常保持默认 |

#### AOA (Attention on Attention)
| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--AOA_layers` | 6 | AOA层数 | 4-8，影响模型复杂度 |
| `--num_hid` | 1024 | 联合表示维度 | 512-2048，影响表达能力 |

#### 激活函数和正则化
| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--activation` | 'swish' | 激活函数 | 'relu', 'swish' |
| `--dropout` | 0.45 | Dropout率 | 0.3-0.6，防止过拟合 |

### 🔍 数据处理参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--question_len` | 12 | 问题最大长度 | 8-20，根据问题复杂度 |
| `--max_boxes` | 50 | 最大边界框数量 | 30-100，根据图像复杂度 |
| `--num_stat_word` | 30 | 统计词数量 | 20-50 |
| `--tfidf` | True | 是否使用TF-IDF | 通常保持True |

### 🎯 ISubGVQA特定参数

#### 图神经网络参数
| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--general_hidden_dim` | 300 | 隐藏层维度 | 256-512 |
| `--mgat_layers` | 4 | MGAT层数 | 2-6 |
| `--sampler_type` | 'imle' | 采样器类型 | 'imle', 'gumbel' |
| `--sample_k` | 2 | 采样K值 | 1-5 |
| `--nb_samples` | 1 | 样本数量 | 1-3 |

#### 控制参数
| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--alpha` | 1.0 | Alpha参数 | 0.5-2.0 |
| `--beta` | 10.0 | Beta参数 | 5.0-20.0 |
| `--tau` | 1.0 | Tau参数 | 0.5-2.0 |
| `--use_instruction` | 1 | 是否使用指令 | 0/1 |
| `--use_masking` | 1 | 是否使用掩码 | 0/1 |
| `--use_mgat` | 0 | 是否使用MGAT | 0/1 |

### 🔧 优化参数

#### 梯度相关
| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--clip_norm` | 0.25 | 梯度裁剪阈值 | 0.1-0.5 |
| `--gradient_accumulation_steps` | 1 | 梯度累积步数 | 1-8，模拟更大批大小 |

#### 内存优化
| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--dynamic_batch_size` | False | 动态批大小 | 内存不足时开启 |
| `--max_memory_usage` | 0.8 | 最大内存使用率 | 0.7-0.9 |
| `--mixed_precision` | False | 混合精度训练 | 通常开启以节省内存 |
| `--memory_efficient_attention` | False | 内存高效注意力 | 内存不足时开启 |

### 🎛️ 微调参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--omega_q` | 0.1 | 问题指令权重 | 0.05-0.2 |
| `--omega_v` | 0.01 | 图像语义权重 | 0.005-0.05 |
| `--fusion_ratio` | 0.1 | 融合比例 | 0.05-0.2 |
| `--topk` | 6 | Top-K值 | 3-10 |

## 🎯 调优策略

### 1. 基础调优流程

```bash
# 第一步：调整学习率和批大小
python main.py \
  --lr 2.8e-4 \
  --xvlm_lr 5e-5 \
  --batch_size 32 \
  --epochs 20

# 第二步：调整模型复杂度
python main.py \
  --AOA_layers 6 \
  --num_hid 1024 \
  --dropout 0.45

# 第三步：调整ISubGVQA参数
python main.py \
  --general_hidden_dim 300 \
  --mgat_layers 4 \
  --alpha 1.0 \
  --beta 10.0
```

### 2. 内存优化调优

```bash
# 内存不足时的配置
python main.py \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --mixed_precision \
  --memory_efficient_attention \
  --max_memory_usage 0.7
```

### 3. 性能优化调优

```bash
# 高性能配置
python main.py \
  --batch_size 64 \
  --update_freq 2 \
  --dynamic_batch_size \
  --max_memory_usage 0.9
```

## 📈 调优建议

### 🎯 按任务类型调优

#### GQA数据集
```bash
python main.py \
  --dataset GQA \
  --lr 2.8e-4 \
  --batch_size 32 \
  --AOA_layers 6 \
  --general_hidden_dim 300 \
  --mgat_layers 4
```

#### VQA数据集
```bash
python main.py \
  --dataset VQA \
  --lr 3e-4 \
  --batch_size 24 \
  --AOA_layers 8 \
  --general_hidden_dim 512 \
  --mgat_layers 6
```

### 🔍 按硬件配置调优

#### 8GB GPU
```bash
python main.py \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --mixed_precision \
  --max_memory_usage 0.7
```

#### 24GB GPU
```bash
python main.py \
  --batch_size 48 \
  --gradient_accumulation_steps 1 \
  --max_memory_usage 0.9
```

## 🧪 实验设计

### 1. 学习率实验
```bash
# 实验1：基础学习率
python main.py --lr 2.8e-4 --xvlm_lr 5e-5

# 实验2：较高学习率
python main.py --lr 3.5e-4 --xvlm_lr 7e-5

# 实验3：较低学习率
python main.py --lr 2e-4 --xvlm_lr 3e-5
```

### 2. 模型复杂度实验
```bash
# 实验1：简单模型
python main.py --AOA_layers 4 --num_hid 512

# 实验2：中等模型
python main.py --AOA_layers 6 --num_hid 1024

# 实验3：复杂模型
python main.py --AOA_layers 8 --num_hid 2048
```

### 3. ISubGVQA参数实验
```bash
# 实验1：基础配置
python main.py --mgat_layers 4 --alpha 1.0 --beta 10.0

# 实验2：增强配置
python main.py --mgat_layers 6 --alpha 1.5 --beta 15.0

# 实验3：保守配置
python main.py --mgat_layers 3 --alpha 0.8 --beta 8.0
```

## 📊 监控指标

### 训练指标
- 训练损失下降趋势
- 验证准确率
- GPU内存使用率
- 训练速度（样本/秒）

### 调优检查点
1. **收敛性**：损失是否稳定下降
2. **过拟合**：验证集性能是否下降
3. **内存使用**：是否出现OOM错误
4. **训练速度**：是否达到预期速度

## 🎯 最佳实践

1. **渐进式调优**：一次只调整一个参数
2. **记录实验**：保存每次实验的配置和结果
3. **早停机制**：设置合理的早停条件
4. **交叉验证**：使用不同的随机种子验证
5. **资源监控**：实时监控GPU和内存使用

## 📝 示例配置

### 高性能配置
```bash
python main.py \
  --checkpoint pretrained/model \
  --config xvlm/configs/VQA_480.yaml \
  --output saved_models/high_performance \
  --lr 3e-4 \
  --xvlm_lr 6e-5 \
  --batch_size 48 \
  --AOA_layers 8 \
  --num_hid 1536 \
  --general_hidden_dim 512 \
  --mgat_layers 6 \
  --alpha 1.2 \
  --beta 12.0 \
  --mixed_precision \
  --max_memory_usage 0.9
```

### 内存优化配置
```bash
python main.py \
  --checkpoint pretrained/model \
  --config xvlm/configs/VQA_480.yaml \
  --output saved_models/memory_optimized \
  --lr 2.5e-4 \
  --xvlm_lr 4e-5 \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --AOA_layers 4 \
  --num_hid 768 \
  --general_hidden_dim 256 \
  --mgat_layers 3 \
  --alpha 0.8 \
  --beta 8.0 \
  --mixed_precision \
  --memory_efficient_attention \
  --max_memory_usage 0.7
```

现在你可以根据这个指南系统地调整超参数了！建议从基础参数开始，逐步调整到更复杂的参数。 