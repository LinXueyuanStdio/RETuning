# Stock Classification with Time-Series-Library

本目录包含使用 Time-Series-Library 中的经典时间序列模型进行股票趋势预测的代码。

## 任务描述

**预测目标**：三分类任务，基于隔夜收益率 `(open - pre_close) / pre_close`
- **up (2)**: 隔夜收益率 > 3%
- **hold (1)**: -3% <= 隔夜收益率 <= 3%
- **down (0)**: 隔夜收益率 < -3%

**评估指标**：F1、Accuracy、Recall

**输出文件**：包含 `[date, code, open, high, low, close, pct_change, label, predicted_label]` 的 CSV 文件

## 支持的模型

- TimesNet
- PatchTST
- Informer
- DLinear
- Autoformer
- TimeMixer

## 数据划分

**Mode 1**: 使用所有历史数据（2015-2024.11）作为训练集，2024年12月作为测试集

**Mode 2**: 仅使用2024年数据（Jan-Nov）作为训练集，2024年12月作为测试集

**滑动窗口大小**: 5, 10, 20, 60 交易日

## 目录结构

```
stock_classification/
├── __init__.py
├── build_stock_classification_dataset.py  # 数据集构建脚本
├── stock_data_loader.py                   # 数据加载器
├── exp_stock_classification.py            # 实验类
├── run_stock_classification.py            # 主入口
├── summarize_results.py                   # 结果汇总脚本
├── dataset/                               # 构建的数据集 (运行后生成)
│   └── Stock_mode{1,2}_sl{5,10,20,60}/
│       ├── TRAIN.npz
│       ├── TEST.npz
│       ├── train_meta.csv
│       ├── test_meta.csv
│       └── label_mapping.txt
└── README.md

scripts/stock_classification/
├── build_all_datasets.sh                  # 构建所有数据集
├── run_all.sh                             # 运行所有实验
├── TimesNet_stock.sh
├── PatchTST_stock.sh
├── Informer_stock.sh
├── DLinear_stock.sh
├── Autoformer_stock.sh
└── TimeMixer_stock.sh
```

## 使用方法

### 1. 构建数据集

首先构建所有配置的数据集：

```bash
cd /mnt/aime/datasets/linxueyuan/RETuning/reproduce/Time-Series-Library
bash scripts/stock_classification/build_all_datasets.sh
```

或者构建单个配置的数据集：

```bash
python stock_classification/build_stock_classification_dataset.py \
    --raw_dir ../../data/raw \
    --out_root ./stock_classification/dataset \
    --mode 1 \
    --seq_len 20
```

### 2. 训练模型

运行所有模型的所有配置：

```bash
bash scripts/stock_classification/run_all.sh
```

或者运行单个模型：

```bash
bash scripts/stock_classification/TimesNet_stock.sh
```

或者运行单个配置：

```bash
python stock_classification/run_stock_classification.py \
    --is_training 1 \
    --model TimesNet \
    --mode 1 \
    --seq_len 20 \
    --root_path ./stock_classification/dataset \
    --batch_size 64 \
    --train_epochs 30 \
    --learning_rate 0.001 \
    --gpu 0
```

### 3. 查看结果

结果保存在以下目录：

- `checkpoints/StockCls_*/`: 模型检查点
- `results/StockCls_*/`: 评估指标
  - `metrics.txt`: 详细指标
  - `evaluation_results.csv`: 预测结果 CSV
  - `daily_summary.csv`: 每日准确率汇总a
åa- `test_results/StockCls_*/`: 测试结果

### 4. 汇总结果

```bash
python stock_classification/summarize_results.py \
    --results_dir ./results \
    --output ./results/summary.csv
```

## 参数说明

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称 | TimesNet |
| `--mode` | 数据划分模式 (1或2) | 1 |
| `--seq_len` | 输入序列长度 | 20 |
| `--batch_size` | 批大小 | 64 |
| `--train_epochs` | 训练轮数 | 30 |
| `--learning_rate` | 学习率 | 0.001 |
| `--patience` | 早停耐心值 | 5 |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--d_model` | 模型维度 | 64 |
| `--d_ff` | FFN维度 | 128 |
| `--e_layers` | 编码器层数 | 2 |
| `--n_heads` | 注意力头数 | 4 |
| `--dropout` | Dropout率 | 0.1 |

## 评估指标

- **Accuracy**: 整体准确率
- **F1 (macro)**: 宏平均F1分数
- **F1 (weighted)**: 加权平均F1分数
- **Recall (macro)**: 宏平均召回率
- **Per-class F1/Recall**: 各类别的F1和召回率

## 输出格式

### evaluation_results.csv

| 列名 | 说明 |
|------|------|
| date | 交易日期 |
| code | 股票代码 |
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 收盘价 |
| pct_change | 日涨跌幅 |
| label | 真实标签 (0/1/2) |
| label_name | 真实标签名 (down/hold/up) |
| predicted_label | 预测标签 |
| predicted_label_name | 预测标签名 |
| prob_down | down概率 |
| prob_hold | hold概率 |
| prob_up | up概率 |

## 注意事项

1. 确保原始数据文件位于 `../../data/raw/` 目录下
2. 数据文件需要包含 `date, code, open, high, low, close` 列
3. 由于类别不平衡，模型训练使用了类别权重
4. 建议先在小数据集上测试，再运行完整实验
