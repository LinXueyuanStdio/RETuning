#!/bin/bash
# Run all stock classification experiments
# This script builds datasets and runs all models

cd /mnt/aime/datasets/linxueyuan/RETuning/reproduce/Time-Series-Library

echo "=========================================="
echo "Stock Classification Experiment Pipeline"
echo "=========================================="
echo ""

# Step 1: Build all datasets
echo "Step 1: Building all datasets..."
echo "-------------------------------------------"
bash scripts/stock_classification/build_all_datasets.sh

echo ""
echo "=========================================="
echo "Step 2: Training all models"
echo "=========================================="
echo ""

# Run each model
MODELS=("TimesNet" "PatchTST" "Informer" "DLinear" "Autoformer" "TimeMixer")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running ${MODEL} experiments..."
    echo "=========================================="
    bash scripts/stock_classification/${MODEL}_stock.sh
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - checkpoints/StockCls_*/"
echo "  - results/StockCls_*/"
echo "  - test_results/StockCls_*/"
