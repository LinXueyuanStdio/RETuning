#!/bin/bash
# Quick test script to verify the stock classification pipeline
# This builds a small dataset and runs a quick training

cd /mnt/aime/datasets/linxueyuan/RETuning/reproduce/Time-Series-Library

echo "=========================================="
echo "Testing Stock Classification Pipeline"
echo "=========================================="

# Step 1: Build one dataset configuration
echo ""
echo "Step 1: Building test dataset (mode=2, seq_len=10)..."
python stock_classification/build_stock_classification_dataset.py \
    --raw_dir ../../data/raw \
    --out_root ./stock_classification/dataset \
    --mode 2 \
    --seq_len 10 \
    --overwrite

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset building failed!"
    exit 1
fi

echo ""
echo "Step 2: Running quick training test with TimesNet..."
python stock_classification/run_stock_classification.py \
    --is_training 1 \
    --model TimesNet \
    --mode 2 \
    --seq_len 10 \
    --root_path ./stock_classification/dataset \
    --model_id "test" \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 1 \
    --n_heads 2 \
    --top_k 2 \
    --num_kernels 4 \
    --dropout 0.1 \
    --batch_size 32 \
    --train_epochs 2 \
    --patience 2 \
    --learning_rate 0.001 \
    --gpu 0 \
    --des "Test"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline test completed successfully!"
echo "=========================================="
