#!/bin/bash
# Informer Stock Classification Training Script
# Covers all combinations of mode and seq_len

cd /mnt/aime/datasets/linxueyuan/RETuning/reproduce/Time-Series-Library

export CUDA_VISIBLE_DEVICES=0

MODEL="Informer"
GPU=0

# Modes: 1=all history, 2=2024 only
MODES=(1 2)

# Sequence lengths
SEQ_LENS=(5 10 20 60)

for MODE in "${MODES[@]}"; do
    for SEQ_LEN in "${SEQ_LENS[@]}"; do
        echo "=========================================="
        echo "Training ${MODEL}: mode=${MODE}, seq_len=${SEQ_LEN}"
        echo "=========================================="

        python stock_classification/run_stock_classification.py \
            --is_training 1 \
            --model ${MODEL} \
            --mode ${MODE} \
            --seq_len ${SEQ_LEN} \
            --root_path ./stock_classification/dataset \
            --model_id "stock_mode${MODE}_sl${SEQ_LEN}" \
            --d_model 64 \
            --d_ff 128 \
            --e_layers 2 \
            --d_layers 1 \
            --n_heads 4 \
            --factor 3 \
            --dropout 0.1 \
            --batch_size 64 \
            --train_epochs 30 \
            --patience 5 \
            --learning_rate 0.001 \
            --gpu ${GPU} \
            --des "Exp"

        echo ""
    done
done

echo "=========================================="
echo "All ${MODEL} experiments completed!"
echo "=========================================="
