#!/bin/bash
# Build all stock classification datasets with different configurations
# Run this script first before training models

cd /mnt/aime/datasets/linxueyuan/RETuning/reproduce/Time-Series-Library

echo "=========================================="
echo "Building Stock Classification Datasets"
echo "=========================================="

# Sequence lengths to build
SEQ_LENS=(5 10 20 60)

# Modes: 1=all history, 2=2024 only
MODES=(1 2)

for MODE in "${MODES[@]}"; do
    for SEQ_LEN in "${SEQ_LENS[@]}"; do
        echo ""
        echo "Building dataset: mode=${MODE}, seq_len=${SEQ_LEN}"
        echo "-------------------------------------------"
        python stock_classification/build_stock_classification_dataset.py \
            --raw_dir ../../data/raw \
            --out_root ./stock_classification/dataset \
            --mode ${MODE} \
            --seq_len ${SEQ_LEN}
    done
done

echo ""
echo "=========================================="
echo "All datasets built successfully!"
echo "=========================================="
