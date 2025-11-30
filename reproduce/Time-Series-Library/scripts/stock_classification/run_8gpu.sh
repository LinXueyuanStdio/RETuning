#!/bin/bash
# Simple parallel execution using background processes
# For 8 H100 GPUs - runs 48 experiments (6 models × 2 modes × 4 seq_lens)

set -e

cd /mnt/aime/datasets/linxueyuan/RETuning/reproduce/Time-Series-Library

NUM_GPUS=1
LOG_DIR="./logs/stock_classification"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Stock Classification - 8 GPU Parallel Run"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Step 1: Build all datasets first (sequentially)
echo "Step 1: Building all datasets..."
bash scripts/stock_classification/build_all_datasets.sh
echo "Datasets ready!"
echo ""

# Step 2: Create experiment list
EXPERIMENT_LIST="${LOG_DIR}/experiment_list.txt"
> ${EXPERIMENT_LIST}

MODELS=("TimesNet" "PatchTST" "Informer" "DLinear" "Autoformer" "TimeMixer")
MODES=(1 2)
SEQ_LENS=(5 10 20 60)

for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for SEQ_LEN in "${SEQ_LENS[@]}"; do
            echo "${MODEL} ${MODE} ${SEQ_LEN}" >> ${EXPERIMENT_LIST}
        done
    done
done

TOTAL=$(wc -l < ${EXPERIMENT_LIST})
echo "Step 2: Running ${TOTAL} experiments across ${NUM_GPUS} GPUs..."
echo ""

# Step 3: Run experiments in parallel
# Using a simple round-robin assignment with background jobs

run_one() {
    MODEL=$1
    MODE=$2
    SEQ_LEN=$3
    GPU_ID=$4

    EXP_NAME="StockCls_${MODEL}_mode${MODE}_sl${SEQ_LEN}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

    # Model-specific parameters
    EXTRA=""
    TOP_K=3
    case ${MODEL} in
        "TimesNet")
            # TimesNet requires seq_len > 2*top_k for FFT period detection
            # Adjust top_k based on seq_len
            if [ ${SEQ_LEN} -le 5 ]; then
                TOP_K=1
            elif [ ${SEQ_LEN} -le 10 ]; then
                TOP_K=2
            else
                TOP_K=3
            fi
            EXTRA="--top_k ${TOP_K} --num_kernels 6"
            ;;
        "PatchTST")
            # Adjust patch_len based on seq_len
            if [ ${SEQ_LEN} -le 5 ]; then
                PATCH_LEN=1
            elif [ ${SEQ_LEN} -le 10 ]; then
                PATCH_LEN=2
            else
                PATCH_LEN=4
            fi
            EXTRA="--patch_len ${PATCH_LEN}"
            ;;
        "Informer")   EXTRA="--factor 3" ;;
        "Autoformer") EXTRA="--factor 3" ;;
        "TimeMixer")  EXTRA="--down_sampling_layers 1 --down_sampling_window 2 --down_sampling_method avg" ;;
    esac

    source /root/anaconda3/bin/activate ts
    echo "[$(date +%H:%M:%S)] GPU ${GPU_ID} | Starting: ${EXP_NAME}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python stock_classification/run_stock_classification.py \
        --is_training 1 \
        --model ${MODEL} \
        --mode ${MODE} \
        --seq_len ${SEQ_LEN} \
        --root_path ./stock_classification/dataset \
        --model_id "stock_mode${MODE}_sl${SEQ_LEN}" \
        --d_model 64 \
        --d_ff 128 \
        --e_layers 2 \
        --n_heads 4 \
        --dropout 0.1 \
        --batch_size 64 \
        --train_epochs 30 \
        --patience 5 \
        --learning_rate 0.001 \
        --gpu 0 \
        --des "Exp" \
        ${EXTRA} \
        > "${LOG_FILE}" 2>&1

    STATUS=$?
    if [ ${STATUS} -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] GPU ${GPU_ID} | ✓ Completed: ${EXP_NAME}"
    else
        echo "[$(date +%H:%M:%S)] GPU ${GPU_ID} | ✗ FAILED: ${EXP_NAME} (check ${LOG_FILE})"
    fi
}

export -f run_one
export LOG_DIR

# Use parallel if available, otherwise use xargs with background jobs
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel..."
    cat ${EXPERIMENT_LIST} | parallel --colsep ' ' -j ${NUM_GPUS} \
        'run_one {1} {2} {3} $(( ({%} - 1) % 8 ))'
else
    echo "Using background jobs (install GNU Parallel for better scheduling)..."

    # Simple semaphore using a named pipe
    FIFO="${LOG_DIR}/gpu_fifo"
    mkfifo ${FIFO} 2>/dev/null || true
    exec 3<>${FIFO}
    rm -f ${FIFO}

    # Initialize semaphore with GPU IDs
    for ((i=0; i<NUM_GPUS; i++)); do
        echo $i >&3
    done

    # Read experiments and run in parallel
    while read MODEL MODE SEQ_LEN; do
        # Get a GPU from the semaphore
        read -u 3 GPU_ID

        (
            run_one ${MODEL} ${MODE} ${SEQ_LEN} ${GPU_ID}
            # Return GPU to the semaphore
            echo ${GPU_ID} >&3
        ) &

    done < ${EXPERIMENT_LIST}

    # Wait for all jobs
    wait
    exec 3>&-
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=========================================="

# Generate summary
echo ""
echo "Generating results summary..."
python stock_classification/summarize_results.py \
    --results_dir ./results \
    --output ./results/summary.csv 2>/dev/null || echo "Summary script not run (may need results first)"

echo ""
echo "Done! Check:"
echo "  - Logs: ${LOG_DIR}/"
echo "  - Results: ./results/"
echo "  - Summary: ./results/summary.csv"
