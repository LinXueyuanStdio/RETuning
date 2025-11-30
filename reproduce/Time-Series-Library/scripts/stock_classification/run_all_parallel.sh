#!/bin/bash
# Parallel execution script for stock classification experiments
# Utilizes 8 H100 GPUs to run all experiments efficiently
#
# Total experiments: 6 models × 2 modes × 4 seq_lens = 48 experiments
# Strategy: Distribute experiments across 8 GPUs with job queue

cd /mnt/aime/datasets/linxueyuan/RETuning/reproduce/Time-Series-Library

# Configuration
NUM_GPUS=8
LOG_DIR="./logs/stock_classification"
RESULTS_DIR="./results"

# Create directories
mkdir -p ${LOG_DIR}
mkdir -p ${RESULTS_DIR}

# Models and configurations
MODELS=("TimesNet" "PatchTST" "Informer" "DLinear" "Autoformer" "TimeMixer")
MODES=(1 2)
SEQ_LENS=(5 10 20 60)

# Model-specific parameters
declare -A MODEL_PARAMS
MODEL_PARAMS["TimesNet"]="--top_k 3 --num_kernels 6"
MODEL_PARAMS["PatchTST"]="--patch_len 4"
MODEL_PARAMS["Informer"]="--factor 3"
MODEL_PARAMS["DLinear"]=""
MODEL_PARAMS["Autoformer"]="--factor 3"
MODEL_PARAMS["TimeMixer"]="--down_sampling_layers 0 --down_sampling_window 1"

# Generate all experiment combinations
EXPERIMENTS=()
for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for SEQ_LEN in "${SEQ_LENS[@]}"; do
            EXPERIMENTS+=("${MODEL}|${MODE}|${SEQ_LEN}")
        done
    done
done

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "=========================================="
echo "Stock Classification Parallel Training"
echo "=========================================="
echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo "Available GPUs: ${NUM_GPUS}"
echo "Models: ${MODELS[*]}"
echo "Modes: ${MODES[*]}"
echo "Sequence lengths: ${SEQ_LENS[*]}"
echo "=========================================="

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local MODEL=$2
    local MODE=$3
    local SEQ_LEN=$4
    local EXTRA_PARAMS="${MODEL_PARAMS[$MODEL]}"

    local EXP_NAME="StockCls_${MODEL}_mode${MODE}_sl${SEQ_LEN}"
    local LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

    echo "[GPU ${GPU_ID}] Starting: ${EXP_NAME}"

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
        ${EXTRA_PARAMS} \
        > "${LOG_FILE}" 2>&1

    local STATUS=$?
    if [ ${STATUS} -eq 0 ]; then
        echo "[GPU ${GPU_ID}] Completed: ${EXP_NAME} ✓"
    else
        echo "[GPU ${GPU_ID}] FAILED: ${EXP_NAME} ✗ (see ${LOG_FILE})"
    fi

    return ${STATUS}
}

export -f run_experiment
export LOG_DIR
export -A MODEL_PARAMS

# Method 1: Using GNU Parallel (recommended if available)
if command -v parallel &> /dev/null; then
    echo ""
    echo "Using GNU Parallel for job scheduling..."
    echo ""

    # Create a job file
    JOB_FILE="${LOG_DIR}/jobs.txt"
    > ${JOB_FILE}

    for EXP in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r MODEL MODE SEQ_LEN <<< "${EXP}"
        EXTRA_PARAMS="${MODEL_PARAMS[$MODEL]}"
        echo "${MODEL} ${MODE} ${SEQ_LEN} \"${EXTRA_PARAMS}\"" >> ${JOB_FILE}
    done

    # Run with GNU Parallel, distributing across GPUs
    cat ${JOB_FILE} | parallel --colsep ' ' -j ${NUM_GPUS} --joblog "${LOG_DIR}/parallel_joblog.txt" '
        GPU_ID=$(( ({%} - 1) % 8 ))
        MODEL={1}
        MODE={2}
        SEQ_LEN={3}
        EXTRA_PARAMS={4}

        EXP_NAME="StockCls_${MODEL}_mode${MODE}_sl${SEQ_LEN}"
        LOG_FILE="'"${LOG_DIR}"'/${EXP_NAME}.log"

        echo "[GPU ${GPU_ID}] Starting: ${EXP_NAME}"

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
            ${EXTRA_PARAMS} \
            > "${LOG_FILE}" 2>&1

        if [ $? -eq 0 ]; then
            echo "[GPU ${GPU_ID}] Completed: ${EXP_NAME} ✓"
        else
            echo "[GPU ${GPU_ID}] FAILED: ${EXP_NAME} ✗"
        fi
    '

else
    # Method 2: Manual job queue with background processes
    echo ""
    echo "Using manual job queue (GNU Parallel not found)..."
    echo ""

    # Array to track running jobs per GPU
    declare -a GPU_PIDS
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_PIDS[$i]=""
    done

    # Function to wait for a free GPU
    wait_for_gpu() {
        while true; do
            for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
                pid=${GPU_PIDS[$gpu]}
                if [ -z "$pid" ] || ! kill -0 $pid 2>/dev/null; then
                    echo $gpu
                    return
                fi
            done
            sleep 5
        done
    }

    # Launch experiments
    for EXP in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r MODEL MODE SEQ_LEN <<< "${EXP}"

        # Wait for a free GPU
        GPU_ID=$(wait_for_gpu)

        # Launch experiment in background
        (
            EXTRA_PARAMS="${MODEL_PARAMS[$MODEL]}"
            EXP_NAME="StockCls_${MODEL}_mode${MODE}_sl${SEQ_LEN}"
            LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

            echo "[GPU ${GPU_ID}] Starting: ${EXP_NAME}"

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
                ${EXTRA_PARAMS} \
                > "${LOG_FILE}" 2>&1

            if [ $? -eq 0 ]; then
                echo "[GPU ${GPU_ID}] Completed: ${EXP_NAME} ✓"
            else
                echo "[GPU ${GPU_ID}] FAILED: ${EXP_NAME} ✗"
            fi
        ) &

        GPU_PIDS[$GPU_ID]=$!
        sleep 2  # Small delay to prevent race conditions
    done

    # Wait for all remaining jobs
    echo ""
    echo "Waiting for remaining jobs to complete..."
    wait
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Logs: ${LOG_DIR}/"
echo "Results: ${RESULTS_DIR}/"
echo ""

# Generate summary
echo "Generating results summary..."
python stock_classification/summarize_results.py \
    --results_dir ${RESULTS_DIR} \
    --output ${RESULTS_DIR}/summary.csv

echo ""
echo "Summary saved to ${RESULTS_DIR}/summary.csv"
