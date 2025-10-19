
SFT_DATA_PATH="/mnt/data/github/RETuning/data/sft/train_10000.jsonl /mnt/data/github/RETuning/data/sft/cold_start.jsonl"
SFT_BASE_MODEL="/mnt/model/Qwen3-8B"
SFT_OUTPUT_DIR="/mnt/data/runs/qwen3_8b_sft"

# ===============================
# 📦 安装依赖
# ===============================
echo "==========================="
echo "Installing dependencies..."
echo "==========================="

sudo apt update && sudo apt install -y libopenmpi-dev
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install liger-kernel mpi4py --index-url https://pypi.tuna.tsinghua.edu.cn/simple

cd ms-swift
pip install -e .

# ===============================
# 🧪 SFT 阶段
# ===============================
echo "==========================="
echo "Starting SFT training..."
echo "==========================="

export NCCL_TIMEOUT_SEC=1200  #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
swift sft \
    --model ${SFT_BASE_MODEL} \
    --output_dir ${SFT_OUTPUT_DIR} \
    --train_type full \
    --model_type qwen3 \
    --dataset ${SFT_DATA_PATH} \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --save_steps 5000 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 40768 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --packing false \
    --save_only_model true \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn \
    --loss_scale ignore_empty_think \
    --sequence_parallel_size 1 \
    --split_dataset_ratio 0 \



