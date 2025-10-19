pip install -e '.[deepspeed]' --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install huggingface-hub==0.25.2  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade opencv-python==4.7.0.72 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

export BASE_MODEL="/mnt/model/DeepSeek-R1-14B"
export WORK_DIR="/mnt/data/runs/DeepSeek-R1-14B-SFT"
export DATA_PATH="/mnt/data/github/RETuning/data/sft"

export SFT_LORA_OUTPUT_DIR="${WORK_DIR}/sft_lora_output"
export MERGE_OUTPUT_DIR="${WORK_DIR}/sft_merge_output"
export MAX_LENGTH=65536
export DEEPSPEED_CONFIG="deepspeed_zero3"



# 运行xtuner训练

echo "=================================="
echo "============SFT  TRAIN============"
echo "=================================="

NPROC_PER_NODE=8 xtuner train ./xtuner_sft_lora.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --work-dir ${WORK_DIR}


LAST_CHECKPOINT=$(cat ${WORK_DIR}/last_checkpoint)

xtuner convert pth_to_hf ./xtuner_sft_lora.py ${LAST_CHECKPOINT} ${SFT_LORA_OUTPUT_DIR}

xtuner convert merge ${BASE_MODEL} ${SFT_LORA_OUTPUT_DIR} ${MERGE_OUTPUT_DIR} --max-shard-size 8GB --safe-serialization