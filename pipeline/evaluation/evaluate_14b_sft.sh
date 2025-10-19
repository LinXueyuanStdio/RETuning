export VLLM_WORKER_MULTIPROC_METHOD=spawn
export THREAD_POOL_SIZE=128

cd /mnt/data/github/RETuning

date_str=$(date "+%Y-%m-%d")
time_str=$(date "+%H-%M-%S")

model="/mnt/data/runs/DeepSeek-R1-14B-SFT/sft_merge_output"
data_path="data/evaluation/Fin-2024-December.parquet"
save_dir="output/evaluation/DSR1_14B_sft/${date_str}_${time_str}"
log_file="${save_dir}/evaluation.log"
mkdir -p ${save_dir}
echo "logging to $log_file"

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" PYTHONUNBUFFERED=1
python pipeline/evaluation/trade.py \
  --data_path ${data_path} \
  --save_dir ${save_dir} \
  --model ${model} \
  --engine vllm \
  --max_tokens $((4 * 1024)) \
  --max_model_len $((64 * 1024)) \
  --batch_size 128 \
  --temperature 0.6 \
  --top_p 1.0 \
  --tensor_parallel_size 8 \
  --n 1,2,4,8,16,32 \
  --debug $@ 2>&1 | tee ${log_file}
