export VLLM_WORKER_MULTIPROC_METHOD=spawn
export THREAD_POOL_SIZE=128

cd /mnt/data/github/RETuning

date_str=$(date "+%Y-%m-%d")
time_str=$(date "+%H-%M-%S")

model="random"
data_path="data/evaluation/Fin-2024-December.parquet"


# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" PYTHONUNBUFFERED=1
for seed in 8 9 10 11 12 13 14 15 16 17 18 19 20 \
             21 22 23 24 25 26 27 28 29 30 \
             31 32 33 34 35; do
    echo "seed: $seed"
    save_dir="output/evaluation/random${seed}"
    log_file="${save_dir}/evaluation.log"
    mkdir -p ${save_dir}
    echo "logging to $log_file"
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
      --tensor_parallel_size 4 \
      --n 1,2,4,8,16,32 \
      --majority_voting \
      --seed ${seed} \
      --debug $@ 2>&1 | tee ${log_file}

done
