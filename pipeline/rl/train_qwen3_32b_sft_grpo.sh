# export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
#export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 检查集群环境
set -x
env | grep -E 'NVIDIA_VISIBLE_DEVICES|RANK|MASTER_|WORLD_SIZE' | grep -v PET
ifconfig
#export MASTER_ADDR=127.0.0.1
#export MASTER_PORT=8888
#export WORLD_SIZE=1
#export RAY_ADDRESS=http://127.0.0.1:8265

# 检查训练环境
cd /mnt/data/github/RETuning/verl

pip config set global.index-url https://pypi.org/simple
# pip install vllm==0.8.4
# pip install flash-attn --no-build-isolation
pip install ray sphinx weave xlin math-verify mathruler nvitop --upgrade
# pip install tensordict==0.6.2 codetiming==1.4.0 hydra-core==1.3.2 torchdata==0.9.0
pip install -r requirements.txt
pip install -e .  # For verl integration
pip install wandb IPython matplotlib
# pip install transformers liger-kernel --upgrade
# pip install liger-kernel --upgrade
pip uninstall deepspeed

# 设置训练参数
export WANDB_API_KEY=your_wandb_api_key
export WANDB_PROJECT=RETuning_rl
export WANDB_EXPERIMENT_NAME=Qwen3_32B_SFT_GRPO
export BASE_MODEL="/mnt/model/Qwen3-32B"
export SFT_OUTPUT_PATH="/mnt/data/runs/qwen3_32b_sft/v3-20250723-111822/checkpoint-60"
export RL_OUTPUT_PATH="/mnt/workspace/checkpoint"
export ROLLOUT_OUTPUT_PATH=/mnt/workspace/inference
export LOG_DIR=/mnt/workspace/log
mkdir -p $ROLLOUT_OUTPUT_PATH
mkdir -p $LOG_DIR

train_difficulty_middle_path=data/rl/Fin-2024-Jan2Nov-difficulty-middle.parquet
train_difficulty_hard_path=data/rl/Fin-2024-Jan2Nov-difficulty-hard.parquet
test_path=data/evaluation/Fin-2024-December.parquet

train_files="['$train_difficulty_middle_path']"
# train_files="['$train_difficulty_middle_path', '$train_difficulty_hard_path']"
test_files="['$test_path']"

max_prompt_length=$((1024 * 32))
max_response_length=$((1024 * 2))
sequence_parallel_size=4
model_parallel_size=4  #  smaller than world_size
n_gpus_per_node=8
nnodes=4
rollout_n=8
rollout_temperature=0.6
reward_path="recipe/trade/reward_score.py"
reward_func_name="reward_func"

# assert one h100 gpu has batch size of 1 and token length of 16k
# assert the total batch size is 256
# assert the total token length is 64k
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) / sequence_parallel_size ))
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) / sequence_parallel_size ))
micro_batch_size=$(( nnodes * n_gpus_per_node / sequence_parallel_size ))
max_num_batched_tokens=$(( max_prompt_length + max_response_length ))  # >= max_model_len (== prompt_len + response_len)

echo "max_prompt_length: ${max_prompt_length}, max_response_length: ${max_response_length}"
echo "actor_ppo_max_token_len: ${actor_ppo_max_token_len}, infer_ppo_max_token_len: ${infer_ppo_max_token_len}"
echo "micro_batch_size: ${micro_batch_size}, max_num_batched_tokens: ${max_num_batched_tokens}"
echo "squence_parallel_size: ${sequence_parallel_size}, model_parallel_size: ${model_parallel_size}"
echo "n_gpus_per_node: ${n_gpus_per_node}, nnodes: ${nnodes}"
echo "rollout_n: ${rollout_n}, rollout_temperature: ${rollout_temperature}"
echo "reward_path: ${reward_path}, reward_func_name: ${reward_func_name}"

address=$MASTER_ADDR:$MASTER_PORT
echo "IP Head: $address"


if [ $RANK == 0 ]; then

export JOBLOG=${LOG_DIR}/master.log
echo "STARTING HEAD at $address" &>> ${JOBLOG}
ray start --head --node-ip-address=$MASTER_ADDR --port=$MASTER_PORT &>> ${JOBLOG}
ray status &>> ${JOBLOG}
echo "Ray head started at $address" &>> ${JOBLOG}
echo "Head initialization complete." &>> ${JOBLOG}
sleep 80
echo "Starting training process..." &>> ${JOBLOG}
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.shuffle=false \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    actor_rollout_ref.model.path=${SFT_OUTPUT_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="all-linear" \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sequence_parallel_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    +actor_rollout_ref.actor.optim.strategy=adamw_bf16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=${rollout_temperature} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sequence_parallel_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=${reward_path} \
    custom_reward_function.name=${reward_func_name} \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.default_local_dir=${RL_OUTPUT_PATH} \
    trainer.rollout_data_dir=${ROLLOUT_OUTPUT_PATH} \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 2>&1 | tee ${JOBLOG}

else

export GLOO_SOCKET_IFNAME=br0
export JOBLOG=${LOG_DIR}/worker_rank${RANK}.log
echo "STARTING WORKER at $address" &>> ${JOBLOG}
ray start --address=$address --block >> ${JOBLOG}
ray status &>> ${JOBLOG}
echo "Ray worker started at $address" &>> ${JOBLOG}
echo "Worker initialization complete." &>> ${JOBLOG}

fi
