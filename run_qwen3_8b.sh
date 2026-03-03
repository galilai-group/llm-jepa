WANDB_FLAGS=""
# Uncomment to enable W&B logging:
# WANDB_FLAGS="--wandb --wandb_project llm-jepa"

run_regular() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}

  echo "Success Rate: regular ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=2 finetune.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=./fine-tuned --num_epochs=${epoch} --finetune_seed=${seed} --regular \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --lora --lora_rank=256 --batch_size=8 --grad_accum=8 \
    --eval_accuracy --max_new_tokens_eval=512 ${WANDB_FLAGS}
  python evaluate.py --model_name=./fine-tuned \
    --input_file=${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database | tee -a output.txt
}

run_jepa() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}

  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=2 finetune.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=./fine-tuned --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --lora --lora_rank=256 --batch_size=8 --grad_accum=8 \
    --eval_accuracy --max_new_tokens_eval=512 ${WANDB_FLAGS}
  python evaluate.py --model_name=./fine-tuned \
    --input_file=${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database | tee -a output.txt
}

model_name=Qwen/Qwen3-8B
learning_rate=4e-5
dataset=gsm8k
seed=82

run_regular ${model_name} ${learning_rate} 4 -3 1 ${seed} 1.0 ${dataset}
run_jepa ${model_name} ${learning_rate} 4 -3 1 ${seed} 1.0 ${dataset}
