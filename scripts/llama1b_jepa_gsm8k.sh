#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)

source .venv/bin/activate
torchrun --nproc_per_node=$NGPUS --master_port=29507 finetune.py \
  --train_file datasets/gsm8k_train.jsonl \
  --output_dir=./fine-tuned-llama1b-jepa \
  --num_epochs=4 --finetune_seed=82 \
  --last_token=-2 --lbd=0.5 --predictors=1 \
  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=1e-5 \
  --batch_size=8 --grad_accum=8 \
  --eval_accuracy --eval_accuracy_steps=100 --wandb --max_eval_samples=50

CUDA_VISIBLE_DEVICES=$FIRST_GPU python evaluate.py \
  --model_name=./fine-tuned-llama1b-jepa \
  --input_file=datasets/gsm8k_test.jsonl \
  --output_file=eval_llama1b_jepa.jsonl \
  --original_model_name=meta-llama/Llama-3.2-1B-Instruct \
  --nosplit_data --device_map=cuda:0 | tee -a output.txt
