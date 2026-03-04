#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)

source .venv/bin/activate
#torchrun --nproc_per_node=$NGPUS --master_port=29508 finetune.py \
#  --output_dir=./fine-tuned-llama1b-jepa-synth \
#  --num_epochs=4 --finetune_seed=82 \
#  --last_token=-2 --lbd=1 --predictors=1 \
#  --model_name=meta-llama/Llama-3.2-1B-Instruct --learning_rate=2e-5 \
#  --batch_size=8 --grad_accum=8 \
#  --eval_accuracy --wandb --max_eval_samples=50

CUDA_VISIBLE_DEVICES=$FIRST_GPU python evaluate.py \
  --model_name=./fine-tuned-llama1b-jepa-synth \
  --input_file=datasets/synth_test.jsonl \
  --output_file=eval_llama1b_jepa_synth.jsonl \
  --original_model_name=meta-llama/Llama-3.2-1B-Instruct \
  --nosplit_data --split_tune_untune --device_map=cuda:0 | tee -a output.txt
