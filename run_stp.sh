run_regular() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}
  model_folder=${9}

  echo "Success Rate: regular ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 stp.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} --regular \
    --model_name=${base_model_name} --learning_rate=${learning_rate}
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 | tee -a output.txt
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
  model_folder=${9}

  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 stp.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=random_span
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 | tee -a output.txt
}

# if [[ "$base_model_name" == google/gemma* ]]
# then
#   last_token=-2
# elif [[ "$base_model_name" == apple/OpenELM* ]]
# then
#   last_token=-4
# elif [[ "$base_model_name" == allenai/OLMo-2* ]]
# then
#   last_token=-1
# elif [[ "$base_model_name" == Qwen/Qwen* ]]
# then
#   last_token=-3
# elif [[ "$base_model_name" == deepseek-ai/DeepSeek* ]]
# then
#   last_token=-1
# else
#   last_token=-2
# fi

models=(meta-llama/Llama-3.2-1B-Instruct apple/OpenELM-1_1B-Instruct google/gemma-2-2b-it \
        microsoft/phi-1_5 allenai/OLMo-2-0425-1B-Instruct \
        deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B Qwen/Qwen3-1.7B)
non_it_models=(meta-llama/Llama-3.2-1B apple/OpenELM-1_1B google/gemma-2-2b \
               microsoft/phi-1_5 allenai/OLMo-2-0425-1B)
dataset=(synth turk gsm8k spider)

model_name=meta-llama/Llama-3.2-1B-Instruct
dataset=synth
lbd=0.02
predictors=0
for learning_rate in 2e-5
do
  for seed in 82 23 37 84 4
  do
    model_folder=ft-r-${learning_rate}-${seed}
    run_regular ${model_name} ${learning_rate} 4 -2 ${predictors} ${seed} ${lbd} ${dataset} ${model_folder}
    model_folder=ft-j-${learning_rate}-${lbd}-${predictors}-${seed}
    run_jepa ${model_name} ${learning_rate} 4 -2 ${predictors} ${seed} ${lbd} ${dataset} ${model_folder}
  done
done

