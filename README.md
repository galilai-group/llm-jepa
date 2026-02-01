# LLM-JEPA

## Set Up

See `setup.sh`.

**NOTE**: Do NOT run `setup.sh` directly. Read the file, choose the configurtion for your envirnoment, and execute the relevant commands manually.

## Semantic Tube Prediction{#stp}

The fine-tuning script is in `stp.py`. A convenient driver script, `run_stp.sh`, provides `run_regular()` for standard fine-tuning, and `run_stp_jepa()` for Semantic Tupe Prediction fine-tuning.

General flags:

*   `--linear=random_span` for Semantic Tube Prediction.
*   `--linear_predictor` for training a linear predictor.

Ablation study flags:

*   `--linear=e2e` for Two View in ablation study.
*   `--random_span_mask` and `--random_span_mask_recover` for Mask in ablation study.
*   `--linear=curvature` for Curvature in ablation study.

Other flags are documented in `stp.py`.

`run_stp_jepa()` will ignore `predictors`.

## LLM-JEPA Fine-tuning

The fine-tuning script is in `finetune.py`. A convenient driver script, `run.sh`, provides `run_regular()` for standard fine-tuning, and `run_jepa()` for LLM-JEPA fine-tuning.

For all experiments, we fix number of epochs to 4. The `last_token` setting depends on the model family; see the commented lines in `run.sh` for how to set it. Each configuration is run with 5 random seeds. We report mean accuracy and standard deviation.

The original implementation required two additional forward passes to encode `Text` and `Code` separately. The latest version combines them into a single forward pass using a 4D additive attention mask. Enable this feature with `--additive_mask`. **NOTE**: `--additive_mask` may not work if the tokenizer applies left-padding.

## Large models

Similarly, we provide `finetune8bh200.py` and `run8bh200.sh` for training modesl up to 8B parameters on NVIDIA H200 GPUs.

## LLM-JEPA with LoRA

Use `--lora` and `--lora_rank <N>` to enable LoRA fine-tuning for LLM-JEPA.

## Pretraining

Use `--pretrain` to start from randomly initialized weights.

For pretraining on the `paraphrase` dataset, pass `--plain --trainall` to disable the OpenAI message format, train next-token prediction, and jointly minimize distances between paraphrase variants.

After pretaining, fine-tune with `--plain` on `rotten_tomatoes` and `yelp`. For evaluation, run with `--plain --statswith` to bypass the OpenAI message format and score only the first token(the model isn't instruction-tuned, so it may not emit a clean stop).

## Ablation of JEPA-loss

We provide several options for ablating JEPA-loss in `finetune.py`:

*  L2 norm: pass `--jepa_l2`
*  Mean squred error: pass `--jepa_mse`
*  Prepend `[PRED]` token to `Text`: pass `--front_pred`
*  Let `Code` predict `Text`: pass `--reverse_pred`
*  Use InfoNCE loss, pass `--infonce`

## FLOPs

To track FLOPs per step, pass `--track_flop` to `finetune.py`. This prints the FLOPs for the first 10 steps. The total FLOPs can be estimated as `PER_STEP_FLOPS * NUMBER_OF_STEPS`. When `--jepa_ratio` is enabled (see [Random JEPA-loss Dropout](#random-jepa-loss-dropout) below), FLOPs may vary across steps; in this case, use the _average_ FLOPs per step instead.

For fair comparisons, we provide `--same_flop`, which computes the number of training steps required to match the total FLOPs of standard fine-tuning, taking into account `--additive_mask` and/or `--jepa_ratio`. Checkpoints are saved at those steps and can be used for evaluatioin. 

*  If `--additive_mask` is enabled, the same number of steps requires `2X` the compute.
*  If `--jepa_ratio` is set to `1 - alpha`, the same number of steps use `(2 - alpha)X` the compute.

## Random JEPA-loss Dropout

The fine-tuning script `finetune.py` supports `--jepa_ratio` to implement **random JEPA-loss dropout**. The idea is that randomly dorpping some JEPA-loss has little impact on performance, but can substaintially reduce compute cost.

When dropout is active, the extra forward pass for `Enc(Text)` and `Enc(Code)` is skipped. If the dropout rate `LD = alpha`, then correspondingly `--jepa_ratio` should be set to `1 - alpha`. On average, one training step costs `(2 - alpha)X` the compute of standard fine-tuning.

Empirical results show that LLM-JEPA can tolerate aggressive dropout rate (e.g., `LD = 0.75`), requiring `1.25X` the compute while maintaining fine-tuning performance.

## Datasets

Most datasets include `_train.jsonl` and `_test.jsonl` files for fine-tuning and evaluation, repsectively. The originals come from prior publications; we preprocessed them and include the results here for convenience.

*  `synth` and `turk`, from https://arxiv.org/abs/1608.03000
*  `gsm8k`, from https://arxiv.org/abs/2110.14168
*  `spider`, from https://arxiv.org/abs/1809.08887. You aslo need to unzip `spider_data.zip` which contains `sqlite` databases to execute and evaluate the generated queries.
*  `paraphrase`, from HuggingFace `cestwc/paraphrase` dataset. Only have `train` split, for pre-training only.
*  `rotten_tomatoes`, from HuggingFace `cornell-movie-review-data/rotten_tomatoes` dataset. Used for fine-tuning and evaluating models pretrained by `paraphrase` dataset.
*  `yelp`, from HuggingFace `Yelp/yelp_review_full` dataset. Used for fine-tuning and evaluating models pretrained by `paraphrase` dataset.
*  `nq_open`, from https://arxiv.org/abs/1906.00300.
*  `hellaswag`, from HuggingFace `hellaswag` dataset.
