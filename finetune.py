"""LLM-JEPA.
"""

import copy
import math
import os
# import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import json
from datasets import load_dataset
import shutil
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse

os.environ["WANDB_PROJECT"] = "arc-llm-jepa"

def get_messages(model_name, messages):
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        return full_messages
    else:
        return messages


def get_user_messages(model_name, messages):
    return copy.deepcopy(messages)[1:2]


# gsm8k_pattern = re.compile(r"\n#### (.+)$")


def get_assistant_messages(model_name, dataset, messages):
    # if dataset.startswith("gsm8k"):
    #     messages = copy.deepcopy(messages)
    #     gt_match = re.search(gsm8k_pattern, messages[2]["content"])
    #     gt_answer = None if not gt_match else gt_match.group(1)
    #     if gt_answer:
    #         messages[2]["content"] = messages[2]["content"].replace(gt_answer, "")

    if "google/gemma" in model_name or "Qwen" in model_name:
        assistant_messages = copy.deepcopy(messages)[2:3]
        assistant_messages[0]["role"] = "user"
        return assistant_messages
    else:
        return messages[2:3]


def load_and_prepare_dataset(data_file, tokenizer, model_name,
                             max_length=2048, debug=0, predictors=0, regular=False, train_all=False,
                             plain=False, front_pred=False, reverse_pred=False):
    """Load JSONL dataset and format for training with proper label masking"""
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_file)['train']
    if  torch.cuda.current_device() == 0:
        print(f"Loaded {len(dataset)} examples from {data_file}")

    def format_full_conversation(messages):
        full_messages = get_messages(model_name, messages)
        if plain:
            if train_all:
                return full_messages, messages[1]["content"] + "<|eot_id|>"
            return full_messages, messages[1]["content"] + "<|perception|>" + messages[2]["content"] + "<|eot_id|>"
        return full_messages, tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def format_prompt_for_generation(messages, full_messages=None):
        if plain:
            if train_all:
                return messages[1]["content"]
            return messages[1]["content"] + "<|perception|>"
        if full_messages is None:
            full_messages = get_messages(model_name, messages)
        return tokenizer.apply_chat_template(
            full_messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize_chat_text(text, truncation, padding=False):
        return tokenizer(
            text,
            truncation=truncation,
            max_length=max_length if truncation else None,
            padding="max_length" if padding else False,
            add_special_tokens=False,
            return_tensors=None,
        )

    def compute_supervised_start(messages, input_length, full_messages=None):
        prompt_text = format_prompt_for_generation(messages, full_messages=full_messages)
        prompt_tokens = tokenize_chat_text(prompt_text, truncation=False)["input_ids"]
        return min(len(prompt_tokens), input_length)

    def is_within_budget(example):
        full_messages, formatted_chat = format_full_conversation(example["messages"])
        prompt_len = len(tokenize_chat_text(
            format_prompt_for_generation(example["messages"], full_messages=full_messages),
            truncation=False,
        )["input_ids"])
        full_len = len(tokenize_chat_text(formatted_chat, truncation=False)["input_ids"])
        # Keep only samples that fit fully and still leave room for assistant labels.
        return full_len <= max_length and prompt_len < max_length

    original_count = len(dataset)
    dataset = dataset.filter(
        is_within_budget,
        num_proc=max(1, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 4,
    )
    if torch.cuda.current_device() == 0:
        dropped = original_count - len(dataset)
        if dropped:
            print(
                f"Dropped {dropped}/{original_count} samples exceeding max_length={max_length} "
                f"(full sequence > {max_length} tokens, or prompt alone fills the budget so no assistant labels remain). "
                f"{len(dataset)} samples remain."
            )
    
    def tokenize_conversations(examples):
        """Tokenize conversations and mask input tokens properly"""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        user_input_ids_list = []
        user_labels_list = []
        user_attention_mask_list = []
        assistant_input_ids_list = []
        assistant_labels_list = []
        assistant_attention_mask_list = []

        for msg_idx, messages in enumerate(examples['messages']):
            # Apply chat template if available, otherwise format manually
            full_messages, formatted_chat = format_full_conversation(messages)
            
            # Tokenize the formatted conversation (no padding here; the collator pads each batch)
            tokenized = tokenize_chat_text(
                formatted_chat,
                truncation=True,
                padding=False,
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Create labels with proper masking
            if train_all:
                labels = create_labels_for_all(input_ids, attention_mask)
            else:
                labels = create_masked_labels(messages, input_ids, attention_mask, full_messages=full_messages)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

            if data_file.startswith("hellaswag"):
                user_messages = examples["text"][msg_idx]
                if debug == 8:
                    print(json.dumps(messages, indent=2))
                    print(json.dumps(user_messages, indent=2))
            else:
                if reverse_pred:
                    user_messages = get_assistant_messages(model_name, data_file, messages)
                else:
                    user_messages = get_user_messages(model_name, messages)
            to_add = predictors
            while to_add > 0:
                if front_pred:
                    user_messages[0]["content"] = f"<|predictor_{to_add}|>" + user_messages[0]["content"]
                else:
                    user_messages[0]["content"] += f"<|predictor_{to_add}|>"
                to_add -= 1
            if plain:
                formatted_chat_user = user_messages[0]["content"]
            else:
                formatted_chat_user = tokenizer.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            tokenized_user = tokenizer(
                formatted_chat_user,
                truncation=True,
                max_length=max_length,
                padding=False,  # Collator pads per batch
                add_special_tokens=False,
                return_tensors=None
            )
            user_input_ids_list.append(tokenized_user["input_ids"])
            user_labels_list.append([-100] * len(tokenized_user["input_ids"]))
            user_attention_mask_list.append(tokenized_user["attention_mask"])

            if data_file.startswith("hellaswag"):
                assistant_messages = examples["code"][msg_idx]
                if debug == 8:
                    print(json.dumps(assistant_messages, indent=2))
                    exit(0)
            else:
                if reverse_pred:
                    assistant_messages = get_user_messages(model_name, messages)
                else:
                    assistant_messages = get_assistant_messages(model_name, data_file, messages)
            if plain:
                formatted_chat_assistant = assistant_messages[0]["content"]
            else:
                formatted_chat_assistant = tokenizer.apply_chat_template(
                    assistant_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            tokenized_assistant = tokenizer(
                formatted_chat_assistant,
                truncation=True,
                max_length=max_length,
                padding=False,  # Collator pads per batch
                add_special_tokens=False,
                return_tensors=None
            )
            assistant_input_ids_list.append(tokenized_assistant["input_ids"])
            assistant_labels_list.append([-100] * len(tokenized_assistant["input_ids"]))
            assistant_attention_mask_list.append(tokenized_assistant["attention_mask"])

            if debug == 3 and torch.cuda.current_device() == 0:
                print(messages)
                print(input_ids_list)
                print(tokenizer.decode(input_ids_list[0]))
                print(labels_list)
                print(tokenizer.decode([item for item in labels_list[0] if item != -100]))
                print(attention_mask_list)
                print("user Token IDs:", tokenized_user["input_ids"])
                print("user Decoded:", tokenizer.decode(tokenized_user["input_ids"]))
                print("assistant Token IDs:", tokenized_assistant["input_ids"])
                print("assistant Decoded:", tokenizer.decode(tokenized_assistant["input_ids"]))
        
            if debug == 3:
                exit(0)

        if regular:
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
                "length": [len(ids) for ids in input_ids_list],
            }
        else:
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
                "input_ids_user": user_input_ids_list,
                "labels_user": user_labels_list,
                "attention_mask_user": user_attention_mask_list,
                "input_ids_assistant": assistant_input_ids_list,
                "labels_assistant": assistant_labels_list,
                "attention_mask_assistant": assistant_attention_mask_list,
                "length": [len(ids) for ids in input_ids_list],
            }
    
    # def format_messages_manually(messages):
    #     """Manual formatting when chat template is not available"""
    #     formatted_parts = []
        
    #     for msg in messages:
    #         role = msg['role']
    #         content = msg['content']
            
    #         if role == 'system':
    #             formatted_parts.append(f"<|system|>\n{content}")
    #         elif role == 'user':
    #             formatted_parts.append(f"<|user|>\n{content}")
    #         elif role == 'assistant':
    #             formatted_parts.append(f"<|assistant|>\n{content}")
        
    #     return "\n\n".join(formatted_parts) + "<|end|>"
    
    def create_labels_for_all(input_ids, attention_mask):
        """
        Create labels for all tokens except padding (mask those with -100).
        """
        labels = []
        for i, mask in enumerate(attention_mask):
            if mask == 0:  # Padding token
                labels.append(-100)
            else:
                labels.append(input_ids[i])
        return labels

    def create_masked_labels(messages, input_ids, attention_mask, full_messages=None):
        """Create labels with input tokens masked (-100)"""
        labels = [-100] * len(input_ids)

        input_length = sum(attention_mask)
        assistant_start = compute_supervised_start(messages, input_length, full_messages=full_messages)

        for j in range(assistant_start, len(input_ids)):
            if attention_mask[j] == 1:
                labels[j] = input_ids[j]

        if debug == 4 and torch.cuda.current_device() == 0:
            print(f"assistant_start={assistant_start}, input_length={input_length}")
            print(tokenizer.decode(input_ids[:input_length]))
            print(tokenizer.decode([item for item in labels if item != -100]))
            exit(0)

        return labels
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_conversations,
        batched=True,
        num_proc=max(1, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 4,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


# def use_llama_3_2_chat_template(tokenizer):
#     llama_3_2_chat_template = """{{- bos_token }}
# {%- if custom_tools is defined %}
#     {%- set tools = custom_tools %}
# {%- endif %}
# {%- if not tools_in_user_message is defined %}
#     {%- set tools_in_user_message = true %}
# {%- endif %}
# {%- if not date_string is defined %}
#     {%- if strftime_now is defined %}
#         {%- set date_string = strftime_now("%d %b %Y") %}
#     {%- else %}
#         {%- set date_string = "26 Jul 2024" %}
#     {%- endif %}
# {%- endif %}
# {%- if not tools is defined %}
#     {%- set tools = none %}
# {%- endif %}

# {#- This block extracts the system message, so we can slot it into the right place. #}
# {%- if messages[0]['role'] == 'system' %}
#     {%- set system_message = messages[0]['content']|trim %}
#     {%- set messages = messages[1:] %}
# {%- else %}
#     {%- set system_message = "" %}
# {%- endif %}

# {#- System message #}
# {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
# {%- if tools is not none %}
#     {{- "Environment: ipython\n" }}
# {%- endif %}
# {{- "Cutting Knowledge Date: December 2023\n" }}
# {{- "Today Date: " + date_string + "\n\n" }}
# {%- if tools is not none and not tools_in_user_message %}
#     {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
#     {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
#     {{- "Do not use variables.\n\n" }}
#     {%- for t in tools %}
#         {{- t | tojson(indent=4) }}
#         {{- "\n\n" }}
#     {%- endfor %}
# {%- endif %}
# {{- system_message }}
# {{- "<|eot_id|>" }}

# {#- Custom tools are passed in a user message with some extra guidance #}
# {%- if tools_in_user_message and not tools is none %}
#     {#- Extract the first user message so we can plug it in here #}
#     {%- if messages | length != 0 %}
#         {%- set first_user_message = messages[0]['content']|trim %}
#         {%- set messages = messages[1:] %}
#     {%- else %}
#         {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
# {%- endif %}
#     {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
#     {{- "Given the following functions, please respond with a JSON for a function call " }}
#     {{- "with its proper arguments that best answers the given prompt.\n\n" }}
#     {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
#     {{- "Do not use variables.\n\n" }}
#     {%- for t in tools %}
#         {{- t | tojson(indent=4) }}
#         {{- "\n\n" }}
#     {%- endfor %}
#     {{- first_user_message + "<|eot_id|>"}}
# {%- endif %}

# {%- for message in messages %}
#     {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
#         {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
#     {%- elif 'tool_calls' in message %}
#         {%- if not message.tool_calls|length == 1 %}
#             {{- raise_exception("This model only supports single tool-calls at once!") }}
#         {%- endif %}
#         {%- set tool_call = message.tool_calls[0].function %}
#         {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
#         {{- '{"name": "' + tool_call.name + '", ' }}
#         {{- '"parameters": ' }}
#         {{- tool_call.arguments | tojson }}
#         {{- "}" }}
#         {{- "<|eot_id|>" }}
#     {%- elif message.role == "tool" or message.role == "ipython" %}
#         {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
#         {%- if message.content is mapping or message.content is iterable %}
#             {{- message.content | tojson }}
#         {%- else %}
#             {{- message.content }}
#         {%- endif %}
#         {{- "<|eot_id|>" }}
#     {%- endif %}
# {%- endfor %}
# {%- if add_generation_prompt %}
#     {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
# {%- endif %}
# """
#     if tokenizer.chat_template != llama_3_2_chat_template:
#         tokenizer.chat_template = llama_3_2_chat_template


def setup_model_and_tokenizer(model_name, use_lora=True, lora_rank=16, pretrain=False, debug=0, seed=None):
    """Setup model and tokenizer with optional LoRA"""
    
    # Load tokenizer
    if "apple/OpenELM" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert tokenizer.chat_template is not None, f"{model_name} does not have chat template."
    
    # use_llama_3_2_chat_template(tokenizer)
    
    # Add special tokens if not present
    if "microsoft/phi" in model_name:
        tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        if torch.cuda.current_device() == 0:
            print("Added <|startoftext|> token")

    special_tokens = ["<|predictor_1|>", "<|predictor_2|>", "<|predictor_3|>", "<|predictor_4|>", "<|predictor_5|>",
                      "<|predictor_6|>", "<|predictor_7|>", "<|predictor_8|>", "<|predictor_9|>", "<|predictor_10|>",
                      "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|perception|>"]
    new_tokens = [token for token in special_tokens if token not in tokenizer.vocab]
    
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if torch.cuda.current_device() == 0:
            print(f"Added {len(new_tokens)} new special tokens")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with better device mapping for multi-GPU
    device_map = None
    if torch.cuda.is_available():
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size == 1:
            device_map = "auto"
        else:
            # For multi-GPU with torchrun, don't use device_map
            device_map = None
    
    if pretrain:
        if seed is not None:
            torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)
        for b in model.buffers():
            torch.distributed.broadcast(b.data, src=0)
        if debug == 6:
            for name, param in model.named_parameters():
                print(f"Parameter name: {name}, Shape: {param.shape}")
                print(param)
                exit(0)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
            # Add these for better multi-GPU stability
            low_cpu_mem_usage=True,
        )

    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Resize embeddings if we added new tokens
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA if requested
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        if torch.cuda.current_device() == 0:
            model.print_trainable_parameters()
    
    return model, tokenizer


class DynamicPaddingCollator:
    """Pad each batch to its own longest sequence instead of a fixed max_length.

    Pads the main / user / assistant sequence groups to one common per-batch
    length so the downstream torch.cat and additive-mask logic still line up.
    When additive_mask is set, the common length is also grown to hold
    user + assistant tokens, since the trainer splices the assistant tokens into
    the tail of the user tensor (see build_with_additive_mask).
    """

    def __init__(self, pad_token_id, additive_mask=False, pad_to_multiple_of=64):
        self.pad_token_id = pad_token_id
        self.additive_mask = additive_mask
        self.pad_to_multiple_of = pad_to_multiple_of

    @staticmethod
    def _pad(sequences, pad_value, length):
        return [seq + [pad_value] * (length - len(seq)) for seq in sequences]

    def __call__(self, features):
        groups = [g for g in ("", "_user", "_assistant") if f"input_ids{g}" in features[0]]

        target_len = max(len(f[f"input_ids{g}"]) for f in features for g in groups)
        if self.additive_mask and "_user" in groups and "_assistant" in groups:
            # Assistant tokens get spliced into the user tensor's tail at train
            # time, so user/main must have room for user + assistant.
            target_len = max(
                target_len,
                max(len(f["input_ids_user"]) + len(f["input_ids_assistant"]) for f in features),
            )
        if self.pad_to_multiple_of:
            target_len = ((target_len + self.pad_to_multiple_of - 1)
                          // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch = {}
        for g in groups:
            batch[f"input_ids{g}"] = torch.tensor(
                self._pad([f[f"input_ids{g}"] for f in features], self.pad_token_id, target_len),
                dtype=torch.long)
            batch[f"labels{g}"] = torch.tensor(
                self._pad([f[f"labels{g}"] for f in features], -100, target_len),
                dtype=torch.long)
            batch[f"attention_mask{g}"] = torch.tensor(
                self._pad([f[f"attention_mask{g}"] for f in features], 0, target_len),
                dtype=torch.long)
        return batch


class RepresentationTrainer(Trainer):
    """
    Trainer to regularize representations.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract custom loss parameters
        self.lbd = kwargs.pop('lbd', 1.0)
        self.gamma = kwargs.pop('gamma', 1.0)
        self.last_token = kwargs.pop('last_token', -2)
        self.debug = kwargs.pop('debug', 0)
        self.additive_mask = kwargs.pop('additive_mask', False)
        self.jepa_l2 = kwargs.pop('jepa_l2', False)
        self.jepa_mse = kwargs.pop('jepa_mse', False)
        self.infonce = kwargs.pop('infonce', False)
        self.jepa_ratio = kwargs.pop('jepa_ratio', -1.0)
        assert self.jepa_l2 + self.jepa_mse <= 1, "Only one of jepa_l2 and jepa_mse can be True."
        super().__init__(*args, **kwargs)
    
    def _last_token_index(self, input_ids, labels, attention_mask):
        index = []
        for i in range(input_ids.shape[0]):
            length = attention_mask[i].sum().item()
            index.append(length + self.last_token)
        
        index_tensor = torch.tensor(index).to(input_ids.device)
        if self.debug == 1 and torch.cuda.current_device() == 0:
            print(index_tensor)

        return index_tensor
    
    def _build_additive_mask(self, k: int):
        mask = torch.zeros((k, k), dtype=torch.float32)
        mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
        return mask

    def build_with_additive_mask(self, inputs):
        if self.jepa_ratio > 0.0:
            if torch.rand(1).item() > self.jepa_ratio:
                return {
                    "input_ids": inputs["input_ids"],
                    "labels": inputs["labels"],
                    "attention_mask": inputs["attention_mask"],
                }, True
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device
        mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf).to(device)
        last_token = self._last_token_index(inputs["input_ids"], inputs["labels"], inputs["attention_mask"])        
        last_token_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
        last_token_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
        clamped_assistant_lengths = []
        for i in range(inputs["input_ids_user"].shape[0]):
            length, length_user, length_assistant = last_token[i] + 1, last_token_user[i] + 1, last_token_assistant[i] + 1
            # Clamp assistant length so user + assistant fits within seq_length
            length_assistant_clamped = min(length_assistant, seq_length - length_user)
            clamped_assistant_lengths.append(length_assistant_clamped)
            if length_assistant_clamped > 0:
                inputs["input_ids_user"][i, length_user:length_user + length_assistant_clamped] = inputs["input_ids_assistant"][i, :length_assistant_clamped]
                inputs["labels_user"][i, length_user:length_user + length_assistant_clamped] = inputs["labels_assistant"][i, :length_assistant_clamped]
            mask[i, :, 0:length, 0:length] = self._build_additive_mask(length)
            mask[i + batch_size, :, 0:length_user, 0:length_user] = self._build_additive_mask(length_user)
            if length_assistant_clamped > 0:
                mask[i + batch_size, :, length_user:length_user + length_assistant_clamped, length_user:length_user + length_assistant_clamped] = self._build_additive_mask(length_assistant_clamped)
        self._last_token_user = last_token_user
        # Use clamped lengths for assistant index to avoid out-of-bounds access
        clamped_assistant_tensor = torch.tensor(clamped_assistant_lengths, device=device) - 1
        self._last_token_assistant = clamped_assistant_tensor + last_token_user + 1
        return {
                "input_ids": torch.cat([inputs["input_ids"],
                                        inputs["input_ids_user"]], dim=0),
                "labels": torch.cat([inputs["labels"],
                                    inputs["labels_user"]], dim=0),
                "attention_mask": mask,
            }, False

    def forward(self, model, inputs):
        """
        Custom forward pass that handles all model calls.
        """
        # Main forward pass for language modeling
        if self.additive_mask:
            llm_inputs, skip_jepa = self.build_with_additive_mask(inputs)
        else:
            llm_inputs = {
                "input_ids": torch.cat([inputs["input_ids"],
                                        inputs["input_ids_user"],
                                        inputs["input_ids_assistant"]], dim=0),
                "labels": torch.cat([inputs["labels"],
                                    inputs["labels_user"],
                                    inputs["labels_assistant"]], dim=0),
                "attention_mask": torch.cat([inputs["attention_mask"],
                                            inputs["attention_mask_user"],
                                            inputs["attention_mask_assistant"]], dim=0),
            }
        if self.debug == 7 and torch.cuda.current_device() == 0:
            torch.set_printoptions(threshold=float("inf"))
            torch.set_printoptions(linewidth=360)
            print(">>>input_ids<<<")
            print(llm_inputs["input_ids"])
            print(">>>labels<<<")
            print(llm_inputs["labels"])
            print(">>>attention_mask<<<")
            print(llm_inputs["attention_mask"])
            if self.additive_mask:
                print(">>>last_token_user<<<")
                print(self._last_token_user)
                print(">>>last_token_assistant<<<")
                print(self._last_token_assistant)
        if self.debug == 7:
            exit(0)
        if self.debug == 2 and torch.cuda.current_device() == 0:
            print("=====before:outputs=====")
            print("input_ids shapes:")
            print(llm_inputs["input_ids"].shape)
            print("labels shapes::")
            print(llm_inputs["labels"].shape)
            print("attention_mask shapes:")
            print(llm_inputs["attention_mask"].shape)

        with torch.set_grad_enabled(True):
            outputs = model(**llm_inputs, output_hidden_states=True)

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(f"=====outputs.loss.shape:{outputs.loss.shape}=====")
            print(f"=====outputs.hidden_states[-1].shape:{outputs.hidden_states[-1].shape}=====")
        
        if self.additive_mask:
            if skip_jepa:
                user_hidden_states = None
                assistant_hidden_states = None
            else:    
                batch_size = llm_inputs["input_ids"].shape[0] // 2
                user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
                assistant_hidden_states = user_hidden_states
        else:
            batch_size = llm_inputs["input_ids"].shape[0] // 3
            user_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2]
            assistant_hidden_states = outputs.hidden_states[-1][batch_size * 2:]

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(f"====={user_hidden_states.shape}=====")
            print(f"====={assistant_hidden_states.shape}=====")
       
        # Return all outputs needed for loss computation
        return {
            'main_outputs': outputs,
            'user_hidden_states': user_hidden_states,
            'assistant_hidden_states': assistant_hidden_states,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with additional regularization terms.
        """
        # Get indeices
        if not self.additive_mask:
            index_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
            index_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])
        first_dim = inputs["input_ids_user"].shape[0]
        if self.debug == 1 and torch.cuda.current_device() == 0:
            print("=====last tokens=====")
            print(inputs["input_ids_user"][range(first_dim), index_user])
            print(inputs["input_ids_user"][range(first_dim), index_user - 1])
            print(inputs["input_ids_assistant"][range(first_dim), index_assistant])
            print(inputs["input_ids_assistant"][range(first_dim), index_assistant - 1])

        # Get all forward pass results
        forward_results = self.forward(model, inputs)
        
        # Extract main language modeling loss
        main_outputs = forward_results['main_outputs']
        lm_loss = main_outputs.loss

        # Compute representation similarity loss
        user_hidden_states = forward_results['user_hidden_states']
        assistant_hidden_states = forward_results['assistant_hidden_states']
        
        # Get embeddings (using last token of each sequence)
        if user_hidden_states is not None:
            if self.additive_mask:
                index_user = self._last_token_user
                index_assistant = self._last_token_assistant
            user_embedding = user_hidden_states[range(first_dim), index_user, :]
            assistant_embedding = assistant_hidden_states[range(first_dim), index_assistant, :]
            
            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(user_embedding, assistant_embedding, dim=-1)
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print(user_embedding.shape, assistant_embedding.shape)
                print(cosine_similarity.shape)
    
            # Compute total loss
            if self.jepa_l2:
                jepa_loss = torch.linalg.norm(user_embedding - assistant_embedding, ord=2, dim=-1).mean()
            elif self.jepa_mse:
                jepa_loss = torch.mean((user_embedding - assistant_embedding) ** 2)
            elif self.infonce:
                ue_norm = F.normalize(user_embedding, p=2, dim=1)
                ae_norm = F.normalize(assistant_embedding, p=2, dim=1)
                cosine_sim = torch.mm(ue_norm, ae_norm.T)
                infonce_logit = cosine_sim / 0.07  # temperature
                infonce_label = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
                jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
                if self.debug == 8:
                    print(cosine_sim.shape, infonce_logit.shape, infonce_label.shape, jepa_loss.shape)
                    exit(0)
            else:
                jepa_loss = 1.0 - torch.mean(cosine_similarity)
        else:
            jepa_loss = 0.0

        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(lm_loss, self.lbd, torch.mean(cosine_similarity))

        if self.debug == 1 or self.debug == 2:
            exit(0)

        if self.debug == 5 and torch.cuda.current_device() == 0:
            print(f"llm_loss: {lm_loss.float()}, jepa_loss: {jepa_loss.float()}")

        return (total_loss, main_outputs) if return_outputs else total_loss


class ProfilerFLOPCallback(TrainerCallback):
    def __init__(self, profile_steps=10):
        self.profile_steps = profile_steps
        self.total_flops = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True  # This enables FLOP counting if available
            )
            self.profiler.__enter__()
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.profile_steps:
            self.profiler.__exit__(None, None, None)
            
            # Extract FLOP information
            events = self.profiler.key_averages()
            step_flops = sum(event.flops for event in events if event.flops > 0)
            self.total_flops += step_flops
            
            if torch.cuda.current_device() == 0:  # and (state.global_step == 63 or state.global_step % 10 == 0):
                print(f"Step {state.global_step}: FLOPs: {step_flops:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-1B")
    parser.add_argument("--train_file", type=str, help="Path to training JSONL file")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation JSONL file")
    parser.add_argument("--data_file", type=str, help="Path to single JSONL file (will be split)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name/path")
    parser.add_argument("--output_dir", type=str, default="./llama3-1b-fted", help="Output directory")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation steps")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA (default: full fine-tuning)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank. Default: 16.")
    parser.add_argument("--eval_split", type=float, default=0.2, help="Evaluation split ratio (if using single data file)")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for train/eval split")
    parser.add_argument("--finetune_seed", type=int, default=42, help="Random seed for fine-tuning")
    parser.add_argument("--predictors", type=int, default=0, help="Number of predictor tokens")
    parser.add_argument("--lbd", type=float, default=0.1, help="Lambda for similarity loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for LLM loss")
    parser.add_argument("--last_token", type=int, default=-1, help="Index of last token, -1 is '<|eot|>'")
    parser.add_argument("--debug", type=int, default=0, help="Debug level. 0 means no debug")
    parser.add_argument("--regular", action="store_true", help="Use regular transformer.")
    parser.add_argument("--track_flop", action="store_true", help="Whether to track FLOPs.")
    parser.add_argument("--pretrain", action="store_true", help="Whether to pretrain from scratch.")
    parser.add_argument("--train_all", action="store_true", help="Whether to compute loss from all tokens.")
    parser.add_argument("--plain", action="store_true", help="When set, do not apply chat format. If --train_all is not set, use `<|perceptioin|>` to connect query and answer. If --train_all is set, only train query.")
    parser.add_argument("--additive_mask", action="store_true", help="When set, Use an additive mask to compute both user and assistant in 1 forward pass.")
    parser.add_argument("--jepa_l2", action="store_true", help="When set, Use l2 norm as JEPA loss.")
    parser.add_argument("--jepa_mse", action="store_true", help="When set, Use Mean Squared Error as JEPA loss.")
    parser.add_argument("--front_pred", action="store_true", help="When set, Put [Pred] token at the beginning of `Text`.")
    parser.add_argument("--reverse_pred", action="store_true", help="When set, Use `Code` to predict `Text`.")
    parser.add_argument("--infonce", action="store_true", help="When set, Use InfoNCE loss.")
    parser.add_argument("--same_flop", action="store_true", help="When set, Use same number of flops per epoch.")
    parser.add_argument("--jepa_ratio", type=float, default=-1.0, help="When >0, randomly select this ratio of batches to apply JEPA. This implments Random JEPA-Loss Dropout (LD). If LD = alpha, jepa_ratio = 1 - alpha")

    args = parser.parse_args()
    
    # Validate arguments
    if not args.train_file and not args.data_file:
        parser.error("Must provide either --train_file or --data_file")
    
    if args.train_file and args.data_file:
        parser.error("Cannot use both --train_file and --data_file. Choose one.")
    
    if torch.cuda.current_device() == 0:
        print("=== Fine-tuning Script ===")

    if torch.cuda.current_device() == 0:
        if args.train_file:
            print(f"Train file: {args.train_file}")
            if args.eval_file:
                print(f"Eval file: {args.eval_file}")
            else:
                print("No eval file provided - training without evaluation")
        else:
            print(f"Data file: {args.data_file} (will split {args.eval_split:.1%} for eval)")
    
        print(f"Model: {args.model_name}")
        print(f"Output: {args.output_dir}")
        print(f"Using LoRA: {args.lora}")
        print(f"LoRA rank: {args.lora_rank}")
    
    # Check if running with torchrun
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        if torch.cuda.current_device() == 0:
            print(f"Running with torchrun: world_size={world_size}, local_rank={local_rank}")
        # Initialize distributed training
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    # Setup model and tokenizer
    if torch.cuda.current_device() == 0:
        print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name, use_lora=args.lora, lora_rank=args.lora_rank, pretrain=args.pretrain,
        debug=args.debug, seed=args.finetune_seed)
    
    # Load and prepare dataset
    if torch.cuda.current_device() == 0:
        print("\n2. Loading and preparing dataset...")
    
    if args.train_file:
        # Load separate train and eval files
        if torch.cuda.current_device() == 0:
            print(f"Loading training data from {args.train_file}")
        train_dataset = load_and_prepare_dataset(
            args.train_file, tokenizer, args.model_name,
            args.max_length, predictors=args.predictors, regular=args.regular,
            debug=args.debug, train_all=args.train_all, plain=args.plain,
            front_pred=args.front_pred, reverse_pred=args.reverse_pred)
        
        if args.eval_file:
            if torch.cuda.current_device() == 0:
                print(f"Loading evaluation data from {args.eval_file}")
            eval_dataset = load_and_prepare_dataset(
                args.eval_file, tokenizer, args.model_name,
                args.max_length, regular=args.regular,
                debug=args.debug, train_all=args.train_all, plain=args.plain,
                front_pred=args.front_pred, reverse_pred=args.reverse_pred)
        else:
            eval_dataset = None
            if torch.cuda.current_device() == 0:
                print("No evaluation file provided")
    
    else:
        # Load single file and split
        if torch.cuda.current_device() == 0:
            print(f"Loading data from {args.data_file} and splitting...")
        full_dataset = load_and_prepare_dataset(
            args.data_file, tokenizer, args.model_name,
            args.max_length, predictors=args.predictors, regular=args.regular,
            debug=args.debug, train_all=args.train_all, plain=args.plain,
            front_pred=args.front_pred, reverse_pred=args.reverse_pred)

        if args.eval_split > 0:
            split_dataset = full_dataset.train_test_split(
                test_size=args.eval_split, 
                seed=args.split_seed,
                shuffle=True
            )
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = full_dataset
            eval_dataset = None
    
    # Print dataset info
    if torch.cuda.current_device() == 0:
        print(f"Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval samples: {len(eval_dataset)}")
        else:
            print("No evaluation dataset")
    
    # Data collator - pad each batch dynamically to its longest sample
    data_collator = DynamicPaddingCollator(
        pad_token_id=tokenizer.pad_token_id,
        additive_mask=args.additive_mask,
        pad_to_multiple_of=64,
    )
    
    # Training arguments - optimized for multi-GPU stability
    eval_steps = args.eval_steps if not args.pretrain else args.eval_steps * 20
    save_steps = len(train_dataset) // (world_size * args.batch_size * args.grad_accum)
    if args.same_flop:
        if args.jepa_ratio > 0.0:
            save_steps = int(save_steps / (1 + args.jepa_ratio))
            args.num_epochs = int(math.ceil(args.num_epochs / (1 + args.jepa_ratio)))
        elif args.additive_mask:
            save_steps = save_steps // 2
            args.num_epochs = int(math.ceil(args.num_epochs / 2))
        elif not args.regular:
            save_steps = save_steps // 3
            args.num_epochs = int(math.ceil(args.num_epochs / 3))
        if torch.cuda.current_device() == 0:
            print(f">>>>> --same_flop is active: Save checkpoint every: {save_steps} steps, run {args.num_epochs} epochs")
    output_dir = os.path.abspath(args.output_dir)
    if torch.cuda.current_device() == 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training parameters
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        
        # Evaluation
        eval_strategy="no",  # "steps" if eval_dataset else "no",
        # eval_steps=eval_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.num_epochs * 4,

        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.eval_steps,
        
        # Optimization - key changes for stability
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,  # Enable for memory efficiency
        dataloader_drop_last=True,   # Drop last incomplete batch
        train_sampling_strategy="group_by_length",  # Batch similar-length samples to minimize padding (transformers 5.x; replaces group_by_length=True)
        
        # Memory optimization
        dataloader_num_workers=0,    # Avoid multiprocessing issues
        
        # Multi-GPU settings - completely disable FSDP
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
        
        # Explicitly disable FSDP and sharding
        fsdp="",
        fsdp_config={},
        
        # Other
        report_to="wandb",
        remove_unused_columns=False,
        load_best_model_at_end=True if eval_dataset else False,
        
        # Disable problematic optimizations
        tf32=False,  # May help with stability
        
        # Set seed for reproducibility
        seed=args.finetune_seed,
        data_seed=args.finetune_seed,
    )
    
    flop_callback = ProfilerFLOPCallback()

    # Initialize trainer
    if args.regular:
        if torch.cuda.current_device() == 0:
            print("\n3. Initializing regular trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[flop_callback] if args.track_flop else [],
        )
    else:
        if torch.cuda.current_device() == 0:
            print("\n3. Initializing representation trainer...")
        trainer = RepresentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[flop_callback] if args.track_flop else [],
            lbd=args.lbd,
            gamma=args.gamma,
            last_token=args.last_token,
            debug=args.debug,
            additive_mask=args.additive_mask,
            jepa_l2=args.jepa_l2,
            jepa_mse=args.jepa_mse,
            infonce=args.infonce,
            jepa_ratio=args.jepa_ratio,
        )
    
    if torch.cuda.current_device() == 0 and args.lora:
        print("=== PEFT Model Check ===")
        model.print_trainable_parameters()

        # Check if any parameters require gradients
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)

        print(f"Trainable parameters: {len(trainable_params)}")
        if len(trainable_params) == 0:
            print("ERROR: No parameters require gradients!")
        else:
            print("First few trainable params:", trainable_params[:5])

    # Start training
    if torch.cuda.current_device() == 0:
        print("\n4. Starting training...")
    try:
        trainer.train()
    except Exception as e:
        if torch.cuda.current_device() == 0:
            print(f"Training failed with error: {e}")
            print("This might be due to FSDP/sharding issues. Try running with --lora flag for LoRA fine-tuning.")
        raise
    
    # Save final model
    if torch.cuda.current_device() == 0:
        print("\n5. Saving final model...")

    def save_model(model):
        if torch.cuda.current_device() == 0:
            if args.lora:
                model = model.merge_and_unload()
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                trainer.save_model()
                trainer.save_state()
                tokenizer.save_pretrained(output_dir)

    retry = 3
    while retry > 0:
        try:
            save_model(model)
            break
        except Exception as e:
            print(f"Success Rate: Saving model encounter error: {e}")
            retry -= 1
            if retry <= 0:
                raise
            time.sleep(10)
    
    if torch.cuda.current_device() == 0:
        print(f"\n✅ Training completed! Model saved to {args.output_dir}")
    
    if torch.cuda.current_device() == 0:
        print("\n🎉 Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()
