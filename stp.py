"""Semantic Tube Prediction.
"""

import copy
import math
import numpy as np
import os
# import re
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import json
from datasets import load_dataset
import shutil
from safetensors.torch import load_file
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

    if "google/gemma" in model_name:
        assistant_messages = copy.deepcopy(messages)[2:3]
        assistant_messages[0]["role"] = "user"
        return assistant_messages
    else:
        return messages[2:3]


def load_and_prepare_dataset(data_file, tokenizer, model_name,
                             max_length=2048, debug=0, predictors=0, regular=False, train_all=False,
                             plain=False, front_pred=False, reverse_pred=False, linear=None, plain_jepa=False,
                             random_span_mask=False, same_predictor=False):
    """Load JSONL dataset and format for training with proper label masking"""
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_file)['train']
    if  torch.cuda.current_device() == 0:
        print(f"Loaded {len(dataset)} examples from {data_file}")
    
    def tokenize_conversations(examples):
        """Tokenize conversations and mask input tokens properly"""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        user_input_ids_list = []
        user_labels_list = []
        user_attention_mask_list = []
        user_start_end_list = []
        assistant_input_ids_list = []
        assistant_labels_list = []
        assistant_attention_mask_list = []
        assistant_start_end_list = []

        for msg_idx, messages in enumerate(examples['messages']):
            # Apply chat template if available, otherwise format manually
            full_messages = get_messages(model_name, messages)
            if plain:
                if train_all:
                    formatted_chat = messages[1]["content"] + "<|eot_id|>"
                else:
                    formatted_chat = messages[1]["content"] + "\n<|perception|>" + messages[2]["content"] + "<|eot_id|>"
            else:
                formatted_chat = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            
            # Tokenize the formatted conversation with padding to max_length
            tokenized = tokenizer(
                formatted_chat,
                truncation=True,
                max_length=max_length,
                padding="max_length",  # Pad to max_length for consistent tensor shapes
                return_tensors=None
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Create labels with proper masking
            if train_all:
                labels = create_labels_for_all(input_ids, attention_mask)
            else:
                labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask)
            
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
                    if same_predictor:
                        user_messages[0]["content"] += f"<|predictor_1|>"
                    else:
                        user_messages[0]["content"] += f"<|predictor_{to_add}|>"
                to_add -= 1
            if plain or plain_jepa:
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
                padding="max_length",  # Pad to max_length for consistent tensor shapes
                return_tensors=None
            )
            user_input_ids_list.append(tokenized_user["input_ids"])
            user_labels_list.append([-100] * len(tokenized_user["input_ids"]))
            user_attention_mask_list.append(tokenized_user["attention_mask"])
            if data_file.startswith("hellaswag"):
                content = examples["text"][msg_idx][0]["content"] + "\n"
                user_start, user_end = find_start_end(content, tokenizer, input_ids, attention_mask)
            elif "allenai/OLMo" in model_name:
                content = messages[1]["content"] + "\n"
                user_start, user_end = find_start_end(content, tokenizer, input_ids, attention_mask)
            else:
                user_start, user_end = find_start_end(messages[1]["content"], tokenizer, input_ids, attention_mask)
            user_start_end_list.append([user_start, user_end])

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
            if plain or plain_jepa:
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
                padding="max_length",  # Pad to max_length for consistent tensor shapes
                return_tensors=None
            )
            assistant_input_ids_list.append(tokenized_assistant["input_ids"])
            assistant_labels_list.append([-100] * len(tokenized_assistant["input_ids"]))
            assistant_attention_mask_list.append(tokenized_assistant["attention_mask"])
            if data_file.startswith("hellaswag"):
                content = examples["code"][msg_idx][0]["content"]
                content = " " + content + "\n"
                # if messages[2]["content"] != "D":
                #     content += "\n"
                assistant_start, assistant_end = find_start_end(content, tokenizer, input_ids, attention_mask)
            elif "apple/OpenELM" in model_name:
                try:
                    assistant_start, assistant_end = find_start_end(messages[2]["content"], tokenizer, input_ids, attention_mask)
                except AssertionError:
                    content = "\n" + messages[2]["content"]
                    assistant_start, assistant_end = find_start_end(content, tokenizer, input_ids, attention_mask, remove_leading_newline=True)
                    # TODO: see if this is needed? (assistant_start += 1)
            else:
                assistant_start, assistant_end = find_start_end(messages[2]["content"], tokenizer, input_ids, attention_mask)
            assistant_start_end_list.append([assistant_start, assistant_end])

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
            
            def print_indexed_list(list_of):
                to_print = []
                for i, item in enumerate(list_of):
                    to_print.append(f"{i}:{item}")
                print("[" + ", ".join(to_print) + "]")
            if debug == 9 and torch.cuda.current_device() == 0:
                print("input_ids")
                print_indexed_list(input_ids)
                print("decoded input_ids")
                print_indexed_list([tokenizer.decode(item) for item in input_ids])
                print(f"user content: {messages[1]['content']}")
                print(f"assistant content: {messages[2]['content']}")
                print(f"user start end: {user_start_end_list[0]}")
                print(f"assistant start end: {assistant_start_end_list[0]}")

            if debug == 9:
                exit(0)

        if regular:
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
            }
        elif linear is not None or random_span_mask:
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
                "user_start_end": user_start_end_list,
                "assistant_start_end": assistant_start_end_list,
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

    def create_masked_labels(messages, tokenizer, input_ids, attention_mask):
        """Create labels with input tokens masked (-100)"""
        labels = [-100] * len(input_ids)
        
        # Mask padding tokens in labels
        for i, mask in enumerate(attention_mask):
            if mask == 0:  # Padding token
                labels[i] = -100
        
        # Find assistant responses and unmask only those tokens
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                
                # Find where this assistant response appears in the tokenized text
                assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
                
                # Find the position of assistant response in input_ids
                decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
                decoded_input = [tokenizer.decode(item) for item in input_ids]
                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    # Only check non-padding tokens
                    if debug == 4 and torch.cuda.current_device() == 0:
                        print(f"=======input_ids: {input_ids[i:i+len(assistant_tokens)]}")
                        print(f"assistant_tokens: {assistant_tokens}")
                    # if attention_mask[i] == 1 and input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    if attention_mask[i] == 1 and decoded_input[i:i+len(assistant_tokens)] == decoded_assistant:
                        # Unmask the assistant response tokens
                        for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                            if attention_mask[j] == 1:  # Only unmask non-padding tokens
                                labels[j] = input_ids[j]
                        break
                
                if debug == 4:
                    exit(0)
        
        return labels
    
    def print_indexed_array(one_d_array):
        to_print = []
        for i, item in enumerate(one_d_array):
            to_print.append(f"{i}:{item}")
        print("[" + ", ".join(to_print) + "]")

    def find_start_end(content, tokenizer, input_ids, attention_mask, remove_leading_newline=False):
        """Find the start and end index of the content in the input_ids."""
        tokens = tokenizer.encode(content, add_special_tokens=False)
        if remove_leading_newline:
            if "apple/OpenELM" in model_name:
                assert tokens[0] == 29871 and tokens[1] == 13, f"{tokens[0]} != 29871 or {tokens[1]} != 13"
                tokens = tokens[2:]
        decoded_content = [tokenizer.decode(item) for item in tokens]
        decoded_input = [tokenizer.decode(item) for item in input_ids]
        if debug == 15 and torch.cuda.current_device() == 0:
            print_indexed_array(decoded_content)
            print_indexed_array(tokens)
            print_indexed_array(decoded_input)
            print_indexed_array(input_ids)
        for i in range(len(input_ids) - len(tokens), -1, -1):
            if debug == 12 and torch.cuda.current_device() == 0:
                print(f"=======input_ids: {input_ids[i:i+len(tokens)]}")
                print(f"assistant_tokens: {tokens}")
            # if attention_mask[i] == 1 and input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
            if attention_mask[i] == 1 and decoded_input[i:i+len(tokens)] == decoded_content:
                # Unmask the assistant response tokens
                if debug == 12 and torch.cuda.current_device() == 0:
                    print(f"start: {i}, end: {i + len(tokens) - 1}")
                if debug == 12:
                    exit(0)
                assert i > 0
                return i - 1, i + len(tokens) - 1

        assert False, f"Cannot find {content} in input {input_ids}"
        return None, None
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_conversations,
        batched=True,
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


class LinearPredictor(nn.Module):

    def __init__(self, dx: int, dy: int | None = None, bias: bool = False):
        super().__init__()
        if dy is None:
            dy = dx
        self.M = nn.Linear(dx, dy, bias=bias)
        nn.init.xavier_uniform_(self.M.weight, gain=1.0)
        if bias:
            nn.init.zeros_(self.M.bias)
        self.dx = dx
        self.dy = dy

    def forward(self, x):
        assert x.dim() == 2 and x.shape[1] == self.dx
        return self.M(x)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_model_and_tokenizer(model_name, use_lora=True, lora_rank=16, pretrain=False, debug=0, seed=None, linear_predictor=False, load_lp=False):
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
    
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        if torch.cuda.current_device() == 0:
            print(f"Added <|mask|> token: {tokenizer.mask_token_id}")

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
    elif load_lp:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
        )
        if not hasattr(model.config, "hidden_size"):
            d = model.config.model_dim
        else:
            d = model.config.hidden_size
        model.linear_predictor = LinearPredictor(d)
        state_dict = load_file(os.path.join(model_name, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            # Add these for better multi-GPU stability
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable KV cache for training
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

    if linear_predictor:
        if not load_lp:
            if seed is not None:
                set_seeds(seed)
            if not hasattr(model.config, "hidden_size"):
                d = model.config.model_dim
            else:
                d = model.config.hidden_size
            model.linear_predictor = LinearPredictor(d)
        if debug == 10:
            print(model.linear_predictor.M.weight)

    return model, tokenizer


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
        self.linear = kwargs.pop('linear', None)
        self.linear_predictor = kwargs.pop('linear_predictor', False)
        self.lbd_warmup = kwargs.pop('lbd_warmup', False)
        self.min_lbd = kwargs.pop('min_lbd', 0.0)
        self.length_adjustment = kwargs.pop('length_adjustment', None)
        self.random_span_zero = kwargs.pop('random_span_zero', False)
        self.random_span_e2e = kwargs.pop('random_span_e2e', False)
        self.random_span_all = kwargs.pop('random_span_all', False)
        self.random_span_mask = kwargs.pop('random_span_mask', False)
        self.random_span_mask_recover=kwargs.pop('random_span_mask_recover', False)
        self.random_span_max_length = kwargs.pop('random_span_max_length', -1)
        self.random_span_draw_both = kwargs.pop('random_span_draw_both', False)
        self.random_span_times = kwargs.pop('random_span_times', 1)
        self.random_span_uniform = kwargs.pop('random_span_uniform', False)
        self.random_span_layer = kwargs.pop('random_span_layer', -1)
        self.curvature_sign = kwargs.pop('curvature_sign', False)
        self.avg_encoding = kwargs.pop('avg_encoding', False)
        if self.avg_encoding:
            assert not self.additive_mask, f"additive_mask cannot be set if avg_encoding is set."
            assert self.linear is None, f"linear cannot be set if avg_encoding is set."
        if self.random_span_mask:
            assert self.random_span_times == 1, f"random_span_times ({self.random_span_times}) must = 1 when random_span_mask is {self.random_span_mask}."
        assert self.jepa_l2 + self.jepa_mse <= 1, "Only one of jepa_l2 and jepa_mse can be True."
        super().__init__(*args, **kwargs)
        rank = getattr(self.args, "process_index", 0)
        self._g = torch.Generator(device=self.args.device)
        self._g.manual_seed(self.args.seed + rank * 3)
        if self.debug == 11:
            print(f"On rank {rank}, using seed {self.args.seed + rank * 3} tied to device {self.args.device}")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
    
    def get_lbd(self):
        if not self.lbd_warmup:
            return self.lbd
        lbd = self.min_lbd + (self.lbd - self.min_lbd) * self.state.global_step / (self.state.max_steps - 1)
        if self.debug == 11 and torch.cuda.current_device() == 0:
            print(f"lbd: {lbd} @ {self.state.global_step} / {self.state.max_steps}")
        return lbd

    def _last_token_index(self, input_ids, labels, attention_mask):
        index = []
        def unpad(input_ids, attention_mask):
            result = []
            can_break = False
            for id, mask in zip(input_ids, attention_mask):
                if mask != 0:
                    can_break = True
                if mask == 0 and can_break:
                    break
                result.append(id)
            return result

        for i in range(input_ids.shape[0]):
            uii = unpad(input_ids[i], attention_mask[i])
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print(f"====={len(uii)}=====")
                print(input_ids[i][len(uii) - 4], input_ids[i][len(uii) - 3], input_ids[i][len(uii) - 2], input_ids[i][len(uii) - 1], -100 if len(uii) >= len(input_ids[i]) else input_ids[i][len(uii)])
                print(labels[i][len(uii) - 4], labels[i][len(uii) - 3], labels[i][len(uii) - 2], labels[i][len(uii) - 1], -100 if len(uii) >= len(labels[i]) else labels[i][len(uii)])
                print(attention_mask[i][len(uii) - 4], attention_mask[i][len(uii) - 3], attention_mask[i][len(uii) - 2], attention_mask[i][len(uii) - 1], -100 if len(uii) >= len(attention_mask[i]) else attention_mask[i][len(uii)])
            if self.random_span_mask:
                index.append(len(uii) - 1)
            else:
                index.append(len(uii) + self.last_token)
        
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
        for i in range(inputs["input_ids_user"].shape[0]):
            length, length_user, length_assistant = last_token[i] + 1, last_token_user[i] + 1, last_token_assistant[i] + 1
            inputs["input_ids_user"][i, length_user:length_user + length_assistant] = inputs["input_ids_assistant"][i, :length_assistant]
            inputs["labels_user"][i, length_user:length_user + length_assistant] = inputs["labels_assistant"][i, :length_assistant]
            mask[i, :, 0:length, 0:length] = self._build_additive_mask(length)
            mask[i + batch_size, :, 0:length_user, 0:length_user] = self._build_additive_mask(length_user)
            mask[i + batch_size, :, length_user:length_user + length_assistant, length_user:length_user + length_assistant] = self._build_additive_mask(length_assistant)
        self._last_token_user = last_token_user
        self._last_token_assistant = last_token_assistant + last_token_user + 1
        return {
                "input_ids": torch.cat([inputs["input_ids"],
                                        inputs["input_ids_user"]], dim=0),
                "labels": torch.cat([inputs["labels"],
                                    inputs["labels_user"]], dim=0),
                "attention_mask": mask,
            }, False

    def print_indexed_array(self, one_d_array):
        to_print = []
        for i, item in enumerate(one_d_array):
            to_print.append(f"{i}:{item}")
        print("[" + ", ".join(to_print) + "]")

    def get_s_t(self, full_length: int):
        if self.random_span_draw_both:
            while True:
                pick_1 = torch.randint(0, full_length, (1,), generator=self._g, device=self._g.device)
                pick_2 = torch.randint(1, full_length + 1, (1,), generator=self._g, device=self._g.device)
                patch_start_offset = torch.min(pick_1, pick_2)
                patch_end_offset = torch.max(pick_1, pick_2)
                if self.random_span_zero:
                    patch_start_offset = 0
                if self.random_span_max_length >= 0 and patch_end_offset - patch_start_offset > self.random_span_max_length:
                    patch_end_offset = patch_start_offset + self.random_span_max_length
                if patch_start_offset < patch_end_offset and patch_end_offset - patch_start_offset < full_length:
                    break
        elif self.random_span_uniform:
            total = full_length * (full_length + 1) // 2
            while True:
                r = torch.randint(total, (1,), generator=self._g, device=self._g.device, dtype=torch.long)
                two_n_plus_1 = 2 * full_length + 1
                patch_start_offset = torch.floor((two_n_plus_1 - torch.sqrt((two_n_plus_1**2 - 8 * r.float()))) / 2).long()
                prev = patch_start_offset * (2 * full_length - patch_start_offset + 1) // 2
                patch_end_offset = patch_start_offset + 1 + (r - prev)
                if patch_end_offset - patch_start_offset < full_length:
                    break
        else:
            if self.random_span_zero:
                patch_start_offset = 0
            else:
                patch_start_offset = torch.randint(0, full_length, (), generator=self._g, device=self._g.device)
            max_offset_exclusive = full_length + 1
            if self.random_span_max_length >= 0:
                max_offset_exclusive = min(max_offset_exclusive, patch_start_offset + 1 + self.random_span_max_length)
            while True:
                patch_end_offset = torch.randint(patch_start_offset + 1, max_offset_exclusive, (), generator=self._g, device=self._g.device)
                if patch_end_offset - patch_start_offset < full_length:
                    break
        return patch_start_offset, patch_end_offset

    def forward(self, model, inputs):
        """
        Custom forward pass that handles all model calls.
        """
        # Main forward pass for language modeling
        if self.random_span_mask:
            inputs["input_ids_user"] = torch.zeros_like(inputs["input_ids"]) + self.eos_token_id
            inputs["labels_user"] = torch.zeros_like(inputs["labels"])
            inputs["attention_mask_user"] = torch.zeros_like(inputs["attention_mask"])
            inputs["input_ids_assistant"] = torch.zeros_like(inputs["input_ids"]) + self.eos_token_id
            inputs["labels_assistant"] = torch.zeros_like(inputs["labels"])
            inputs["attention_mask_assistant"] = torch.zeros_like(inputs["attention_mask"])
            for i in range(inputs["input_ids"].shape[0]):
                user_start = inputs["user_start_end"][i, 0] + 1
                user_end = inputs["user_start_end"][i, 1] + 1
                assistant_start = inputs["assistant_start_end"][i, 0] + 1
                assistant_end = inputs["assistant_start_end"][i, 1] + 1
                if self.random_span_e2e:
                    assistant_start = user_end
                if self.random_span_all:
                    user_start = 0
                user_length = user_end - user_start
                assistant_length = assistant_end - assistant_start
                full_length = user_length + assistant_length
                # if self.random_span_zero:
                #     patch_start_offset = 0
                # else:
                #     patch_start_offset = torch.randint(0, full_length, (), generator=self._g, device=self._g.device)
                # max_offset_exclusive = full_length + 1
                # if self.random_span_max_length >= 0:
                #     max_offset_exclusive = min(max_offset_exclusive, patch_start_offset + 1 + self.random_span_max_length)
                # while True:
                #     patch_end_offset = torch.randint(patch_start_offset + 1, max_offset_exclusive, (), generator=self._g, device=self._g.device)
                #     if patch_end_offset - patch_start_offset < full_length:
                #         break
                patch_start_offset, patch_end_offset = self.get_s_t(full_length)
                if patch_start_offset + user_start < user_end:
                    patch_start = user_start + patch_start_offset
                else:
                    patch_start = assistant_start + patch_start_offset - (user_end - user_start)
                if patch_end_offset + user_start < user_end:
                    patch_end = user_start + patch_end_offset
                else:
                    patch_end = assistant_start + patch_end_offset - (user_end - user_start)
                assert patch_end > patch_start, f"{patch_end} <= {patch_start}, {user_start}, {user_end}, {user_length}, {assistant_start}, {assistant_end}, {assistant_length}, {patch_start}, {patch_end}, {full_length}, {patch_start_offset}, {patch_end_offset}."
                if patch_start >= assistant_start:
                    inputs["input_ids_user"][i, 0: assistant_end - user_start] = inputs["input_ids"][i, user_start: assistant_end]
                    inputs["input_ids_user"][i, patch_start - user_start: patch_end - user_start] = self.mask_token_id
                    inputs["labels_user"][i, :] = -100
                    inputs["attention_mask_user"][i, 0: assistant_end - user_start] = 1
                    if self.random_span_mask_recover:
                        inputs["input_ids_assistant"][i, 0: assistant_end - user_start] = inputs["input_ids"][i, user_start: assistant_end]
                        inputs["labels_assistant"][i, :] = -100
                        inputs["attention_mask_assistant"][i, 0: assistant_end - user_start] = 1
                    else:
                        inputs["input_ids_assistant"][i, 0: patch_end - patch_start] = inputs["input_ids"][i, patch_start: patch_end]
                        inputs["labels_assistant"][i, :] = -100
                        inputs["attention_mask_assistant"][i, 0: patch_end - patch_start] = 1
                elif patch_end <= user_end:
                    inputs["input_ids_user"][i, 0: assistant_end - user_start] = inputs["input_ids"][i, user_start: assistant_end]
                    inputs["input_ids_user"][i, patch_start - user_start: patch_end - user_start] = self.mask_token_id
                    inputs["labels_user"][i, :] = -100
                    inputs["attention_mask_user"][i, 0: assistant_end - user_start] = 1
                    if self.random_span_mask_recover:
                        inputs["input_ids_assistant"][i, 0: assistant_end - user_start] = inputs["input_ids"][i, user_start: assistant_end]
                        inputs["labels_assistant"][i, :] = -100
                        inputs["attention_mask_assistant"][i, 0: assistant_end - user_start] = 1
                    else:
                        inputs["input_ids_assistant"][i, 0: patch_end - patch_start] = inputs["input_ids"][i, patch_start: patch_end]
                        inputs["labels_assistant"][i, :] = -100
                        inputs["attention_mask_assistant"][i, 0: patch_end - patch_start] = 1
                else:
                    inputs["input_ids_user"][i, 0: assistant_end - user_start] = inputs["input_ids"][i, user_start: assistant_end]
                    inputs["input_ids_user"][i, patch_start - user_start: user_end - user_start] = self.mask_token_id
                    inputs["input_ids_user"][i, assistant_start - user_start: patch_end - user_start] = self.mask_token_id
                    inputs["labels_user"][i, :] = -100
                    inputs["attention_mask_user"][i, 0: assistant_end - user_start] = 1
                    if self.random_span_mask_recover:
                        inputs["input_ids_assistant"][i, 0: assistant_end - user_start] = inputs["input_ids"][i, user_start: assistant_end]
                        inputs["labels_assistant"][i, :] = -100
                        inputs["attention_mask_assistant"][i, 0: assistant_end - user_start] = 1
                    else:
                        inputs["input_ids_assistant"][i, 0: user_end - patch_start] = inputs["input_ids"][i, patch_start: user_end]
                        inputs["input_ids_assistant"][i, user_end - patch_start: patch_end - assistant_start + user_end - patch_start] = inputs["input_ids"][i, assistant_start: patch_end]
                        inputs["labels_assistant"][i, :] = -100
                        inputs["attention_mask_assistant"][i, 0: patch_end - patch_start] = 1

                if self.debug == 1 and torch.cuda.current_device() == 0:
                    print(f"{i}, {user_start}, {user_end}, {user_length}, {assistant_start}, {assistant_end}, {assistant_length}, {patch_start}, {patch_end}, {full_length}")
                    self.print_indexed_array([self.tokenizer.decode(item) for item in inputs["input_ids"][i, : assistant_end + 2]])
                    self.print_indexed_array(inputs["input_ids_user"][i, : assistant_end - user_start + 2])
                    self.print_indexed_array([self.tokenizer.decode(item) for item in inputs["input_ids_user"][i, : assistant_end - user_start + 2]])
                    self.print_indexed_array(inputs["labels_user"][i, : assistant_end - user_start + 2])
                    self.print_indexed_array(inputs["attention_mask_user"][i, : assistant_end - user_start + 2])
                    self.print_indexed_array(inputs["input_ids_assistant"][i, : patch_end - user_start + 2])
                    self.print_indexed_array([self.tokenizer.decode(item) for item in inputs["input_ids_assistant"][i, : patch_end - user_start + 2]])
                    self.print_indexed_array(inputs["labels_assistant"][i, : patch_end - user_start + 2])
                    self.print_indexed_array(inputs["attention_mask_assistant"][i, : patch_end - user_start + 2])

        if self.additive_mask:
            llm_inputs, skip_jepa = self.build_with_additive_mask(inputs)
        elif self.linear is not None:
            llm_inputs = {
                # "input_ids": torch.tensor(inputs["input_ids"]),
                # "labels": torch.tensor(inputs["labels"]),
                # "attention_mask": torch.tensor(inputs["attention_mask"]),
                "input_ids": inputs["input_ids"],
                "labels": inputs["labels"],
                "attention_mask": inputs["attention_mask"],
            }
            # self.user_start_end = torch.tensor(inputs["user_start_end"])
            # self.assistant_start_end = torch.tensor(inputs["assistant_start_end"])
            self.user_start_end = inputs["user_start_end"]
            self.assistant_start_end = inputs["assistant_start_end"]
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print("input_ids")
                self.print_indexed_array(llm_inputs["input_ids"][0])
                self.print_indexed_array([self.tokenizer.decode(item) for item in llm_inputs["input_ids"][0]])
                print("user_start_end")
                self.print_indexed_array(self.user_start_end[0])
                print("assistant_start_end")
                self.print_indexed_array(self.assistant_start_end[0])
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
        elif self.linear is not None:
            user_hidden_states = None
            assistant_hidden_states = None
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
            'hidden_states': outputs.hidden_states[self.random_span_layer],
            'user_hidden_states': user_hidden_states,
            'assistant_hidden_states': assistant_hidden_states,
        }

    def unwrap(self, m):
        return getattr(m, "module", m)
    
    def get_embeddings(self, hidden_states, user_start_end, assistant_start_end,
                       patch_start_offset: int, patch_end_offset: int):
        """Returns the embeddings of the `before`, `patch`, and `after` spans.

        Args:
            hidden_states: The hidden_state in shape (s, hd), where s is the sequence length,
                and hd is the hidden dimension.
            user_start_end: A pair of integers mark the start and end index of the user message.
                If a message is [start, end), then _start_end[0] = start - 1, and _start_end[1] = end - 1.
            assistant_start_end: Similar to above, but for assistant message.
            patch_start_offset: The offset of the patch start.
            patch_end_offset: The offset of the patch end. The patch is [patch_start, patch_end).
        
        Returns:
            (before_embedding, patch_embedding, after_embedding): The embedding of the before,
                patch, and after span. The before and/or after embedding can be 0 if there is
                0 tokens in them.
        """
        # user message is [user_start, user_end), assistant message is [assistant_start, assistant_end),
        # and patch is [patch_start, patch_end]
        user_start = user_start_end[0] + 1
        user_end = user_start_end[1] + 1
        assistant_start = assistant_start_end[0] + 1
        assistant_end = assistant_start_end[1] + 1
        if patch_start_offset + user_start < user_end:
            patch_start = user_start + patch_start_offset
        else:
            patch_start = assistant_start + patch_start_offset - (user_end - user_start)
        if patch_end_offset + user_start < user_end:
            patch_end = user_start + patch_end_offset
        else:
            patch_end = assistant_start + patch_end_offset - (user_end - user_start)

        user_start_embedding = hidden_states[user_start - 1]
        user_end_embedding = hidden_states[user_end - 1]
        assistant_start_embedding = hidden_states[assistant_start - 1]
        assistant_end_embedding = hidden_states[assistant_end - 1]
        patch_start_embedding = hidden_states[patch_start - 1]
        patch_end_embedding = hidden_states[patch_end - 1]

        if patch_start >= assistant_start:
            before = user_end_embedding - user_start_embedding + patch_start_embedding - assistant_start_embedding
            patch = patch_end_embedding - patch_start_embedding
            after = assistant_end_embedding - patch_end_embedding
        elif patch_end <= user_end:
            before = patch_start_embedding - user_start_embedding
            patch = patch_end_embedding - patch_start_embedding
            after = user_end_embedding - patch_end_embedding + assistant_end_embedding - assistant_start_embedding
        else:
            before = patch_start_embedding - user_start_embedding
            patch = user_end_embedding - patch_start_embedding + patch_end_embedding - assistant_start_embedding
            after = assistant_end_embedding - patch_end_embedding
        
        return before, patch, after

    def get_weights(self, full_length: int, patch_length: int):
        rest_length = full_length - patch_length
        if self.length_adjustment is None:
            return 1.0
        elif self.length_adjustment == "cosine_like":
            return 2.0 * rest_length * patch_length / (rest_length * rest_length + patch_length * patch_length)
        elif self.length_adjustment == "jaccard_like":
            return 1.0 - abs(rest_length - patch_length) / (rest_length + patch_length)
        else:
            assert False, f"Unknown length_adjustment: {self.length_adjustment}."

    def get_curvature(self, hidden_states, start, end_exclusive):
        length = end_exclusive - start
        if length > 1:
            curvature = 0.0
            for i in range(start + 1, end_exclusive):
                prev = hidden_states[i - 1] - hidden_states[i - 2]
                curr = hidden_states[i] - hidden_states[i - 1]
                dot = torch.dot(prev, curr)
                norms = torch.norm(prev) * torch.norm(curr)
                cosine = torch.clamp(dot / norms, -1.0, 1.0)
                angle_rad = torch.acos(cosine)
                if self.curvature_sign:
                    curvature += angle_rad
                else:
                    curvature += torch.abs(angle_rad)
            return curvature, length - 1
        return 0.0, 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with additional regularization terms.
        """
        # Get indeices
        if not self.additive_mask and self.linear is None:
            index_user = self._last_token_index(inputs["input_ids_user"], inputs["labels_user"], inputs["attention_mask_user"])
            index_assistant = self._last_token_index(inputs["input_ids_assistant"], inputs["labels_assistant"], inputs["attention_mask_assistant"])

        # Get all forward pass results
        forward_results = self.forward(model, inputs)

        if self.linear is None:
            first_dim = inputs["input_ids_user"].shape[0]
            if not self.additive_mask:
                if self.debug == 1 and torch.cuda.current_device() == 0:
                    print("=====last tokens=====")
                    print(inputs["input_ids_user"][range(first_dim), index_user])
                    print(inputs["input_ids_user"][range(first_dim), index_user - 1])
                    print(inputs["input_ids_assistant"][range(first_dim), index_assistant])
                    print(inputs["input_ids_assistant"][range(first_dim), index_assistant - 1])
        
        # Extract main language modeling loss
        main_outputs = forward_results['main_outputs']
        lm_loss = main_outputs.loss

        # Compute representation similarity loss
        hidden_states = forward_results['hidden_states']
        user_hidden_states = forward_results['user_hidden_states']
        assistant_hidden_states = forward_results['assistant_hidden_states']
        
        if self.linear is not None:
            weights = torch.ones((hidden_states.shape[0] * self.random_span_times,)).to(hidden_states.device)
            if self.linear == "e2e":
                # Suppose sequence is [start, end), Enc(end - 1) - Enc(start - 1)
                user_start_embedding = hidden_states[range(hidden_states.shape[0]), self.user_start_end[:, 0]]
                user_end_embedding = hidden_states[range(hidden_states.shape[0]), self.user_start_end[:, 1]]
                assistant_start_embedding = hidden_states[range(hidden_states.shape[0]), self.assistant_start_end[:, 0]]
                assistant_end_embedding = hidden_states[range(hidden_states.shape[0]), self.assistant_start_end[:, 1]]

                user_embedding = user_end_embedding - user_start_embedding
                assistant_embedding = assistant_end_embedding - assistant_start_embedding
            elif self.linear == "mean":
                # Suppose sequence is [start, end), mean(Enc(i)) for start <= i < end
                user_embedding = torch.zeros((hidden_states.shape[0], hidden_states.shape[-1])).to(hidden_states.device)
                assistant_embedding = torch.zeros((hidden_states.shape[0], hidden_states.shape[-1])).to(hidden_states.device)
                if self.debug == 1 and torch.cuda.current_device() == 0:
                    print("preprocessing shapes")
                    print(user_embedding.shape, assistant_embedding.shape,
                          hidden_states[0, self.user_start_end[0, 0] + 1: self.user_start_end[0, 1] + 1].mean(dim=0).shape)
                for i in range(hidden_states.shape[0]):
                    user_embedding[i] = hidden_states[i, self.user_start_end[i, 0] + 1: self.user_start_end[i, 1] + 1].mean(dim=0)
                    assistant_embedding[i] = hidden_states[i, self.assistant_start_end[i, 0] + 1: self.assistant_start_end[i, 1] + 1].mean(dim=0)
            elif self.linear == "random_span":
                # Suppose sequence is [start, end), randomly carve out a patch within the sequence.
                user_embedding = torch.zeros((hidden_states.shape[0] * self.random_span_times,
                                              hidden_states.shape[-1])).to(hidden_states.device)
                assistant_embedding = torch.zeros((hidden_states.shape[0] * self.random_span_times,
                                                   hidden_states.shape[-1])).to(hidden_states.device)
                for j in range(hidden_states.shape[0] * self.random_span_times):
                    i = j // self.random_span_times
                    if self.random_span_e2e:
                        self.assistant_start_end[i, 0] = self.user_start_end[i, 1]
                    if self.random_span_all:
                        self.user_start_end[i, 0] = 0
                    user_length = self.user_start_end[i, 1] - self.user_start_end[i, 0]
                    assistant_length = self.assistant_start_end[i, 1] - self.assistant_start_end[i, 0]
                    full_length = user_length + assistant_length
                    # if self.random_span_zero:
                    #     random_span_start = 0
                    # else:
                    #     random_span_start = torch.randint(0, full_length, (), generator=self._g, device=self._g.device)
                    # max_offset_exclusive = full_length + 1
                    # if self.random_span_max_length >= 0:
                    #     max_offset_exclusive = min(max_offset_exclusive, random_span_start + 1 + self.random_span_max_length)
                    # random_span_end = torch.randint(random_span_start + 1, max_offset_exclusive, (), generator=self._g, device=self._g.device)
                    random_span_start, random_span_end = self.get_s_t(full_length)
                    before, patch, after = self.get_embeddings(
                        hidden_states[i], self.user_start_end[i], self.assistant_start_end[i],
                        random_span_start, random_span_end
                    )
                    user_embedding[j] = before + after
                    assistant_embedding[j] = patch
                    weights[j] = self.get_weights(full_length, random_span_end - random_span_start)
                    if self.debug == 1 and torch.cuda.current_device() == 0:
                        print("patch start/end and length adjustment")
                        print(random_span_start, random_span_end, full_length, weights[j], self.length_adjustment)
            elif self.linear == "curvature":
                curvature = torch.zeros((hidden_states.shape[0],)).to(hidden_states.device)
                for i in range(hidden_states.shape[0]):
                    user_curvature, user_count = self.get_curvature(hidden_states[i], self.user_start_end[i, 0] + 1, self.user_start_end[i, 1] + 1)
                    assistant_curvature, assistante_count = self.get_curvature(hidden_states[i], self.assistant_start_end[i, 0] + 1, self.assistant_start_end[i, 1] + 1)
                    if user_count + assistante_count > 0:
                        curvature[i] = (user_curvature + assistant_curvature) / (user_count + assistante_count)
                jepa_loss = torch.mean(curvature)
            else:
                assert False, f"Unknown linear mode: {self.linear}."
            if self.linear_predictor:
                user_embedding = self.unwrap(model).linear_predictor(user_embedding)
            if self.jepa_mse:
                jepa_loss = torch.mean((user_embedding - assistant_embedding) ** 2)
            elif self.linear == "curvature":
                pass
            else:
                assert not self.jepa_l2 and not self.infonce, "jepa_l2 and infonce is not implemented for random_span_mask."
                cosine_similarity = F.cosine_similarity(user_embedding, assistant_embedding, dim=-1)
                assert not self.jepa_l2 and not self.jepa_mse and not self.infonce, "Linear does not support l2, mse, or infonce"
                assert cosine_similarity.shape == weights.shape, f"{cosine_similarity.shape} != {weights.shape}"
                # jepa_loss = 1.0 - torch.mean(cosine_similarity * weights)
                jepa_loss = 1.0 - torch.sum(cosine_similarity * weights) / torch.sum(weights)
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print("start_end_index")
                print(self.user_start_end[:, 0])
                print(self.user_start_end[:, 1])
                print(self.assistant_start_end[:, 0])
                print(self.assistant_start_end[:, 1])
                print("shape")
                print(user_embedding.shape, assistant_embedding.shape)
                print(cosine_similarity.shape)
        elif user_hidden_states is not None:
            if self.additive_mask:
                index_user = self._last_token_user
                index_assistant = self._last_token_assistant
            if self.avg_encoding:
                user_embedding = torch.zeros((first_dim, user_hidden_states.shape[-1])).to(user_hidden_states.device)
                assistant_embedding = torch.zeros((first_dim, assistant_hidden_states.shape[-1])).to(assistant_hidden_states.device)
                for i in range(first_dim):
                    user_embedding[i] = user_hidden_states[i, :index_user[i] + 1].mean(dim=0)
                    assistant_embedding[i] = assistant_hidden_states[i, :index_assistant[i] + 1].mean(dim=0)
            else:
                user_embedding = user_hidden_states[range(first_dim), index_user, :]
                assistant_embedding = assistant_hidden_states[range(first_dim), index_assistant, :]

            if self.linear_predictor:
                user_embedding = self.unwrap(model).linear_predictor(user_embedding)
            
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

        total_loss = self.gamma * lm_loss + self.get_lbd() * jepa_loss

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(lm_loss, self.get_lbd(), torch.mean(cosine_similarity))

        if self.debug == 1 or self.debug == 2:
            exit(0)

        if self.debug == 5 and torch.cuda.current_device() == 0:
            if (self.state.global_step % 10) == 0:
                print(f"llm_loss_10: {lm_loss.float()}, jepa_loss: {jepa_loss.float()}")
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
    parser.add_argument("--linear", type=str, default=None, help="Linear mode. Can be 'e2e', 'mean', 'random_span'.")
    parser.add_argument("--plain_jepa", action="store_true", help="When set, do not apply chat format when compute JEPA loss.")
    parser.add_argument("--linear_predictor", action="store_true", help="Use a inear predictor when set to true.")
    parser.add_argument("--lbd_warmup", action="store_true", help="Linearly warmup lambda when set to True.")
    parser.add_argument("--min_lbd", type=float, default=0.0, help="Min lambda, effective only when --lbd_warmup is set.")
    parser.add_argument("--length_adjustment", type=str, default=None, help="Length adjustment mode. Can be 'consine_like', 'jaccard_like'.")
    parser.add_argument("--random_span_zero", action="store_true", help="Random span always start at 0 when set to True.")
    parser.add_argument("--random_span_e2e", action="store_true", help="When set to True, random span includes the structural tokens in between user and assistant messages.")
    parser.add_argument("--random_span_all", action="store_true", help="When set to True, use all chat message when constructing random span.")
    parser.add_argument("--random_span_mask", action="store_true", help="When set to True, mask the token in the random span. Note this is separated from linear hypothesis.")
    parser.add_argument("--random_span_mask_recover", action="store_true", help="When set to True, recover the masked tokens in the random span. Note this is separated from linear hypothesis.")
    parser.add_argument("--enable_save", action="store_true", help="When set to True, save checkpoints.")
    parser.add_argument("--load_lp", action="store_true", help="When set to True, load linear predictor.")
    parser.add_argument("--random_span_max_length", type=int, default=-1, help="The maximum length of the random span. -1 means infinite.")
    parser.add_argument("--random_span_times", type=int, default=1, help="How many times to compute random_span per sample.")
    parser.add_argument("--random_span_draw_both", action="store_true", help="When set to True, draw both start and end at the same time.")
    parser.add_argument("--random_span_uniform", action="store_true", help="When set to True, draw both start and end uniformly.")
    parser.add_argument("--random_span_layer", type=int, default=-1, help="Which layer of the hidden_state to use as the encoding.")
    parser.add_argument("--constant_lr", action="store_true", help="When set to True, use constant learning rate.")
    parser.add_argument("--curvature_sign", action="store_true", help="When set to True, use signed angle to compute curvature.")
    parser.add_argument("--same_predictor", action="store_true", help="When set to True, use same predictor token.")
    parser.add_argument("--avg_encoding", action="store_true", help="When set to True, use average encoding.")

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
        debug=args.debug, seed=args.finetune_seed, linear_predictor=args.linear_predictor,
        load_lp=args.load_lp)
    
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
            front_pred=args.front_pred, reverse_pred=args.reverse_pred,
            linear=args.linear, plain_jepa=args.plain_jepa, random_span_mask=args.random_span_mask,
            same_predictor=args.same_predictor)
        
        if args.eval_file:
            if torch.cuda.current_device() == 0:
                print(f"Loading evaluation data from {args.eval_file}")
            eval_dataset = load_and_prepare_dataset(
                args.eval_file, tokenizer, args.model_name,
                args.max_length, regular=args.regular,
                debug=args.debug, train_all=args.train_all, plain=args.plain,
                front_pred=args.front_pred, reverse_pred=args.reverse_pred,
                linear=args.linear, plain_jepa=args.plain_jepa, random_span_mask=args.random_span_mask,
                same_predictor=args.same_predictor)
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
            front_pred=args.front_pred, reverse_pred=args.reverse_pred,
            linear=args.linear, plain_jepa=args.plain_jepa, random_span_mask=args.random_span_mask,
            same_predictor=args.same_predictor)

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
    
    # Data collator - don't use padding since we already padded
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=None,  # We already padded to max_length
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
        overwrite_output_dir=True,
        
        # Training parameters
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="constant" if args.constant_lr else "linear",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        
        # Evaluation
        eval_strategy="no",  # "steps" if eval_dataset else "no",
        # eval_steps=eval_steps,
        
        # Saving
        save_strategy="steps" if args.enable_save else "no",
        save_steps=save_steps if args.enable_save else None,
        save_total_limit=args.num_epochs * 4 if args.enable_save else None,

        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.eval_steps,
        
        # Optimization - key changes for stability
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,  # Enable for memory efficiency
        dataloader_drop_last=True,   # Drop last incomplete batch
        
        # Memory optimization
        dataloader_num_workers=0,    # Avoid multiprocessing issues
        
        # Multi-GPU settings - completely disable FSDP
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
        
        # Explicitly disable FSDP and sharding
        fsdp="",
        fsdp_config={},
        
        # Other
        report_to="none",
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
            tokenizer=tokenizer,
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
            tokenizer=tokenizer,
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
            linear=args.linear,
            linear_predictor=args.linear_predictor,
            lbd_warmup=args.lbd_warmup,
            min_lbd=args.min_lbd,
            length_adjustment=args.length_adjustment,
            random_span_zero=args.random_span_zero,
            random_span_e2e=args.random_span_e2e,
            random_span_all=args.random_span_all,
            random_span_mask=args.random_span_mask,
            random_span_mask_recover=args.random_span_mask_recover,
            random_span_max_length=args.random_span_max_length,
            random_span_times=args.random_span_times,
            random_span_draw_both=args.random_span_draw_both,
            random_span_uniform=args.random_span_uniform,
            random_span_layer=args.random_span_layer,
            curvature_sign=args.curvature_sign,
            avg_encoding=args.avg_encoding,
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

    if args.debug == 10:
        print(model.linear_predictor.M.weight)


if __name__ == "__main__":
    main()
