from datetime import datetime
import requests
import re
import time
import os
import json
import math
import multiprocessing
import numpy as np
import torch
import psutil
import gc
import fnmatch
from datasets import load_dataset
from transformers import AutoTokenizer
import shutil
import scripts.my_constants as my_cst
from scripts.my_logger import get_logger
logger = get_logger(__name__)
custom_models = {
    # 2048, 2, 2, False, 8.7
    'fxmarty/really-tiny-falcon-testing': {
        'train_steps_per_sec': 8.7,
        'eval_steps_per_sec': 88,
        'model_size': 1000000000,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 1024, 2, 2, False, 12
    'fxmarty/tiny-llama-fast-tokenizer': {
        'train_steps_per_sec': 12,
        'eval_steps_per_sec': 121,
        'model_size': 1000000000,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 2048, 4, 4, False, 6.4
    'katuni4ka/tiny-random-olmo-hf': {
        'train_steps_per_sec': 6.4,
        'eval_steps_per_sec': 65,
        'model_size': 6537728,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    'bigscience/bloom-560m': {
        'train_steps_per_sec': 0.28,
        'eval_steps_per_sec': 2.8,
        'model_size': 559214592,
        'flash_support': False,
        'sequence_len': 4096,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    'unsloth/SmolLM-135M': {
        'train_steps_per_sec': 1.0,
        'eval_steps_per_sec': 11.1,
        'model_size': 134515008,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    'unsloth/SmolLM2-360M': {
        'train_steps_per_sec': 1.3,
        'eval_steps_per_sec': 14,
        'model_size': 360000000,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    'unsloth/SmolLM-360M': {
        'train_steps_per_sec': 1.2,
        'eval_steps_per_sec': 12,
        'model_size': 361821120,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    
    'unsloth/SmolLM2-1.7B': {
        'train_steps_per_sec': 1.85,
        'eval_steps_per_sec': 19,
        'model_size': 1711378432,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 1024, 8, 8, False, 4.5
    'peft-internal-testing/tiny-dummy-qwen2': {
        'train_steps_per_sec': 4.5,
        'eval_steps_per_sec': 45,
        'model_size': 1217480,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 8,
        'eval_batch_size': 8,
        'gradient_checkpointing': False
    },



    # 2048, 2, 2, False, 1.4
    'unsloth/tinyllama-chat': {
        'train_steps_per_sec': 1.4,
        'eval_steps_per_sec': 15,
        'model_size': 1100048384,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    'unsloth/tinyllama': {
        'train_steps_per_sec': 2.2,
        'eval_steps_per_sec': 25,
        'model_size': 1100048384,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 1024, 2, 2, True, 0.5
    'unsloth/llama-3-8b': {
        'train_steps_per_sec': 0.5,
        'eval_steps_per_sec': 5.1,
        'model_size': 8030261248,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, True, 0.51
    'unsloth/Meta-Llama-3.1-8B': {
        'train_steps_per_sec': 0.51,
        'eval_steps_per_sec': 5.2,
        'model_size': 8030261248,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    'unsloth/Llama-3.1-Storm-8B': {
        'train_steps_per_sec': 0.51,
        'eval_steps_per_sec': 5.2,
        'model_size': 8030261248,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, False, 0.81
    'unsloth/llama-2-7b-chat': {
        'train_steps_per_sec': 0.81,
        'eval_steps_per_sec': 8.2,
        'model_size': 6738415616,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    'unsloth/codellama-7b': {
        'train_steps_per_sec': 0.75,
        'eval_steps_per_sec': 7.5,
        'model_size': 6738546688,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },


    # 512, 4, 4, False, 0.95 3821079552
    'unsloth/Phi-3-mini-4k-instruct':{
        'train_steps_per_sec': 0.95,
        'eval_steps_per_sec': 9.5,
        'model_size': 3821079552,
        'flash_support': True,
        'sequence_len': 512,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    # 4096, 1, 1, True, 0.138 13960238080
    'unsloth/Phi-3-medium-4k-instruct':{
        'train_steps_per_sec': 0.138,
        'eval_steps_per_sec': 1.4,
        'model_size': 13960238080,
        'flash_support': True,
        'sequence_len': 4096,
        'micro_batch_size': 1,
        'eval_batch_size': 1,
        'gradient_checkpointing': True
    },

    # 4096, 1, 1, True, 0.35
    'unsloth/mistral-7b-v0.2':{
        'train_steps_per_sec': 0.35,
        'eval_steps_per_sec': 3.5,
        'model_size': 7241732096,
        'flash_support': True,
        'sequence_len': 4096,
        'micro_batch_size': 1,
        'eval_batch_size': 1,
        'gradient_checkpointing': True
    },
    # 4096, 1, 1, True, 0.28
    'unsloth/mistral-7b-v0.3':{
        'train_steps_per_sec': 0.28,
        'eval_steps_per_sec': 2.8,
        'model_size': 7248023552,
        'flash_support': True,
        'sequence_len': 4096,
        'micro_batch_size': 1,
        'eval_batch_size': 1,
        'gradient_checkpointing': True
    },
    # 2048, 2, 2, True, 0.32
    'unsloth/mistral-7b':{
        'train_steps_per_sec': 0.32,
        'eval_steps_per_sec': 3.2,
        'model_size': 7241732096,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },

    # 4096, 2, 2, True, 0.36
    'unsloth/codegemma-2b':{
        'train_steps_per_sec': 0.36,
        'eval_steps_per_sec': 3.6,
        'model_size': 2506172416,
        'flash_support': True,
        'sequence_len': 4096,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
 
    # 4096, 1, 1, True, 0.24
    'unsloth/gemma-7b-it':{
        'train_steps_per_sec': 0.24,
        'eval_steps_per_sec': 2.5,
        'model_size': 8537680896,
        'flash_support': True,
        'sequence_len': 4096,
        'micro_batch_size': 1,
        'eval_batch_size': 1,
        'gradient_checkpointing': True
    },

    # 2048, 2, 2, True, 0.14 
    'unsloth/Qwen2.5-14B-Instruct':{
        'train_steps_per_sec': 0.14,
        'eval_steps_per_sec': 1.5,
        'model_size': 14770033664,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 2048, 2, 2, True, 0.14 
    'unsloth/Qwen2.5-Coder-1.5B-Instruct':{
        'train_steps_per_sec': 0.91,
        'eval_steps_per_sec': 9.2,
        'model_size': 1543714304,
        'flash_support': True,
        'sequence_len': 512,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    'unsloth/Qwen2.5-Math-1.5B-Instruct':{
        'train_steps_per_sec': 1.0,
        'eval_steps_per_sec': 11,
        'model_size': 1543714304,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },


    # 512, 2, 2, False, 1.3
    'NousResearch/Hermes-2-Pro-Mistral-7B':{
        'train_steps_per_sec': 1.3,
        'eval_steps_per_sec': 13,
        'model_size': 7241994240,
        'flash_support': True,
        'sequence_len': 512,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 2048, 2, 2, True, 0.34
    'NousResearch/Nous-Capybara-7B-V1.9':{
        'train_steps_per_sec': 0.34,
        'eval_steps_per_sec': 3.5,
        'model_size': 7_000_000_000,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 512, 2, 2, False, 1.2
    'NousResearch/CodeLlama-7b-hf':{
        'train_steps_per_sec': 1.2,
        'eval_steps_per_sec': 12.5,
        'model_size': 7_000_000_000,
        'flash_support': False,
        'sequence_len': 512,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    'NousResearch/Yarn-Mistral-7b-64k':{
        'train_steps_per_sec': 1.2,
        'eval_steps_per_sec': 12.5,
        'model_size': 7_000_000_000,
        'flash_support': False,
        'sequence_len': 512,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 512, 2, 2, False, 1.2
    'NousResearch/Yarn-Llama-2-7b-128k':{
        'train_steps_per_sec': 1.35,
        'eval_steps_per_sec': 14,
        'model_size': 7_000_000_000,
        'flash_support': False,
        'sequence_len': 512,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    'NousResearch/CodeLlama-13b-hf':{
        'train_steps_per_sec': 0.08,
        'eval_steps_per_sec': 0.85,
        'model_size': 13000000000,
        'flash_support': False,
        'sequence_len': 4096,
        'micro_batch_size': 1,
        'eval_batch_size': 1,
        'gradient_checkpointing': True
    },


    # 2048,4,4,False, 4.3
    'EleutherAI/pythia-70m-deduped':{
        'train_steps_per_sec': 4.3,
        'eval_steps_per_sec': 45,
        'model_size': 95592496,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },

    # 512, 4,4,False, 10.4
    'HuggingFaceH4/tiny-random-LlamaForCausalLM':{
        'train_steps_per_sec': 10.4,
        'eval_steps_per_sec': 104,
        'model_size': 1_000_000_000,
        'flash_support': True,
        'sequence_len': 512,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    # 2048, 2, 2, True, 0.32
    'HuggingFaceH4/zephyr-7b-beta':{
        'train_steps_per_sec': 0.32,
        'eval_steps_per_sec': 3.2,
        'model_size': 7241732096,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },


    # 1024, 8, 8, False, 1.5
    'facebook/opt-350m':{
        'train_steps_per_sec': 1.5,
        'eval_steps_per_sec': 15,
        'model_size': 350000000,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 8,
        'eval_batch_size': 8,
        'gradient_checkpointing': False
    },
    # 2048, 4, 4, False, 5.0
    'JackFram/llama-68m':{
        'train_steps_per_sec': 5.1,
        'eval_steps_per_sec': 51,
        'model_size': 68000000,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },

    
    # 512, 2, 2, False, 1.25
    'codellama/CodeLlama-7b-hf':{
        'train_steps_per_sec': 1.25,
        'eval_steps_per_sec': 12.5,
        'model_size': 6738546688,
        'flash_support': False,
        'sequence_len': 512,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 1024, 2, 2, True, 0.52
    'jhflow/mistral7b-lora-multi-turn-v2':{
        'train_steps_per_sec': 0.52,
        'eval_steps_per_sec': 5.3,
        'model_size': 7_000_000_000,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, True, 0.52
    'The-matt/llama2_ko-7b_distinctive-snowflake-182_1060':{
        'train_steps_per_sec': 0.21,
        'eval_steps_per_sec': 2.3,
        'model_size': 7_000_000_000,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, True, 0.52
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4':{
        'train_steps_per_sec': 0.28,
        'eval_steps_per_sec': 2.8,
        'model_size': 7_000_000_000,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, False, 0.82
    'defog/sqlcoder-7b-2':{
        'train_steps_per_sec': 0.82,
        'eval_steps_per_sec': 8.3,
        'model_size': 6738546688,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    'Artples/L-MChat-7b':{
        'train_steps_per_sec': 0.5,
        'eval_steps_per_sec': 5.1,
        'model_size': 7241748480,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, False, 0.82
    'furiosa-ai/mlperf-gpt-j-6b':{
        'train_steps_per_sec': 0.82,
        'eval_steps_per_sec': 8.3,
        'model_size': 6_000_000_000,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 1024, 2, 2, True, 0.53
    'Vikhrmodels/Vikhr-7B-instruct_0.4':{
        'train_steps_per_sec': 0.53,
        'eval_steps_per_sec': 5.3,
        'model_size': 7627567104,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    
    # 1024, 2, 2, False, 0.7
    'Qwen/Qwen2.5-7B-Instruct':{
        'train_steps_per_sec': 0.8,
        'eval_steps_per_sec': 9.1,
        'model_size': 7615616512,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    # 1024, 2, 2, False, 0.7
    'Qwen/Qwen2.5-Math-7B-Instruct':{
        'train_steps_per_sec': 0.8,
        'eval_steps_per_sec': 9.2,
        'model_size': 7615616512,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    
    'Qwen/Qwen2-0.5B-Instruct':{
        'train_steps_per_sec': 1.05,
        'eval_steps_per_sec': 10.9,
        'model_size': 494032768,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    'Qwen/Qwen2.5-0.5B':{
        'train_steps_per_sec': 1.95,
        'eval_steps_per_sec': 19,
        'model_size': 494032768,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },
    'Qwen/Qwen2-1.5B-Instruct':{
        'train_steps_per_sec': 1.5,
        'eval_steps_per_sec': 15,
        'model_size': 1543714304,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': False
    },
    'Qwen/Qwen1.5-0.5B':{
        'train_steps_per_sec': 1.5,
        'eval_steps_per_sec': 15,
        'model_size': 619570176,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_checkpointing': False
    },

    # 256, 8, 8, False, 1.2
    'heegyu/WizardVicuna-open-llama-3b-v2':{
        'train_steps_per_sec': 1.2,
        'eval_steps_per_sec': 12.5,
        'model_size': 3426474900,
        'flash_support': True,
        'sequence_len': 256,
        'micro_batch_size': 8,
        'eval_batch_size': 8,
        'gradient_checkpointing': False
    },
    # 2048, 2, 2, True, 0.25
    'teknium/OpenHermes-2.5-Mistral-7B':{
        'train_steps_per_sec': 0.25,
        'eval_steps_per_sec': 2.5,
        'model_size': 7241748480,
        'flash_support': True,
        'sequence_len': 2048,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, True, 0.75
    'tiiuae/falcon-7b':{
        'train_steps_per_sec': 0.75,
        'eval_steps_per_sec': 7.5,
        'model_size': 7217189760,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 1024, 2, 2, True, 0.75
    'elyza/Llama-3-ELYZA-JP-8B':{
        'train_steps_per_sec': 0.6,
        'eval_steps_per_sec': 6.1,
        'model_size': 8030261248,
        'flash_support': True,
        'sequence_len': 1024,
        'micro_batch_size': 2,
        'eval_batch_size': 2,
        'gradient_checkpointing': True
    },
    # 'upstage/SOLAR-10.7B-Instruct-v1.0':{
    #     'train_steps_per_sec': 0.5,
    #     'eval_steps_per_sec': 5.5,
    #     'model_size': 10731524096,
    #     'flash_support': True,
    #     'sequence_len': 512,
    #     'micro_batch_size': 2,
    #     'eval_batch_size': 2,
    #     'gradient_checkpointing': False
    # }
}

def get_model_size(model_id):
    if model_id in my_cst.MODEL_SIZE_DICT:
        return my_cst.MODEL_SIZE_DICT[model_id]
    # url = f"https://huggingface.co/api/models/{model_id}"  
    # max_retries = 3
    # retry_delay = 5
    # b_request_success = False
    # for attempt in range(max_retries):  
    #     try:  
    #         response = requests.get(url, timeout=10) 
    #         b_request_success = True 
    #         break
    #     except Exception as e:  
    #         # Log the error  
    #         logger(f"Attempt {attempt + 1} failed: {e}")  
    #         if attempt < max_retries - 1:  # Check if there are more retries available  
    #             logger(f"Retrying in {retry_delay} seconds...")  
    #             time.sleep(retry_delay)  # Wait before the next attempt  
    #         else:  
    #             logger("Max retries reached. request failed.")  # Inform user of failure  
    # if b_request_success and response.ok:  
    #     model_details = response.json()  
    #     if 'safetensors' in model_details:
    #         return model_details['safetensors']['total']
            
    #     else:
    #         logger("No Safetensor parameter size")
    # else:  
    #     logger(f"Error retrieving model details:, {response.status_code}")
    pattern = r"(\d+(\.\d+)?)([mMbB])"
    match = re.search(pattern, model_id)  
    if match:  
        try:  
            # Extract the number and unit  
            size = float(match.group(1))  # Get the numeric part  
            unit = match.group(3).lower()  # Get the unit part (convert to lowercase)  
            
            # Convert based on the unit  
            if unit == 'b':  
                size *= 1e9  # Billion  
            elif unit == 'm':  
                size *= 1e6  # Million 
            return size
                        
        except ValueError as e:  
            logger.info(f"Error converting size for {model_id}: {e}")  
            return None
    if 'tiny' in model_id:
        return 1_000_000_000
    logger.info(f"No size found for {model_id}")  
    return None
def get_model_architecture(model_id):
    
    file_name = "config.json"  # Replace with the specific file you wish to download  
    local_path = os.path.join(model_id, file_name)
    config_dict = None

    if os.path.exists(local_path): 
        with open(local_path, 'r', encoding='utf-8') as file:  
            config_dict = json.load(file)  
    else:
        logger.info(f'======= No file: {local_path}')
            
    logger.info(f"== model architecture: {config_dict}")
    return config_dict
def process_dataset(model_id: str, dataset_path: str):   
    # Load the dataset  
    data_all = load_dataset('json', data_files=dataset_path, split='all')   

    if len(data_all) > 0:
        logger.info(data_all[0])
    else:
        logger.warning("Dataset is empty!")
        return 0
    num_rows = len(data_all)  # Total number of rows  
    
    logger.info(f"Total Rows: {num_rows}")  
    return num_rows


    # logger.info(data_all)
    # if len(data_all) > 0:
    #     logger.info(data_all[0])
    # else:
    #     logger.warning("Dataset is empty!")
    #     return 0

    # num_rows = len(data_all)  # HuggingFace Dataset objects are list-like

    # logger.info(f"Total Rows: {num_rows}")
    # return num_rows
def send_request_post_sync_with_retry(url, data, timeout=10):  
    """Sends a synchronous POST request and returns the JSON response."""  
    max_attempts = 3  
    delay = 5  
    
    for attempt in range(max_attempts):  
        try:  
            response = requests.post(url=url, json=data, timeout=timeout)  # Assuming model_dump() serializes correctly  
            response.raise_for_status()  # Check for HTTP errors  
            logger.info(f'Response Status Code: {response.status_code}, JSON Response: {response.json()}')  
            return response.json()  # Return the response JSON for further processing  
        except requests.exceptions.RequestException as e:  
            logger.error(f"Error during attempt {attempt + 1}/{max_attempts}, retrying after {delay} seconds. Error: {e}")  
            if attempt == max_attempts - 1:  
                logger.error("Max attempts reached. Returning None.")  
                return None  
            time.sleep(delay) 
def cleanup_gpu_proc(proc):
    """Completely clean up vLLM server process and GPU memory"""
    try:
        # 1. Terminate the process tree
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        gone, still_alive = psutil.wait_procs(children, timeout=10)
        for p in still_alive:
            p.kill()
        parent.terminate()
        try:
            parent.wait(timeout=15)
        except psutil.TimeoutExpired:
            logger.warning(f"Parent process {proc.pid} did not terminate in time; killing it.")
            parent.kill()
            parent.wait(timeout=10)
        children = parent.children(recursive=True)
        for child in children:
            if child.is_running():
                child.kill()
        # 2. Force Python cleanup
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. Verify cleanup
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
            
        logger.info("processs and GPU memory fully cleaned up")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clean up GPU: {str(e)}")
        return False
    
def get_merge_lora_info(model_id) -> bool:  
        
    model_size = get_model_size(model_id)
    if model_size is None:
        return True
    if model_size > my_cst.BIG_MODEL_SIZE:
        return False
    return True

def get_batch_checkpointing_info(model_size: int, sequence_len: int) -> tuple[int, int, bool]:
    if model_size >= 13_000_000_000:
        gradient_checkpointing = True
        micro_batch_size = 2
        eval_batch_size = 2
        if sequence_len >= 1024:
            gradient_checkpointing = True
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len >= 4096:
            gradient_checkpointing = True
            micro_batch_size = 1
            eval_batch_size = 1

        if sequence_len <=256:
            gradient_checkpointing = False
            micro_batch_size = 2
            eval_batch_size = 2

    elif model_size >= 7_000_000_000:
        gradient_checkpointing = False
        micro_batch_size = 2
        eval_batch_size = 2
        if sequence_len >= 512:
            gradient_checkpointing = False
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len >= 1024:
            gradient_checkpointing = True
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len >= 4096:
            gradient_checkpointing = True
            micro_batch_size = 1
            eval_batch_size = 1
        if sequence_len <=128:
            gradient_checkpointing = False
            micro_batch_size = 4
            eval_batch_size = 4


    elif model_size >= 5_000_000_000:
        gradient_checkpointing = False
        micro_batch_size = 4
        eval_batch_size = 4
        if sequence_len >= 512:
            gradient_checkpointing = False
            micro_batch_size = 4
            eval_batch_size = 4
        if sequence_len >= 1024:
            gradient_checkpointing = False
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len >= 2048:
            gradient_checkpointing = True
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len >= 4096:
            gradient_checkpointing = True
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len <=128:
            gradient_checkpointing = False
            micro_batch_size = 8
            eval_batch_size = 8
    elif model_size >= 3_000_000_000:
        gradient_checkpointing = False
        micro_batch_size = 4
        eval_batch_size = 4
        if sequence_len >= 1024:
            gradient_checkpointing = False        
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len >= 4096:
            gradient_checkpointing = True
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len <=256:
            gradient_checkpointing = False        
            micro_batch_size = 8
            eval_batch_size = 8
    elif model_size >= 1_000_000_000:
        gradient_checkpointing = False
        micro_batch_size = 4
        eval_batch_size = 4
        if sequence_len >= 1024:
            gradient_checkpointing = False
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len >= 4096:
            gradient_checkpointing = True
            micro_batch_size = 2
            eval_batch_size = 2
        if sequence_len <=256:
            gradient_checkpointing = False
            micro_batch_size = 8
            eval_batch_size = 8
    else:
        gradient_checkpointing = False
        micro_batch_size = 8
        eval_batch_size = 8
        if sequence_len >= 2048:
            gradient_checkpointing = False
            micro_batch_size = 4
            eval_batch_size = 4
        if sequence_len >= 4096:
            gradient_checkpointing = True
            micro_batch_size = 2
            eval_batch_size = 2

    return micro_batch_size, eval_batch_size, gradient_checkpointing

def jaccard_similarity(str1, str2):
    
    set1 = set(re.split(r'[/-]', str1.lower()))
    set2 = set(re.split(r'[/-]', str2.lower()))
    # print(set1, set2)
    # print(len(set1 & set2))
    # print(len(set1 | set2))
    return len(set1 & set2) / len(set1 | set2)

def calc_train_and_eval_steps_from_similar_size_model(
    current_model_id: str,
    current_model_size: int,
    current_flash_support: bool,
    current_sequence_len: int,
    current_micro_batch_size: int,
    current_eval_batch_size: int, 
    current_gradient_checkpointing: bool, 
) -> tuple[float, float]:
    model_size_score = {}
    for model_id, model_info in custom_models.items():
        model_size_score[model_id] = abs(model_info['model_size'] - current_model_size)
    sorted_models = dict(sorted(model_size_score.items(), key=lambda item: item[1]))
    first_key, first_value = next(iter(sorted_models.items()))
    # Find all models with that same score
    same_size_model_ids = [
        key for key, value in sorted_models.items() if value == first_value
    ]
    current_username = current_model_id.split('/')[0]
    current_reponame = current_model_id.split('/')[1]


    reference_model_id = first_key
    logger.info(f'reference_model_id: {reference_model_id}')
    if len(same_size_model_ids) > 1:
            
        model_text_scores = []

        for model_id in same_size_model_ids:
            score1 = jaccard_similarity(model_id, current_model_id)
            score2 = jaccard_similarity(model_id, current_reponame)
            total_score = score1 + score2
            model_text_scores.append((model_id, total_score))

            # Sort by total_score in descending order (most similar first)
            sorted_model_ids = [model_id for model_id, score in sorted(model_text_scores, key=lambda x: x[1], reverse=True)]
            
            reference_model_id = sorted_model_ids[0]
            logger.info(f'reference_model_id changed with text smilarity: {reference_model_id}')


    default_flash_support = custom_models[reference_model_id]['flash_support']
    default_model_size = custom_models[reference_model_id]['model_size']
    default_sequence_len = custom_models[reference_model_id]['sequence_len']
    default_micro_batch_size = custom_models[reference_model_id]['micro_batch_size']
    default_eval_batch_size = custom_models[reference_model_id]['eval_batch_size']
    default_gradient_checkpointing = custom_models[reference_model_id]['gradient_checkpointing']
    default_train_steps_per_sec = custom_models[reference_model_id]['train_steps_per_sec']
    default_eval_steps_per_sec = custom_models[reference_model_id]['eval_steps_per_sec']
    ratio_flash_support = 1
    if default_flash_support != current_flash_support:
        if current_flash_support:
            ratio_flash_support = 1.15
        else:
            ratio_flash_support = 0.85
    ratio_model_size = default_model_size / current_model_size
    if ratio_model_size > 1:
        ratio_model_size = math.pow(ratio_model_size, 0.9)
    else:
        ratio_model_size = math.pow(ratio_model_size, 1.1)

    ratio_sequence_len = default_sequence_len / current_sequence_len
    if ratio_sequence_len > 1:
        ratio_sequence_len = math.pow(ratio_sequence_len, 0.85)
    if ratio_sequence_len < 1:
        ratio_sequence_len = math.pow(ratio_sequence_len, 1.15)
    
    ratio_micro_batch_size = default_micro_batch_size / current_micro_batch_size
    if ratio_micro_batch_size > 1:
        ratio_micro_batch_size = math.pow(ratio_micro_batch_size, 0.95)
    if ratio_micro_batch_size < 1:
        ratio_micro_batch_size = math.pow(ratio_micro_batch_size, 1.05)
    
    ratio_eval_batch_size = default_eval_batch_size / current_eval_batch_size
    if ratio_eval_batch_size > 1:
        ratio_eval_batch_size = math.pow(ratio_eval_batch_size, 0.95)
    if ratio_eval_batch_size < 1:
        ratio_eval_batch_size = math.pow(ratio_eval_batch_size, 1.05)
    ratio_gradient_checkpointing = 1.0
    if default_gradient_checkpointing != current_gradient_checkpointing:
        if current_gradient_checkpointing == True:
            ratio_gradient_checkpointing = 0.8
        else: 
            ratio_gradient_checkpointing = 1.2
    estimated_train_steps_per_sec = default_train_steps_per_sec * ratio_sequence_len * ratio_micro_batch_size * ratio_gradient_checkpointing * ratio_model_size * ratio_flash_support
    estimated_eval_steps_per_sec = default_eval_steps_per_sec * ratio_sequence_len * ratio_eval_batch_size * ratio_gradient_checkpointing * ratio_model_size * ratio_flash_support
    return estimated_train_steps_per_sec, estimated_eval_steps_per_sec

def calc_train_and_eval_steps_from_info(
    default_train_steps_per_sec: float,
    default_eval_steps_per_sec: float,
    default_model_size: int,
    current_model_size: int,
    default_flash_support: bool,
    current_flash_support: bool,
    default_sequence_len: int, 
    current_sequence_len: int,
    default_micro_batch_size: int,
    current_micro_batch_size: int,
    default_eval_batch_size: int,
    current_eval_batch_size: int, 
    default_gradient_checkpointing: bool, 
    current_gradient_checkpointing: bool, 
) -> tuple[float, float]:
    
    ratio_flash_support = 1
    if default_flash_support != current_flash_support:
        if current_flash_support:
            ratio_flash_support = 1.15
        else:
            ratio_flash_support = 0.85
    ratio_model_size = default_model_size / current_model_size
    if ratio_model_size > 1:
        ratio_model_size = math.pow(ratio_model_size, 0.9)
    else:
        ratio_model_size = math.pow(ratio_model_size, 1.1)

    ratio_sequence_len = default_sequence_len / current_sequence_len
    if ratio_sequence_len > 1:
        ratio_sequence_len = math.pow(ratio_sequence_len, 0.95)
    if ratio_sequence_len < 1:
        ratio_sequence_len = math.pow(ratio_sequence_len, 1.05)
    
    ratio_micro_batch_size = default_micro_batch_size / current_micro_batch_size
    if ratio_micro_batch_size > 1:
        ratio_micro_batch_size = math.pow(ratio_micro_batch_size, 0.95)
    if ratio_micro_batch_size < 1:
        ratio_micro_batch_size = math.pow(ratio_micro_batch_size, 1.05)
    
    ratio_eval_batch_size = default_eval_batch_size / current_eval_batch_size
    if ratio_eval_batch_size > 1:
        ratio_eval_batch_size = math.pow(ratio_eval_batch_size, 0.95)
    if ratio_eval_batch_size < 1:
        ratio_eval_batch_size = math.pow(ratio_eval_batch_size, 1.05)
    ratio_gradient_checkpointing = 1.0
    if default_gradient_checkpointing != current_gradient_checkpointing:
        if current_gradient_checkpointing == True:
            ratio_gradient_checkpointing = 0.8
        else: 
            ratio_gradient_checkpointing = 1.2
    estimated_train_steps_per_sec = default_train_steps_per_sec * ratio_sequence_len * ratio_micro_batch_size * ratio_gradient_checkpointing * ratio_model_size * ratio_flash_support
    estimated_eval_steps_per_sec = default_eval_steps_per_sec * ratio_sequence_len * ratio_eval_batch_size * ratio_gradient_checkpointing * ratio_model_size * ratio_flash_support
    return estimated_train_steps_per_sec, estimated_eval_steps_per_sec


def get_train_and_eval_steps_from_data(model_id: str, model_size: int, flash_support: bool, sequence_len: int, train_micro_batch_size: int, eval_batch_size: int, gradient_checkpointing: bool, num_gpus: int, adapter: str = 'lora'):

    username = model_id.split('/')[0]
    reponame = model_id.split('/')[1]
    if adapter == 'qlora':
        return calc_train_and_eval_steps_from_info(
            default_train_steps_per_sec = 0.05,
            default_eval_steps_per_sec = 0.605,
            default_model_size = 70553706496,
            current_model_size = model_size,
            default_flash_support = True,
            current_flash_support = flash_support,
            default_sequence_len = 1024, 
            current_sequence_len = sequence_len,
            default_micro_batch_size = 4,
            current_micro_batch_size = train_micro_batch_size,
            default_eval_batch_size = 4,
            current_eval_batch_size = eval_batch_size, 
            default_gradient_checkpointing = True, 
            current_gradient_checkpointing = gradient_checkpointing, 
        )
    elif model_id in custom_models:
        return calc_train_and_eval_steps_from_info(
            default_train_steps_per_sec = custom_models[model_id]['train_steps_per_sec'],
            default_eval_steps_per_sec = custom_models[model_id]['eval_steps_per_sec'],
            default_model_size = custom_models[model_id]['model_size'],
            # model size should be same, the model size recorded and current input model size might bill be diff.
            current_model_size = custom_models[model_id]['model_size'],
            # current_model_size = model_size,
            default_flash_support = custom_models[model_id]['flash_support'],
            current_flash_support = flash_support,
            default_sequence_len = custom_models[model_id]['sequence_len'],
            current_sequence_len = sequence_len,
            default_micro_batch_size = custom_models[model_id]['micro_batch_size'],
            current_micro_batch_size = train_micro_batch_size,
            default_eval_batch_size = custom_models[model_id]['eval_batch_size'],
            current_eval_batch_size = eval_batch_size, 
            default_gradient_checkpointing = custom_models[model_id]['gradient_checkpointing'],
            current_gradient_checkpointing = gradient_checkpointing, 
        )
    else:
        return calc_train_and_eval_steps_from_similar_size_model(
            current_model_id=model_id,
            current_model_size = model_size,
            current_flash_support = flash_support,
            current_sequence_len = sequence_len,
            current_micro_batch_size = train_micro_batch_size,
            current_eval_batch_size = eval_batch_size, 
            current_gradient_checkpointing = gradient_checkpointing, 
        )
def remove_contents_in_dir(src_dir, keep_files=[]):
                
    # Iterate over all items in the directory
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)

        # Remove everything except the 'merged' directory
        if item not in keep_files:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

def copy_contents_to_dir(src_dir, dest_dir, ignore_patterns=None):
    if ignore_patterns is None:
        ignore_patterns = []

    def _should_skip(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in ignore_patterns)
    
    for item in os.listdir(src_dir):
        if _should_skip(item):
            continue
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def get_liger_info(model_conf):
    if model_conf.get('model_type', None) in [
        'deepseek_v2',
        'granite', 
        'jamba', 
        'llama', 
        'mistral', 
        'mixtral', 
        'mllama',
        'mllama_text_model',
        'olmo2',
        'phi3',
        'qwen2',
        'qwen2_5_vl',
        'qwen2_vl',
        ]:
        return True
    return False
def get_flash_attention_info(model_id, model_conf):
    flash_attention = True
    model_architecture = None
    if model_conf is not None and isinstance(model_conf, dict):  
        # Check if 'architectures' key exists and is a list  
        architectures = model_conf.get('architectures')  
        if architectures and isinstance(architectures, list):  
            model_architecture = architectures[0]  # Access the first architecture  
            # Check if 'rope_scaling' exists and is also a dictionary  
            rope_scaling = model_conf.get('rope_scaling')  
            if rope_scaling and isinstance(rope_scaling, dict):  
                rope_type = rope_scaling.get('type')  
                if rope_type == 'yarn':  
                    flash_attention = False  
                    
            flash_not_support_architectures = ['FalconForCausalLM', 'CodeGenForCausalLM', 'BloomForCausalLM', 'GPTNeoForCausalLM']
            flash_support_archtectures = ['Qwen2ForCausalLM', 'GPTNeoXForCausalLM']
            if model_architecture in flash_not_support_architectures:            
                flash_attention = False
            elif model_architecture in flash_support_archtectures:   
                flash_attention = True
    else:  
        # Handle the case where model_conf is None or not structured as expected  
        logger.error("Error: model_conf is not properly defined.")
        
    if model_id in my_cst.FLASH_UNSUPPORT_MODEL_LIST:
        flash_attention = False
        logger.info(f"{model_id}: flash not support model")
    return flash_attention