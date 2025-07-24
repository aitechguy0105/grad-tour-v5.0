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
from datasets import load_dataset
from transformers import AutoTokenizer

import scripts.my_constants as my_cst
from scripts.my_logger import get_logger
logger = get_logger(__name__)

def get_model_size(model_id):
 
    url = f"https://huggingface.co/api/models/{model_id}"  
    max_retries = 3
    retry_delay = 5
    b_request_success = False
    for attempt in range(max_retries):  
        try:  
            response = requests.get(url, timeout=10) 
            b_request_success = True 
            break
        except Exception as e:  
            # Log the error  
            logger(f"Attempt {attempt + 1} failed: {e}")  
            if attempt < max_retries - 1:  # Check if there are more retries available  
                logger(f"Retrying in {retry_delay} seconds...")  
                time.sleep(retry_delay)  # Wait before the next attempt  
            else:  
                logger("Max retries reached. request failed.")  # Inform user of failure  
    if b_request_success and response.ok:  
        model_details = response.json()  
        if 'safetensors' in model_details:
            return model_details['safetensors']['total']
            
        else:
            logger("No Safetensor parameter size")
    else:  
        logger(f"Error retrieving model details:, {response.status_code}")
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
            logger(f"Error converting size for {model_id}: {e}")  
            return None
    if 'tiny' in model_id:
        return 1_000_000_000
    logger(f"No size found for {model_id}")  
    return None
def get_model_architecture(model_id):
    
    file_name = "config.json"  # Replace with the specific file you wish to download  
    local_path = os.path.join(model_id, file_name)
    config_dict = None

    if os.path.exists(local_path): 
        with open(local_path, 'r', encoding='utf-8') as file:  
            config_dict = json.load(file)  
    else:
        logger(f'======= No file: {local_path}')
            
    logger(f"== model architecture: {config_dict}")
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
    
def get_merge_lora_info(self, model_id) -> bool:  
        
        try:  
            response = requests.post(  
                f"{my_cst.NODE_URL}/merge-lora",  
                json={
                    "model": model_id
                },  # Use json instead of data for proper content type  
                timeout=10
            )  
            response.raise_for_status()  # Raise an error for bad responses  
            result = response.json()  # Return the JSON response 
            logger.info(f'b_merge_lora: {result}')  
            if 'b_merge_lora'  in result:
                return result['b_merge_lora']
        except requests.exceptions.Timeout:  
            logger.error("Request timed out while getting b_merge_lora.")  
        except requests.exceptions.ConnectionError:  
            logger.error("Connection error occurred while getting b_merge_lora.")  
        except requests.exceptions.HTTPError as http_err:  
            logger.error(f"HTTP error occurred: {http_err}")  
        except requests.exceptions.RequestException as req_err:  
            logger.error(f"An error occurred while getting b_merge_lora: {req_err}")  
        except Exception as req_err:  
            logger.error(f"Unexpected while getting b_merge_lora: {req_err}")

def process_dataset(model_id: str, dataset_path: str):   
    # Load the dataset  
    data_all = load_dataset('json', data_files=dataset_path, split='all')   
    tokenizer = AutoTokenizer.from_pretrained(model_id)  
    logger.info(data_all)  

    def tokenize_text(row):  
        # Create a combined text from the row values  
        combined_text = ' '.join(str(value) for value in row.values() if value is not None)  
        inputs = tokenizer(  
            combined_text,  
            return_tensors="pt",  
            # padding=True,  # Enable padding if required  
            # truncation=True  # Truncate if necessary  
        )  
        token_len = inputs['input_ids'].size(1)  # Get the length of the first item  
        return {"token_len": token_len}  # Ensure this returns a dictionary  

    # Apply the function to the dataset with multiprocessing  
    data_all = data_all.map(tokenize_text, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)  

    logger.info(data_all)  
    logger.info(data_all[0])  

    num_rows = data_all.num_rows  # Total number of rows  
   
    return num_rows

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

def calc_train_and_eval_steps_less_12_5b(model_size: int, sequence_len: int, micro_batch_size: int, eval_batch_size: int, gradient_checkpointing: bool, flash_support: bool = False) -> tuple[float, float]:
    default_model_size = 7241748480
    default_sequence_len = 512
    default_eval_batch_size = 4
    default_micro_batch_size = 4
    default_eval_steps = 4.6
    default_train_steps = 0.46
    ratio_model_size =  1.0
    ratio_flash_support = 0.9
    if flash_support:
        ratio_flash_support = 1.1
    if model_size < default_model_size:
        ratio_model_size = math.pow(default_model_size / model_size, 0.85)
    if model_size > default_model_size:
        ratio_model_size = math.pow(default_model_size / model_size, 1.15) 
    
    ratio_sequence_len = default_sequence_len / sequence_len
    if ratio_sequence_len > 1:
        ratio_sequence_len = math.pow(ratio_sequence_len, 0.85)
    if ratio_sequence_len < 1:
        ratio_sequence_len = math.pow(ratio_sequence_len, 1.15)
    
    ratio_micro_batch_size = 1.0
    if default_micro_batch_size > micro_batch_size:
        ratio_micro_batch_size = math.pow(default_micro_batch_size / micro_batch_size, 0.95)
    if default_micro_batch_size < micro_batch_size:
        ratio_micro_batch_size = math.pow(default_micro_batch_size / micro_batch_size, 1.05)
    
    ratio_eval_batch_size = 1.0
    if default_eval_batch_size > eval_batch_size:
        ratio_eval_batch_size = math.pow(default_eval_batch_size / eval_batch_size, 0.95)
    if default_eval_batch_size < eval_batch_size:
        ratio_eval_batch_size = math.pow(default_eval_batch_size / eval_batch_size, 1.05)
    
    ratio_gradient_checkpointing = 1.0
    if gradient_checkpointing:
        ratio_gradient_checkpointing = 0.85
    print(f'flash: {ratio_flash_support}, eval: {ratio_eval_batch_size}, micro: {ratio_micro_batch_size} chckpointing: {ratio_gradient_checkpointing} model size: {ratio_model_size} seq len: {ratio_sequence_len}')
    train_steps = default_train_steps * ratio_model_size * ratio_sequence_len * ratio_micro_batch_size * ratio_gradient_checkpointing * ratio_flash_support
    eval_steps = default_eval_steps * ratio_model_size * ratio_sequence_len * ratio_eval_batch_size * ratio_gradient_checkpointing * ratio_flash_support
    return train_steps, eval_steps
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
    else:
        return calc_train_and_eval_steps_from_info(
            default_train_steps_per_sec = 0.45,
            default_eval_steps_per_sec = 4.6,
            default_model_size = 7241748480,
            current_model_size = model_size,
            default_flash_support = True,
            current_flash_support = flash_support,
            default_sequence_len = 512, 
            current_sequence_len = sequence_len,
            default_micro_batch_size = 4,
            current_micro_batch_size = train_micro_batch_size,
            default_eval_batch_size = 4,
            current_eval_batch_size = eval_batch_size, 
            default_gradient_checkpointing = False, 
            current_gradient_checkpointing = gradient_checkpointing, 
        )