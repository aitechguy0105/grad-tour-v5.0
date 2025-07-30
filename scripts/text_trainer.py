#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import pathlib

import yaml
from transformers import AutoTokenizer


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import TaskType
from miner.logic.job_handler import create_reward_funcs_file

from datetime import datetime, timedelta

from scripts.my_utils import get_model_size, get_model_architecture, process_dataset, send_request_post_sync_with_retry, cleanup_gpu_proc, get_merge_lora_info, remove_contents_in_dir, copy_contents_to_dir
import scripts.my_constants as my_cst
from scripts.grpo_config import make_config as make_grpo_config
from scripts.dpo_config import make_config as make_dpo_config
from scripts.instruct_config import make_config as make_instruct_config
from scripts.chat_config import make_config as make_chat_config
from scripts.my_logger import get_logger
import time
import torch
import re
logger = get_logger(__name__)

def patch_wandb_symlinks(base_dir:str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()

def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass


def copy_dataset_if_needed(dataset_path, file_format):
    """Copy dataset to Axolotl directories for non-HF datasets."""
    if file_format != FileFormat.HF.value:
        dataset_filename = os.path.basename(dataset_path)

        os.makedirs("/workspace/axolotl/data", exist_ok=True)
        os.makedirs("/workspace/axolotl", exist_ok=True)

        data_path = f"/workspace/axolotl/data/{dataset_filename}"
        root_path = f"/workspace/axolotl/{dataset_filename}"

        shutil.copy(dataset_path, data_path)
        shutil.copy(dataset_path, root_path)

        return data_path
    return dataset_path
def copy_dataset_to_axolotl_directories(dataset_path):
    dataset_filename = os.path.basename(dataset_path)
    data_path, root_path = train_paths.get_axolotl_dataset_paths(dataset_filename)
    shutil.copy(dataset_path, data_path)
    shutil.copy(dataset_path, root_path)

    return data_path

def create_config(
        task_id, 
        model, 
        dataset, 
        dataset_type, 
        file_format, 
        output_dir, 
        hours_to_complete,
        expected_repo_name=None,
        huggingface_username=None,
        huggingface_token=None,
        disable_upload=True,
        log_wandb=True
    ):
    """Create the axolotl config file with appropriate settings."""
    # if isinstance(dataset_type, InstructTextDatasetType | DpoDatasetType | ChatTemplateDatasetType):
    #     config_path = "/workspace/axolotl/base.yml"
    # elif isinstance(dataset_type, GrpoDatasetType):
    #     config_path = "/workspace/axolotl/base_grpo.yml"
    # else:
    #     raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")
    config_path = train_paths.get_axolotl_base_config_path(dataset_type)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    # model_path = f"{train_cst.CACHE_PATH}/models/{model.replace('/', '--')}"
    model_path = str(train_paths.get_text_base_model_path(model))
    config["base_model"] = model_path
    config["mlflow_experiment_name"] = dataset
    os.makedirs(output_dir, exist_ok=True)
    # config["output_dir"] = output_dir
    config["output_dir"] = str(output_dir)
    
    if log_wandb:
        config["wandb_runid"] = f"{task_id}_{expected_repo_name}"
        config["wandb_name"] = f"{task_id}_{expected_repo_name}"
        config["wandb_mode"] = "offline"
        os.makedirs(train_cst.WANDB_LOGS_DIR, exist_ok=True)
    else:
        for key in list(config.keys()):
            if key.startswith("wandb"):
                config.pop(key)
    config = update_flash_attention(config, model)


    # if not disable_upload:
    #     hf_username = huggingface_username or os.environ.get("HUGGINGFACE_USERNAME", "rayonlabs")
    #     os.environ["HUGGINGFACE_USERNAME"] = hf_username

    #     repo_name = expected_repo_name or str(uuid.uuid4())
    #     config["hub_model_id"] = f"{hf_username}/{repo_name}"

    #     if huggingface_token:
    #         os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
    # else:
    #     for key in list(config.keys()):
    #         if key.startswith("wandb") or key.startswith("hub"):
    #             config.pop(key)

    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"

            if "path" in ds:
                # ds["path"] = "/workspace/axolotl/data"
                ds["path"] = train_cst.AXOLOTL_DIRECTORIES["data"]

            ds["data_files"] = [os.path.basename(dataset)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}

    model_size = get_model_size(model)
    model_config = get_model_architecture(model_path)
    dataset_num_rows = process_dataset(model, dataset)
    remained_minutes = hours_to_complete * 60 - 15
    config_updated = {}
    if isinstance(dataset_type, InstructTextDatasetType):
        config_updated = make_instruct_config(
            model_id=model,
            model_size=model_size,
            remained_minutes=remained_minutes,
            model_conf = model_config,
            dataset_num_rows=dataset_num_rows,
        )
    elif isinstance(dataset_type, ChatTemplateDatasetType):
        config_updated = make_chat_config(
            model_id=model,
            model_size=model_size,
            remained_minutes=remained_minutes,
            model_conf = model_config,
            dataset_num_rows=dataset_num_rows,
        )
    elif isinstance(dataset_type, DpoDatasetType):
        config_updated = make_dpo_config(
            model_id=model,
            model_size=model_size,
            remained_minutes=remained_minutes,
            model_conf = model_config,
            dataset_num_rows=dataset_num_rows,
        )
    elif isinstance(dataset_type, GrpoDatasetType):
        config_updated = make_grpo_config(
            model_id=model,
            model_size=model_size,
            remained_minutes=remained_minutes,
            model_conf = model_config,
            dataset_num_rows=dataset_num_rows,
            dataset_type=dataset_type,
        )
    else:
        logger.error(f'❌❌❌Unknown task type !!!!! ❌❌❌')


    for key, item in config_updated.items():
        config[key] = item
    logger.info(f'config_updated: {config}')
    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions],
            task_id,
            # destination_dir="/workspace/axolotl/src/",
            destination_dir=train_cst.AXOLOTL_DIRECTORIES["src"],
        )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    config['max_steps'] = 12
    config['save_steps'] = 5
    config['eval_steps'] = 5
    config['val_set_size'] = 0.0001
    # config['micro_batch_size'] = 2
    # config['eval_batch_size'] = 2

    # config_path = os.path.join("/workspace/axolotl/configs", f"{task_id}.yml")
    config_path = os.path.join(train_cst.AXOLOTL_DIRECTORIES["configs"], f"{task_id}.yml")
    save_config(config, config_path)
    return config_path


def run_training(config_path, model_id, hours_to_complete):
    print(f"Starting training with config: {config_path}", flush=True)
    """Run the training process using the specified config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vllm_proc = None
    train_proc = None
    training_env = os.environ.copy()
    training_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    training_env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    b_merge_lora = False
    b_overtime_training = False
    try:
        print("Starting training subprocess...\n", flush=True)
        current_job_train_limit_at = datetime.utcnow() + timedelta(hours=hours_to_complete, minutes=-15)
        # current_job_train_limit_at = datetime.utcnow() + timedelta(seconds=500)
        logger.info(f'current_job_train_limit_at; {current_job_train_limit_at}')
        
        if my_cst.NUM_GPUS == 1:
            train_proc = subprocess.Popen(
                [
                    "accelerate",
                    "launch",
                    "-m",
                    "axolotl.cli.train",
                    config_path,
                    
                ],
                env=training_env
            )
        else:
            if config.get("rl") == "grpo" and config.get('trl').get('use_vllm', False):
                
                half_gpus = my_cst.NUM_GPUS // 2
                vllm_gpus = [i for i in range(half_gpus, my_cst.NUM_GPUS)]
                str_vllm_gpus = ",".join(map(str, vllm_gpus))
                vllm_env = os.environ.copy()
                vllm_env["CUDA_VISIBLE_DEVICES"] = str_vllm_gpus
                vllm_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                vllm_env["HF_HUB_DISABLE_TELEMETRY"] = "1"
                logger.info(f'starting vllm server on gpus: {str_vllm_gpus}')
                vllm_proc = subprocess.Popen(
                    [
                        "axolotl",
                        "vllm-serve",
                        config_path
                    ], 
                    env=vllm_env
                )
                
                train_gpus = [i for i in range(0, half_gpus)]
                str_train_gpus=",".join(map(str, train_gpus))
                training_env["CUDA_VISIBLE_DEVICES"] = str_train_gpus
                logger.info(f'starting train server on gpus: {str_train_gpus}')
                train_proc = subprocess.Popen(
                    [
                        "accelerate",
                        "launch",
                        "--num-processes",
                        str(half_gpus),
                        "-m",
                        "axolotl.cli.train",
                        config_path,
                        
                    ],
                    env=training_env
                )
            else:
                train_proc = subprocess.Popen(
                    [
                    "accelerate",
                    "launch",
                    "--num-processes",
                    str(my_cst.NUM_GPUS),
                    "-m",
                    "axolotl.cli.train",
                    config_path,
                        
                    ],
                    env=training_env
                )

        # for line in process.stdout:
        #     print(line, end="", flush=True)
        n_cnt = 0
        while train_proc.poll() is None:
            if datetime.utcnow() > current_job_train_limit_at:
                b_overtime_training = True
                logger.info(f'current_job_train_limit_at; {current_job_train_limit_at}, process is terminating')
                if not cleanup_gpu_proc(train_proc):
                    logger.warning("Forcing train full GPU memory cleanup")
                    # As last resort, kill all Python processes using GPU
                    time.sleep(2)
                    train_proc.kill()
                    torch.cuda.empty_cache()
                if vllm_proc:
                    if not cleanup_gpu_proc(vllm_proc):
                        logger.warning("Forcing vllm full GPU memory cleanup")
                        # As last resort, kill all Python processes using GPU
                        time.sleep(2)
                        train_proc.kill()
                        torch.cuda.empty_cache()
            logger.info(f'Waiting, n_cnt: {n_cnt} to complete...')
            n_cnt += 1
            time.sleep(10)  
        return_code = train_proc.wait()

        if b_overtime_training:
            logger.error('Training was stopped due to timeout.')
        if b_overtime_training == False:
            entries = os.listdir(config['output_dir'])
            print(entries)
            adapter_bin_path = os.path.join(config['output_dir'], 'adapter_model.bin')
            adapter_safetensors_path = os.path.join(config['output_dir'], 'adapter_model.safetensors')

            bin_exists = os.path.exists(adapter_bin_path)
            safetensors_exists = os.path.exists(adapter_safetensors_path)
            logger.info(f"adapter_model.bin exists: {bin_exists}")
            logger.info(f"adapter_model.safetensors exists: {safetensors_exists}")
            if bin_exists or safetensors_exists:
                logger.info('adapter_model.bin or adapter_model.safetensors exists, training was successful')
            else:
                logger.error('adapter_model.bin or adapter_model.safetensors does not exists, training was failed')
                b_overtime_training = True
        # if return_code != 0:
        #     raise subprocess.CalledProcessError(return_code, training_command)
        if b_overtime_training:
                
            entries = os.listdir(config['output_dir'])

            # Pattern to match 'checkpoint-<number>'
            pattern = re.compile(r'checkpoint-(\d+)')

            # Filter and find the maximum checkpoint number
            max_checkpoint = -1
            latest_checkpoint_dir = None

            for entry in entries:
                match = pattern.match(entry)
                if match:
                    checkpoint_num = int(match.group(1))
                    if checkpoint_num > max_checkpoint:
                        max_checkpoint = checkpoint_num
                        latest_checkpoint_dir = entry
            latest_checkpoint_dir=f"{config['output_dir']}/{latest_checkpoint_dir}"
            logger.info(f'latest_checkpoint_dir: {latest_checkpoint_dir}')

        print("Training subprocess completed successfully.", flush=True)
        b_merge_lora = get_merge_lora_info(model_id)
        # b_merge_lora = False
        if b_merge_lora:
            if b_overtime_training and latest_checkpoint_dir is not None:
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "axolotl.cli.merge_lora",
                        config_path,
                        f'--lora-model-dir={latest_checkpoint_dir}'
                    ]
                )
            else:
                subprocess.run(
                    [
                        "axolotl",
                        "merge-lora",
                        config_path
                    ]
                )
            
            src_dir = f"{config['output_dir']}/merged"
            dest_dir = config['output_dir']

            remove_contents_in_dir(dest_dir, ['merged'])

            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dest_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            shutil.rmtree(src_dir)
        else:

            if b_overtime_training == False:
                src_dir = config['output_dir']
                
                keep_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors", ".gitattributes"]
                remove_contents_in_dir(src_dir, keep_files)
            else:
                dest_dir = config['output_dir']
                src_dir = latest_checkpoint_dir
                remove_contents_in_dir(dest_dir, [os.path.basename(src_dir)])
                keep_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors", ".gitattributes"]
                remove_contents_in_dir(src_dir, keep_files)
                
                copy_contents_to_dir(src_dir, dest_dir)
                shutil.rmtree(src_dir)

        # os.remove(config_path)
    except Exception as e:
        print(f"Training subprocess failed!, {str(e)}", flush=True)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Exit Code: {e.returncode}", flush=True)
            print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        
        # raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")
        src_dir = config['base_model']
        dest_dir = config['output_dir']
        ignore_patterns = ["checkpoint-*", "runs*", "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "README.md", "*.msgpack", "*.ot"]
        copy_contents_to_dir(src_dir, dest_dir)



async def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument("--task-type", required=True, choices=["InstructTextTask", "DpoTask", "GrpoTask"], help="Type of task")
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()
    # temp_env = os.environ.copy()
    # print(f"Environment variables: {temp_env}", flush=True)
    # for directory in [
    #     "/workspace/axolotl/data",
    #     "/workspace/axolotl/data_prepared",
    #     "/workspace/axolotl/configs",
    #     "/workspace/axolotl/outputs",
    #     "/workspace/input_data",
    #     "/workspace/axolotl"
    # ]:
    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)

        if args.task_type == TaskType.DPOTASK.value:
            dataset_type = DpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.INSTRUCTTEXTTASK.value:
            dataset_type = InstructTextDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.GRPOTASK.value:
            dataset_type = GrpoDatasetType(**dataset_type_dict)
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    # base_dataset_path = f"{train_cst.CACHE_PATH}/datasets"
    # dataset_path = f"{base_dataset_path}/{args.task_id}_train_data.json" if args.file_format == FileFormat.S3.value else f"{base_dataset_path}/{args.dataset.replace('/', '--')}"
    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    
    if args.task_type == TaskType.DPOTASK.value:
        adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True)
    elif args.task_type == TaskType.GRPOTASK.value:
        adapt_columns_for_grpo_dataset(dataset_path, dataset_type)

    # dataset_path = copy_dataset_if_needed(dataset_path, args.file_format)
    dataset_path = copy_dataset_to_axolotl_directories(dataset_path)

    # output_dir = f"/workspace/axolotl/outputs/{args.task_id}/{args.expected_repo_name}"
    output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config_path = create_config(
        args.task_id,
        args.model,
        dataset_path,
        dataset_type,
        args.file_format,
        output_dir,
        int(args.hours_to_complete),
        args.expected_repo_name,
        log_wandb=True
    )
    # NUM_GPUS=torch.cuda.device_count()
    # print(f"Number of GPUs available: {NUM_GPUS}", flush=True)
    run_training(config_path, args.model, args.hours_to_complete)

    patch_model_metadata(output_dir, args.model)

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)

if __name__ == "__main__":
    asyncio.run(main())
