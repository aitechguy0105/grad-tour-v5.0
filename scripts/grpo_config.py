from scripts.my_logger import get_logger
import scripts.my_constants as cst
from scripts.my_utils import get_batch_checkpointing_info, get_train_and_eval_steps_from_data, get_liger_info, get_flash_attention_info
import math
from datetime import datetime
from core.models.utility_models import GrpoDatasetType
logger = get_logger(__name__)

def create_reward_funcs_file(reward_funcs: list[str], task_id: str) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.
    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    # filepath = os.path.join(f'../reward_funcs', f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    # with open(filepath, "w") as f:
    #     f.write("# Auto-generated reward functions file\n\n")
    #     for reward_func in reward_funcs:
    #         f.write(f"{reward_func}\n\n")

    return filename, func_names
def make_config(
    model_id: str,
    model_size: int,
    remained_minutes: int,
    model_conf: dict,
    dataset_num_rows: int,
    dataset_type: GrpoDatasetType | None = None,

):
    # model_size = max(cst.MIN_MODEL_SIZE, model_size)
    # logger.info(f'updated model size: {model_size}')
    #################### output
    
    micro_batch_size = cst.MIN_MICRO_BATCH_SIZE
    eval_batch_size = cst.MIN_EVAL_BATCH_SIZE
    gradient_checkpointing = False
    max_steps = cst.MIN_MAX_STEPS
    flash_attention = True
    early_stopping_patience = cst.MIN_EARLY_STOPPING_PATIENCE
    eval_steps = cst.DEFAULT_CHECKPOINT_STEPS
    save_steps = cst.DEFAULT_CHECKPOINT_STEPS
    lr_scheduler = 'constant'
    lr_scheduler_kwargs = None
    sequence_len = cst.MIN_SEQUENCE_LEN
    gradient_accumulation_steps = cst.MIN_ACCUMULATION_STEPS
    num_epochs = cst.DEFAULT_NUM_EPOCHS
    val_set_size = cst.DEFAULT_VAL_SET_SIZE
    learning_rate = cst.DEFAULT_LEARNING_RATE
    lora_dropout = 0.05
    num_gpus = cst.NUM_GPUS
    reward_func_num = len(dataset_type.reward_functions)

    config = {}
    config_to_save = {}
    #################### sequence_len
    if model_conf is None or not isinstance(model_conf, dict):
        logger.warning(f'model_conf is None return max_steps = -1')
        config = { "max_steps": -1 }
        return config
    max_sequence_len = min(4096, model_conf.get('max_position_embeddings', 1024))
    sequence_len = min(1024, max_sequence_len)
    logger.info(f'sequence_len: {sequence_len}')

    ##################### reward_func, vllm config
    # filename, reward_funcs_names = create_reward_funcs_file(
    #     [reward_function.reward_func for reward_function in dataset_type.reward_functions], job_id
    # )
    config["trl"] = {
        "beta": 0.04,
        "max_completion_length": 256,
        "use_vllm": True,
        "num_generations": 4,
    }
    
    # config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
    # config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]
    
    if num_gpus <= 2 or model_size >= cst.LARGE_MODEL_SIZE or \
        model_conf.get('num_attention_heads', 2) % 2 == 1 or \
        model_id in [
            'unsloth/OpenHermes-2.5-Mistral-7B', 
            "defog/sqlcoder-7b-2",
            "facebook/opt-350m",
            "huggyllama/llama-7b",
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'dunzhang/stella_en_1.5B_v5',
            "NousResearch/Meta-Llama-3-8B-Alternate-Tokenizer",
            "elyza/Llama-3-ELYZA-JP-8B",
            "Korabbit/llama-2-ko-7b",
            "furiosa-ai/mlperf-gpt-j-6b",
            "databricks/dolly-v2-3b",
            'Eurdem/Defne_llama3_2x8B'] or \
        model_conf.get('model_type', None) in ['mistral']:
        config["trl"]["use_vllm"] = False
    else:
        config["trl"]["vllm_server_host"] = "0.0.0.0"
        config["trl"]["vllm_server_port"] = 8000
        config["trl"]["vllm_server_timeout"] = 300
        half_gpus = num_gpus // 2
        config["vllm"] = {
            "host": "0.0.0.0",
            "port": 8000,
            "tensor_parallel_size": half_gpus,
            "gpu_memory_utilization": 0.85,
            "dtype": "auto",
        }
        if model_conf.get('num_attention_heads', 2) % 4 == 2:
            config["vllm"]["tensor_parallel_size"] = 2
    #################### eval, micro batch size, gradient_checkpointing <- sequence len, model size
    # config["trl"]["use_vllm"] = False
    if config["trl"]["use_vllm"]:
        micro_batch_size = 2
        eval_batch_size = 2
        gradient_accumulation_steps = 4
        gradient_checkpointing = True

    else:
        micro_batch_size = 2
        eval_batch_size = 2
        if model_size >= cst.LARGE_MODEL_SIZE:
            micro_batch_size = 1
            eval_batch_size = 1

        gradient_accumulation_steps = 4
        gradient_checkpointing = True
    
    # train_batch_size_total = num_gpus * micro_batch_size * gradient_accumulation_steps
    
    half_gpus = num_gpus // 2
    train_batch_size_total = half_gpus * micro_batch_size * gradient_accumulation_steps
    eval_batch_size_total = num_gpus * 4
    
    if config["trl"]["use_vllm"] == False:
        train_batch_size_total = num_gpus * micro_batch_size * gradient_accumulation_steps
        eval_batch_size_total = num_gpus * eval_batch_size
    
    logger.info(f'micro_batch_size: {micro_batch_size}, eval_batch_size: {eval_batch_size}, num_gpus: {num_gpus}, train_batch_size_total: {train_batch_size_total}, eval_batch_size_total: {eval_batch_size_total}')
    
    num_generations = 4
    if eval_batch_size_total <= 2:
        num_generations = 2
        config["trl"]["num_generations"] = num_generations
    #################### flash attention, eager attention
    flash_attention = get_flash_attention_info(model_id, model_conf)
    model_architecture = None
    if model_conf is not None and isinstance(model_conf, dict):  
        # Check if 'architectures' key exists and is a list  
        architectures = model_conf.get('architectures')  
        if architectures and isinstance(architectures, list):  
            model_architecture = architectures[0]  # Access the first architecture  
    
    
    #################### train_steps_per_sec, eval_steps_per_sec
    if model_architecture == 'GPTNeoXForCausalLM':
        gradient_checkpointing = False
        micro_batch_size = 1
        eval_batch_size = 1
        gradient_accumulation_steps = 4
    adapter = 'lora' if model_size < cst.QLORA_MODEL_SIZE else 'qlora'
    train_steps_per_sec, eval_steps_per_sec = get_train_and_eval_steps_from_data(
        model_id = model_id,
        model_size = model_size,
        flash_support = flash_attention, 
        sequence_len = sequence_len, 
        train_micro_batch_size = micro_batch_size, 
        eval_batch_size = eval_batch_size, 
        gradient_checkpointing = gradient_checkpointing, 
        num_gpus = num_gpus,
        adapter = adapter
    )

    b_ratio_reward_func_num = 1.0
    default_reward_func_num = 3
    if reward_func_num is not None and reward_func_num > 3:
        b_ratio_reward_func_num = math.pow(1.25, reward_func_num / default_reward_func_num)
        b_ratio_reward_func_num = max(1, b_ratio_reward_func_num)
    if model_size >= cst.EXTREMELY_LARGE_MODEL_SIZE:
        
        train_steps_per_sec = train_steps_per_sec / 25.0 / b_ratio_reward_func_num
        eval_steps_per_sec = train_steps_per_sec * 1.2
    elif model_size >= cst.LARGE_MODEL_SIZE:
        
        train_steps_per_sec = train_steps_per_sec / 13.0 / b_ratio_reward_func_num
        eval_steps_per_sec = train_steps_per_sec * 1.2
        
    elif model_size >= cst.MEDIUM_MODEL_SIZE:
        train_steps_per_sec = train_steps_per_sec / 11.5 / b_ratio_reward_func_num
        eval_steps_per_sec = train_steps_per_sec * 1.2

    elif model_size >= cst.SMALL_MODEL_SIZE:
        train_steps_per_sec = train_steps_per_sec / 8.5 / b_ratio_reward_func_num
        eval_steps_per_sec = train_steps_per_sec * 1.2

    elif model_size >= cst.EXTREMELY_SMALL_MODEL_SIZE:
        train_steps_per_sec = train_steps_per_sec / 12.5 / b_ratio_reward_func_num
        eval_steps_per_sec = train_steps_per_sec * 1.2
    else:
        train_steps_per_sec = train_steps_per_sec / 8.0 / b_ratio_reward_func_num
        eval_steps_per_sec = train_steps_per_sec * 1.1
    
    if sequence_len <= 256:
        train_steps_per_sec = train_steps_per_sec / 2.1
        eval_steps_per_sec = train_steps_per_sec * 1.2
    elif sequence_len <= 512:
        train_steps_per_sec = train_steps_per_sec / 1.2
        eval_steps_per_sec = train_steps_per_sec * 1.2

    train_steps_per_sec = min(0.07, train_steps_per_sec)
    eval_steps_per_sec = min(0.1, eval_steps_per_sec)
    if num_gpus == 1:
        train_steps_per_sec = min(0.055, train_steps_per_sec)
        eval_steps_per_sec = min(0.8, eval_steps_per_sec)
    logger.info(f'Adjust for GRPO task. train_steps_per_sec: {train_steps_per_sec}, eval_steps_per_sec: {eval_steps_per_sec}')
    n_long_grpo_func = 0
    if dataset_type is not None:
        for reward_function in dataset_type.reward_functions:
            if 'detoxify' in reward_function.reward_func.lower():
                n_long_grpo_func += 1

    if config["trl"]["use_vllm"]:
        train_steps_per_sec = train_steps_per_sec * 3
        eval_steps_per_sec = train_steps_per_sec * 2
        logger.info(f'===========Increase for GRPO task (VLLM 4 more GPUs). train_steps_per_sec: {train_steps_per_sec}, eval_steps_per_sec: {eval_steps_per_sec}')
    if n_long_grpo_func >= 1:
        train_steps_per_sec = train_steps_per_sec / min(4, (n_long_grpo_func * 1.5))
        eval_steps_per_sec = train_steps_per_sec * 2
        logger.info(f'===========Reduce for GRPO task (long grpo func) n_long_grpo_func: {n_long_grpo_func}. train_steps_per_sec: {train_steps_per_sec}, eval_steps_per_sec: {eval_steps_per_sec}')

    #################### checkpoint_steps, val_set_size



    steps_per_epoch = math.ceil(dataset_num_rows * num_generations / train_batch_size_total)
    rough_estimated_total_train_steps = math.ceil(remained_minutes * 60 * train_steps_per_sec)
    if rough_estimated_total_train_steps >= steps_per_epoch * 3:
        checkpoint_steps = math.ceil(steps_per_epoch / 2)
    elif rough_estimated_total_train_steps >= steps_per_epoch * 1.5:
        checkpoint_steps = math.ceil(steps_per_epoch / 3)
    else:
        checkpoint_steps = math.ceil(rough_estimated_total_train_steps / 3)
    
    checkpoint_steps = min(1000, max(150, checkpoint_steps))
    logger.info(f'=========== checkpoint_steps: {checkpoint_steps}')

    n_grpo_max_val_examples = cst.MAX_VAL_DATASET_EXAMPLES_GRPO_LARGE_INSTANCE if config["trl"]["use_vllm"] and rough_estimated_total_train_steps >= 1.2 * steps_per_epoch else cst.MAX_VAL_DATASET_EXAMPLES_GRPO_SMALL_INSTANCE
    n_grpo_min_val_examples = cst.MIN_VAL_DATASET_EXAMPLES_GRPO_LARGE_INSTANCE if config["trl"]["use_vllm"] and rough_estimated_total_train_steps >= 1.2 * steps_per_epoch else cst.MIN_VAL_DATASET_EXAMPLES_GRPO_SMALL_INSTANCE

    if dataset_num_rows * cst.DEFAULT_VAL_SET_SIZE > n_grpo_max_val_examples:
        val_set_size = n_grpo_max_val_examples / dataset_num_rows
    if dataset_num_rows * cst.DEFAULT_VAL_SET_SIZE < n_grpo_min_val_examples:
        val_set_size = n_grpo_min_val_examples / dataset_num_rows
    val_set_size = max(min(1, val_set_size), 0)

  
    

    #################### eval_steps, save_steps, learning_rate
    eval_steps = checkpoint_steps
    save_steps = checkpoint_steps
    valid_examples = val_set_size * dataset_num_rows
    train_examples = dataset_num_rows - valid_examples

    eval_steps_per_checkpoint = valid_examples * num_generations / eval_batch_size_total

    estimated_train_sec_per_checkpoint = checkpoint_steps / train_steps_per_sec
    estimated_eval_sec_per_checkpoint = eval_steps_per_checkpoint / eval_steps_per_sec

    total_remained_sec = remained_minutes * 60
    estimated_epochs = 0
    not_multiplied_estimated_epochs = 0
    checkpoint_times_multiplier = 1.0

    total_remained_sec_after_first_eval = total_remained_sec - estimated_eval_sec_per_checkpoint
    estimated_train_eval_sec_per_checkpoint = (estimated_eval_sec_per_checkpoint + estimated_train_sec_per_checkpoint) * 1.1
    n_estimated_checkpoint_times = math.ceil(total_remained_sec_after_first_eval / estimated_train_eval_sec_per_checkpoint)
    logger.info(f'=========== n_estimated_checkpoint_times: {n_estimated_checkpoint_times}')
    max_steps = n_estimated_checkpoint_times * checkpoint_steps + 5
    logger.info(f'=========== max_steps: {max_steps}')

    logger.info(f'=========== steps_per_epoch: {steps_per_epoch}')

    estimated_epochs = max_steps / steps_per_epoch
    if estimated_epochs < 3.0 and sequence_len >= 2048:
        logger.warning(f'=========== adjusting sequence_len to 1024, estimated_epochs: {estimated_epochs} is too small, sequence_len: {sequence_len}')
        sequence_len = 1024
        n_estimated_checkpoint_times = math.ceil(n_estimated_checkpoint_times * 1.5) 
        max_steps = math.ceil(n_estimated_checkpoint_times * checkpoint_steps) + 5
        logger.info(f'=========== max_steps: {max_steps}')
        estimated_epochs = max_steps / steps_per_epoch
        logger.info(f'=========== estimated_epochs: {estimated_epochs}')
    
    if estimated_epochs > cst.MAX_EPOCHS:
        logger.warning(f'=========== estimated_epochs: {estimated_epochs} is too big, reduce to {cst.MAX_EPOCHS}, max_steps: {max_steps}')
        n_estimated_checkpoint_times = math.ceil(n_estimated_checkpoint_times * cst.MAX_EPOCHS / estimated_epochs)
        max_steps = math.ceil(n_estimated_checkpoint_times * checkpoint_steps) + 5
        estimated_epochs = cst.MAX_EPOCHS
    
    not_multiplied_estimated_epochs = estimated_epochs
    logger.info(f'=========== not_multiplied_estimated_epochs: {estimated_epochs}')

    if estimated_epochs < 0.5:
        checkpoint_times_multiplier = 3.5
    elif estimated_epochs < 1.0:
        checkpoint_times_multiplier = 2.5
    elif estimated_epochs < 5.0:
        checkpoint_times_multiplier = 1.5

    n_estimated_checkpoint_times = math.ceil(n_estimated_checkpoint_times * checkpoint_times_multiplier)
    if checkpoint_times_multiplier > 1.0:
        b_maximized = True
        logger.info(f'=========== maximize n_estimated_checkpoint_times for TEST  🙏🙏🙏 : {n_estimated_checkpoint_times}')
        max_steps = n_estimated_checkpoint_times * checkpoint_steps + 5
        logger.info(f'=========== increae max_steps: {max_steps}')
        
        estimated_epochs = max_steps / steps_per_epoch
        logger.info(f'=========== increase estimated_epochs: {estimated_epochs}')

    early_stopping_patience = max( min(math.ceil(steps_per_epoch / checkpoint_steps), 10), cst.MIN_EARLY_STOPPING_PATIENCE)

    learning_rate = 5e-6

    if not_multiplied_estimated_epochs <= 1:
        learning_rate = 5e-6
        lr_scheduler = 'constant'
    elif not_multiplied_estimated_epochs <= 1.5:
        learning_rate = 5e-6
        lr_scheduler = 'cosine_with_min_lr'
        lr_scheduler_kwargs = {
            'min_lr_rate': 0.3
        }
    elif not_multiplied_estimated_epochs <= 3:
        learning_rate = 3.5e-6
        lr_scheduler = 'cosine_with_min_lr'
        lr_scheduler_kwargs = {
            'min_lr_rate': 0.1
        }
    else:
        not_multiplied_estimated_epochs = 3e-6
        lr_scheduler = 'cosine_with_min_lr'
        lr_scheduler_kwargs = {
            'min_lr_rate': 0.01
        }
    if 'mistral' in model_id.lower() or model_architecture in ['MistralForCausalLM', 'MixtralForCausalLM']:
        if not_multiplied_estimated_epochs <= 1:
            learning_rate = 2e-6
            lr_scheduler = 'constant'
        elif not_multiplied_estimated_epochs <= 1.5:
            learning_rate = 2e-6
            lr_scheduler = 'cosine_with_min_lr'
            lr_scheduler_kwargs = {
                'min_lr_rate': 0.3
            }
        elif not_multiplied_estimated_epochs <= 3:
            learning_rate = 1.5e-6
            lr_scheduler = 'cosine_with_min_lr'
            lr_scheduler_kwargs = {
                'min_lr_rate': 0.1
            }
        else:
            not_multiplied_estimated_epochs = 1.5e-6
            lr_scheduler = 'cosine_with_min_lr'
            lr_scheduler_kwargs = {
                'min_lr_rate': 0.01
            }
    # if 'qwen' in model_id.lower() or model_conf.get('model_type', None) in ['qwen2', 'qwen2_5_vl', 'qwen2_vl', 'qwen3']:
    #     if not_multiplied_estimated_epochs <= 1:
    #         learning_rate = 2e-5
    #         lr_scheduler = 'constant'
    #     elif not_multiplied_estimated_epochs <= 1.5:
    #         learning_rate = 2e-5
    #         lr_scheduler = 'cosine_with_min_lr'
    #         lr_scheduler_kwargs = {
    #             'min_lr_rate': 0.3
    #         }
    #     else:
    #         learning_rate = 2e-5
    #         lr_scheduler = 'cosine_with_min_lr'
    #         lr_scheduler_kwargs = {
    #             'min_lr_rate': 0.01
    #         }
    #################### customize lora
    lora_r = 256 if model_size < cst.LARGE_MODEL_SIZE else 128
    lora_alpha = 512 if model_size < cst.LARGE_MODEL_SIZE else 256
    optimizer = 'adamw_bnb_8bit'
    
    config['adapter'] = adapter
    config['micro_batch_size'] = micro_batch_size
    config['eval_batch_size'] = eval_batch_size
    config['gradient_checkpointing'] = gradient_checkpointing
    config['max_steps'] = max_steps
    config['flash_attention'] = flash_attention
    # config['eager_attention'] = eager_attention
    config['early_stopping_patience'] = None
    config['load_best_model_at_end'] = False
    config['eval_steps'] = eval_steps
    config['save_steps'] = save_steps
    config['lr_scheduler'] = lr_scheduler
    config['lr_scheduler_kwargs'] = lr_scheduler_kwargs
    config['learning_rate'] = learning_rate
    config['sequence_len'] = sequence_len
    config['lora_r'] = lora_r
    config['lora_alpha'] = lora_alpha
    config['optimizer'] = optimizer
    config['num_epochs'] = num_epochs
    config['val_set_size'] = val_set_size
    if num_gpus == 1:
        config['optimizer'] = 'adamw_bnb_8bit'
        config['optim_args'] = {
            'adam_beta1': 0.9,
            'adam_beta2': 0.99,
            'adam_epsilon': 1e-8
        }
        config['lora_target_linear'] = True
        if model_size < cst.LARGE_MODEL_SIZE:
            config['peft_init_lora_weights'] ='eva'
        if 'torch_dtype' in model_conf and model_conf['torch_dtype'] == 'bfloat16':
            config['bf16'] = True
            if adapter == 'qlora':
                config['bf16'] = 'auto'

        if adapter == 'qlora':
            config['load_in_4bit'] = True
            if 'model_type' in model_conf and model_conf['model_type'] in ['deepseek_v3', 'deepseek_v2']:
                config['lora_r'] = 256
                config['lora_alpha'] = 256 
    else:
        
        config['optimizer'] = 'adamw_bnb_8bit'
        config['optim_args'] = {
            'adam_beta1': 0.9,
            'adam_beta2': 0.99,
            'adam_epsilon': 1e-8
        }
        config['lora_target_linear'] = True
        if model_size < cst.LARGE_MODEL_SIZE:
            config['peft_init_lora_weights'] ='eva'
        if adapter == 'qlora':
            config['load_in_4bit'] = True
            if 'model_type' in model_conf and model_conf['model_type'] in ['deepseek_v3', 'deepseek_v2']:
                config['lora_r'] = 256
                config['lora_alpha'] = 256 


        if 'torch_dtype' in model_conf and model_conf['torch_dtype'] == 'bfloat16':
            config['bf16'] = True
        
        config['deepspeed'] = 'deepspeed_configs/zero1.json'

        if gradient_checkpointing == True:
      
            gradient_checkpointing_kwargs = {
                'use_reentrant': False
            }
        
            config['gradient_checkpointing_kwargs'] = gradient_checkpointing_kwargs
  
        # optimizer = 'adamw_torch_fused'
        # adam_beta2= 0.95
        # adam_eps= 0.00001
        # config['optimizer'] = optimizer
        # config['adam_beta2']= adam_beta2
        # config['adam_eps']= adam_eps
        # if model_size > FSDP_MODEL_SIZE or (num_gpus == 8 and model_size >= 13_000_000_000):
        #     decoder_layer = get_fsdp_transformer_layer_cls_to_wrap_info(model_id, model_architecture)
        #     fsdp = [
        #         'full_shard',
        #         'auto_wrap'
        #     ]
        #     fsdp_config = {
        #         'fsdp_state_dict_type': 'FULL_STATE_DICT',
        #         'fsdp_sharding_strategy': 'FULL_SHARD',
        #         'fsdp_auto_wrap_policy': 'TRANSFORMER_BASED_WRAP',
        #         'fsdp_transformer_layer_cls_to_wrap': decoder_layer,
        #         'fsdp_limit_all_gathers': True,
        #         'fsdp_sync_module_states': True,
        #         'fsdp_backward_prefetch': 'BACKWARD_PRE',
        #         'fsdp_use_orig_params': False,
        #         'fsdp_cpu_ram_efficient_loading': True,
        #         'fsdp_offload_params': True,
        #     }
            

        #     config['fsdp'] = fsdp
        #     config['fsdp_config'] = fsdp_config
        
    if get_liger_info(model_conf):
        config['plugins'] = ['axolotl.integrations.liger.LigerPlugin']
        config['liger_rope'] = True
        config['liger_rms_norm'] = True
        config['liger_glu_activation'] = True
        config['liger_layer_norm'] = True          

    max_grad_norm = 1.0
    config['num_epochs'] = 10
    config['lora_dropout'] = lora_dropout
    config['max_grad_norm'] = max_grad_norm
    if 'pythia' in model_id and model_architecture is not None and model_architecture == 'GPTNeoXForCausalLM':
        lora_fan_in_fan_out = True
        config['lora_fan_in_fan_out'] = lora_fan_in_fan_out
        micro_batch_size = max(1, micro_batch_size // 2)
        eval_batch_size = max(1, eval_batch_size // 2)

    
    logger.info(config)
    
    return config
