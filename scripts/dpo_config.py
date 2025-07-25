from scripts.my_logger import get_logger
import scripts.my_constants as my_cst
from scripts.my_utils import get_batch_checkpointing_info, get_train_and_eval_steps_from_data
import math
from datetime import datetime
logger = get_logger(__name__)

def make_config(
    model_id: str,
    model_size: int,
    remained_minutes: int,
    model_conf: dict,
    dataset_num_rows: int,
):
    # model_size = max(my_cst.MIN_MODEL_SIZE, model_size)
    # logger.info(f'updated model size: {model_size}')
    #################### output
    micro_batch_size = my_cst.MIN_MICRO_BATCH_SIZE
    eval_batch_size = my_cst.MIN_EVAL_BATCH_SIZE
    gradient_checkpointing = False
    max_steps = my_cst.MIN_MAX_STEPS
    flash_attention = True
    early_stopping_patience = my_cst.MIN_EARLY_STOPPING_PATIENCE
    eval_steps = my_cst.DEFAULT_CHECKPOINT_STEPS
    save_steps = my_cst.DEFAULT_CHECKPOINT_STEPS
    lr_scheduler = 'constant'
    lr_scheduler_kwargs = None
    sequence_len = my_cst.MIN_SEQUENCE_LEN
    gradient_accumulation_steps = my_cst.MIN_ACCUMULATION_STEPS
    num_epochs = my_cst.DEFAULT_NUM_EPOCHS
    val_set_size = my_cst.DEFAULT_VAL_SET_SIZE
    learning_rate = my_cst.DEFAULT_LEARNING_RATE
    lora_dropout = 0.05
    num_gpus = my_cst.NUM_GPUS


    config = {}
    config_to_save = {}
    #################### sequence_len

    max_sequence_len = min(4096, model_conf.get('max_position_embeddings', 1024))
    sequence_len = min(1024, max_sequence_len)
    logger.info(f'sequence_len: {sequence_len}')

    #################### eval, micro batch size, gradient_checkpointing <- sequence len, model size
    micro_batch_size, eval_batch_size, gradient_checkpointing = get_batch_checkpointing_info(model_size, sequence_len)
    if num_gpus >= 8:
        micro_batch_size = 2
        eval_batch_size = 2
        gradient_accumulation_steps = 4
        gradient_checkpointing = True
    
    micro_batch_size = min(4, micro_batch_size)
    eval_batch_size = min(4, eval_batch_size)

    if model_id == 'EleutherAI/gpt-neo-1.3B' and sequence_len >= 2048:
        logger.info(f'{model_id} sequence_len is bigger than 2048, gradient_checkpointing to True')
        gradient_checkpointing = True


    train_batch_size_total = num_gpus * micro_batch_size * gradient_accumulation_steps
    eval_batch_size_total = num_gpus * eval_batch_size
    logger.info(f'micro_batch_size: {micro_batch_size}, eval_batch_size: {eval_batch_size}, num_gpus: {num_gpus}, train_batch_size_total: {train_batch_size_total}, eval_batch_size_total: {eval_batch_size_total}')
            
    #################### flash attention, eager attention
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
        
    if 'codellama' in model_id.lower():
        if 'unsloth' in model_id.lower():
            flash_attention = True
        else:
            flash_attention = False
    if 'unsloth' in model_id.lower():
        flash_attention = True
   
    
    #################### train_steps_per_sec, eval_steps_per_sec
    
    adapter = 'lora' if model_size < my_cst.QLORA_MODEL_SIZE else 'qlora'
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
    
    if model_size >= my_cst.SMALL_MODEL_SIZE:
        train_steps_per_sec = train_steps_per_sec / 2.2
        eval_steps_per_sec = eval_steps_per_sec / 2.2
    elif model_size >= my_cst.EXTREMELY_SMALL_MODEL_SIZE:
        train_steps_per_sec = train_steps_per_sec / 2.1
        eval_steps_per_sec = eval_steps_per_sec / 2.1
    else:
        train_steps_per_sec /= 2.5
        eval_steps_per_sec /= 2.5
    train_steps_per_sec = min(0.4, train_steps_per_sec)
    eval_steps_per_sec = min(4.0, eval_steps_per_sec)
    logger.info(f'Reduce for DPO task. train_steps_per_sec: {train_steps_per_sec}, eval_steps_per_sec: {eval_steps_per_sec}')
    #################### checkpoint_steps, val_set_size

    if model_size < my_cst.LARGE_MODEL_SIZE:
        n_max_val_examples_for_not_large_models = my_cst.MAX_VAL_DATASET_EXAMPLES_DPO
        n_min_val_examples_for_not_large_models = my_cst.MIN_VAL_DATASET_EXAMPLES_DPO
        if num_gpus <= 2:
            n_max_val_examples_for_not_large_models /= 2
            n_min_val_examples_for_not_large_models /= 2
        if dataset_num_rows * my_cst.DEFAULT_VAL_SET_SIZE > n_max_val_examples_for_not_large_models:
            val_set_size = n_max_val_examples_for_not_large_models / dataset_num_rows
        if dataset_num_rows * my_cst.DEFAULT_VAL_SET_SIZE < n_min_val_examples_for_not_large_models:
            val_set_size = n_min_val_examples_for_not_large_models / dataset_num_rows
    else:
        n_max_val_examples_for_large_models = my_cst.MAX_VAL_DATASET_EXAMPLES_DPO / 2
        n_min_val_examples_for_large_models = my_cst.MIN_VAL_DATASET_EXAMPLES_DPO / 2
        if num_gpus <= 2:
            n_max_val_examples_for_not_large_models /= 2
            n_min_val_examples_for_not_large_models /= 2
        if dataset_num_rows * my_cst.DEFAULT_VAL_SET_SIZE > n_max_val_examples_for_large_models:
            val_set_size = n_max_val_examples_for_large_models / dataset_num_rows
        if dataset_num_rows * my_cst.DEFAULT_VAL_SET_SIZE < n_min_val_examples_for_large_models:
            val_set_size = n_min_val_examples_for_large_models / dataset_num_rows
    val_set_size = max(min(1, val_set_size), 0)

    steps_per_epoch = math.ceil(dataset_num_rows / train_batch_size_total)
    rough_estimated_total_train_steps = math.ceil(remained_minutes * 60 * train_steps_per_sec)
    if rough_estimated_total_train_steps >= steps_per_epoch * 3:
        checkpoint_steps = math.ceil(steps_per_epoch / 2)
    elif rough_estimated_total_train_steps >= steps_per_epoch * 1.5:
        checkpoint_steps = math.ceil(steps_per_epoch / 3)
    else:
        checkpoint_steps = math.ceil(rough_estimated_total_train_steps / 3)
    
    checkpoint_steps = min(1000, max(150, checkpoint_steps))
    

    #################### eval_steps, save_steps, learning_rate
    eval_steps = checkpoint_steps
    save_steps = checkpoint_steps
    valid_examples = val_set_size * dataset_num_rows
    train_examples = dataset_num_rows - valid_examples

    eval_steps_per_checkpoint = valid_examples / eval_batch_size_total

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
    
    if estimated_epochs > my_cst.MAX_EPOCHS:
        max_steps = math.ceil(n_estimated_checkpoint_times * checkpoint_steps * my_cst.MAX_EPOCHS / estimated_epochs ) + 5
        estimated_epochs = my_cst.MAX_EPOCHS
        logger.warning(f'=========== estimated_epochs: {estimated_epochs} is too big, reduce to {my_cst.MAX_EPOCHS}, max_steps: {max_steps}')

    not_multiplied_estimated_epochs = estimated_epochs
    logger.info(f'=========== estimated_epochs: {estimated_epochs}')

    if estimated_epochs < 7.0:
        checkpoint_times_multiplier = 1.5
    n_estimated_checkpoint_times = math.ceil(n_estimated_checkpoint_times * checkpoint_times_multiplier)
    if checkpoint_times_multiplier > 1.0:
        b_maximized = True
        logger.info(f'=========== maximize n_estimated_checkpoint_times for TEST  🙏🙏🙏 : {n_estimated_checkpoint_times}')
        max_steps = n_estimated_checkpoint_times * checkpoint_steps + 5
        logger.info(f'=========== increae max_steps: {max_steps}')
        
        estimated_epochs = max_steps / steps_per_epoch
        logger.info(f'=========== increase estimated_epochs: {estimated_epochs}')

    early_stopping_patience = max( min(math.ceil(steps_per_epoch / checkpoint_steps), 10), my_cst.MIN_EARLY_STOPPING_PATIENCE)

    learning_rate = 1e-5

    if not_multiplied_estimated_epochs <= 1:
        lr_scheduler = 'constant'
    elif not_multiplied_estimated_epochs <= 2:
        learning_rate = 8e-6
        lr_scheduler = 'cosine_with_min_lr'
        lr_scheduler_kwargs = {
            'min_lr_rate': 0.3
        }
    elif not_multiplied_estimated_epochs <= 3:
        learning_rate = 7e-6
        lr_scheduler = 'cosine_with_min_lr'
        lr_scheduler_kwargs = {
            'min_lr_rate': 0.01
        }
    else:
        learning_rate = 5e-6
        lr_scheduler = 'cosine_with_min_lr'
        lr_scheduler_kwargs = {
            'min_lr_rate': 0.01
        }
    if 'qwen' in model_id.lower() or \
        model_architecture in ['Qwen2ForCausalLM', 'Qwen2VLForCausalLM'] or \
        model_conf.get('model_type', None) in ['qwen', 'qwen2', 'qwen2_vl', 'qwen2_5_vl']:
        logger.info(f'{model_id} is Qwen model, increase  learning rate')
        learning_rate *= 1.5

    #################### customize lora
    lora_r = 256 if model_size < my_cst.LARGE_MODEL_SIZE else 128
    lora_alpha = 512 if model_size < my_cst.LARGE_MODEL_SIZE else 256
    optimizer = 'adamw_bnb_8bit'

    config['adapter'] = adapter
    config['micro_batch_size'] = micro_batch_size
    config['eval_batch_size'] = eval_batch_size
    config['gradient_checkpointing'] = gradient_checkpointing
    config['max_steps'] = max_steps
    config['flash_attention'] = flash_attention
    # config['eager_attention'] = eager_attention
    # config['early_stopping_patience'] = early_stopping_patience
    # if not_multiplied_estimated_epochs >= 7:
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

        if model_size < my_cst.LARGE_MODEL_SIZE:
            config['peft_init_lora_weights'] ='eva'
            config['lora_target_modules'] = '(?:language_model.)?model.layers.[\d]+.(mlp|cross_attn|self_attn).(up|down|gate|q|k|v|o)_proj'

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
   
    config['num_epochs'] = 10
    config['lora_dropout'] = lora_dropout
    if 'pythia' in model_id and model_architecture is not None and model_architecture == 'GPTNeoXForCausalLM':
        lora_fan_in_fan_out = True
        config['lora_fan_in_fan_out'] = lora_fan_in_fan_out
    
    logger.info(config)
    
    return config