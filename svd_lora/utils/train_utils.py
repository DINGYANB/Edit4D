"""训练相关工具函数"""
import os
import shutil
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict

def setup_accelerator(args):
    """设置accelerator
    Args:
        args: 训练参数
    Returns:
        accelerator: Accelerator实例
    """
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    return accelerator

def save_model_checkpoint(accelerator, unet, args, global_step):
    """保存模型检查点
    Args:
        accelerator: Accelerator实例
        unet: UNet模型
        args: 训练参数
        global_step: 当前步数
    """
    if args.checkpoints_total_limit is not None:
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            for removing_checkpoint in removing_checkpoints:
                ckpt_path = os.path.join(args.output_dir, removing_checkpoint)
                shutil.rmtree(ckpt_path)

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)

    unwrapped_unet = accelerator.unwrap_model(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unwrapped_unet)
    )

    StableVideoDiffusionPipeline.save_lora_weights(
        save_directory=save_path,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )