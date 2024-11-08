"""训练器类"""
import logging
import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate.utils import set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import (
    StableVideoDiffusionPipeline,
    AutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModel,
    DDIMScheduler
)
from peft import LoraConfig, get_peft_model

from ..models.model_utils import tensor_to_vae_latent, get_add_time_ids, rand_log_normal
from ..utils.train_utils import setup_accelerator, save_model_checkpoint
from ..utils.video_utils import export_to_gif

logger = logging.getLogger(__name__)

class SVDTrainer:
    """Stable Video Diffusion训练器"""
    
    def __init__(self, args):
        """初始化训练器
        Args:
            args: 训练参数
        """
        logger.info("初始化训练器...")
        self.args = args
        logger.info("设置accelerator...")
        self.accelerator = setup_accelerator(args)
        logger.info("设置模型...")
        self.setup_model()
        
    def setup_model(self):
        """设置模型"""
        args = self.args
        
        # 加载模型组件
        logger.info("加载feature_extractor...")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            args.model.pretrained_model_name_or_path,
            subfolder="feature_extractor",
            revision=args.model.revision,
            local_files_only=True
        )
        
        logger.info("加载image_encoder...")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.model.pretrained_model_name_or_path,
            subfolder="image_encoder",
            revision=args.model.revision,
            local_files_only=True
        )
        
        logger.info("加载vae...")
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.model.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.model.revision,
            variant="fp16",
            local_files_only=True
        )
        
        logger.info("加载unet...")
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.model.pretrained_model_name_or_path if args.model.pretrain_unet is None else args.model.pretrain_unet,
            subfolder="unet",
            revision=args.model.revision,
            variant="fp16",
            local_files_only=True
        )
        
        # 加载噪声调度器
        logger.info("加载噪声调度器...")
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.model.pretrained_model_name_or_path,
            subfolder="scheduler",
            local_files_only=True
        )
        
        # 设置LoRA
        logger.info("设置LoRA配置...")
        self.setup_lora_config()
        
        # 将模型移动到对应设备
        logger.info(f"将模型移动到设备: {self.accelerator.device}")
        self.image_encoder = self.image_encoder.to(self.accelerator.device)
        self.vae = self.vae.to(self.accelerator.device)
        self.unet = self.unet.to(self.accelerator.device)
        
        logger.info("模型加载完成")
        
    def setup_lora_config(self):
        """设置LoRA配置"""
        lora_config = LoraConfig(
            r=self.args.model.rank,
            lora_alpha=self.args.model.rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        logger.info(f"LoRA配置: rank={self.args.model.rank}")
        self.unet = get_peft_model(self.unet, lora_config)
        
    def train(self, train_dataloader):
        """训练模型
        Args:
            train_dataloader: 训练数据加载器
        """
        logger.info("开始训练循环...")
        
        # 设置模型为训练模式
        self.unet.train()
        self.vae.eval()
        self.image_encoder.eval()
        
        # 配置优化器
        logger.info("配置优化器...")
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.args.optimizer.learning_rate,
            betas=(self.args.optimizer.adam_beta1, self.args.optimizer.adam_beta2),
            weight_decay=self.args.optimizer.adam_weight_decay,
            eps=self.args.optimizer.adam_epsilon,
        )
        
        # 将模型和优化器包装到accelerator
        self.unet, optimizer, train_dataloader = self.accelerator.prepare(
            self.unet, optimizer, train_dataloader
        )
        
        # 确保其他模型也在正确的设备上
        self.vae = self.accelerator.prepare_model(self.vae, evaluation_mode=True)
        self.image_encoder = self.accelerator.prepare_model(self.image_encoder, evaluation_mode=True)
        
        # 计算训练步数
        total_batch_size = self.args.training.per_gpu_batch_size * self.accelerator.num_processes
        num_update_steps_per_epoch = len(train_dataloader)
        num_train_epochs = self.args.training.num_train_epochs
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        
        logger.info("***** 开始训练 *****")
        logger.info(f"  每个GPU的批次大小 = {self.args.training.per_gpu_batch_size}")
        logger.info(f"  总批次大小 = {total_batch_size}")
        logger.info(f"  训练总步数 = {max_train_steps}")
        logger.info(f"  每轮更新步数 = {num_update_steps_per_epoch}")
        logger.info(f"  训练轮数 = {num_train_epochs}")
        
        global_step = 0
        first_epoch = 0
        
        progress_bar = tqdm(
            total=max_train_steps,
            disable=not self.accelerator.is_local_main_process,
            position=0
        )
        progress_bar.set_description("训练进度")
        
        for epoch in range(first_epoch, num_train_epochs):
            logger.info(f"开始第 {epoch+1}/{num_train_epochs} 轮训练")
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # 获取输入图像
                    pixel_values = batch["pixel_values"].to(self.accelerator.device)
                    
                    # 将图像编码为潜在表示
                    with torch.no_grad():
                        latents = tensor_to_vae_latent(pixel_values, self.vae)
                        image_embeddings = self.image_encoder(pixel_values[:, 0]).image_embeds
                    
                    # 添加噪声
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, 
                        self.noise_scheduler.config.num_train_timesteps, 
                        (latents.shape[0],), 
                        device=latents.device
                    )
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # 预测噪声
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        image_embeddings,
                    ).sample
                    
                    # 计算损失
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.unet.parameters(),
                            self.args.optimizer.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 更新进度条
                progress_bar.update(1)
                global_step += 1
                
                # 记录训练信息
                if global_step % 10 == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)
                    logger.info(
                        f"Epoch {epoch+1}, "
                        f"Step {step}/{len(train_dataloader)}, "
                        f"Loss: {loss.detach().item():.4f}"
                    )
                
                # 验证和保存模型
                if global_step % self.args.training.validation_steps == 0:
                    self.validate()
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        save_model_checkpoint(
                            self.unet,
                            self.args.output.output_dir,
                            global_step
                        )
        
        progress_bar.close()
        logger.info("训练完成")
        
    def validate(self):
        """验证模型"""
        logger.info("开始验证...")
        self.unet.eval()
        
        with torch.no_grad():
            # 这里添加验证逻辑
            # 可以生成一些视频样本并保存
            pass
        
        self.unet.train()
        logger.info("验证完成")