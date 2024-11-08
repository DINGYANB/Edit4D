"""配置加载器"""
import os
import yaml
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class DataConfig:
    base_folder: Optional[str]
    num_frames: int
    width: int
    height: int
    num_workers: int

@dataclass
class ModelConfig:
    pretrained_model_name_or_path: Optional[str]
    revision: Optional[str]
    pretrain_unet: Optional[str]
    rank: int

@dataclass
class TrainingConfig:
    seed: Optional[int]
    num_validation_images: int
    validation_steps: int
    per_gpu_batch_size: int
    num_train_epochs: int
    max_train_steps: Optional[int]
    gradient_accumulation_steps: int
    gradient_checkpointing: bool

@dataclass
class OptimizerConfig:
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    conditioning_dropout_prob: float
    use_8bit_adam: bool
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: float

@dataclass
class SystemConfig:
    allow_tf32: bool
    mixed_precision: Optional[str]
    enable_xformers_memory_efficient_attention: bool

@dataclass
class OutputConfig:
    output_dir: str
    logging_dir: str
    report_to: str
    checkpointing_steps: int
    checkpoints_total_limit: int
    resume_from_checkpoint: Optional[str]

@dataclass
class HubConfig:
    push_to_hub: bool
    hub_token: Optional[str]
    hub_model_id: Optional[str]

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    system: SystemConfig
    output: OutputConfig
    hub: HubConfig

    def __getattr__(self, name):
        """兼容旧式访问方式"""
        # 在所有配置类中查找属性
        for config_section in [self.data, self.model, self.training, 
                             self.optimizer, self.system, self.output, self.hub]:
            if hasattr(config_section, name):
                return getattr(config_section, name)
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 确保optimizer配置中的数值类型正确
        optimizer_dict = config_dict['optimizer']
        optimizer_dict['learning_rate'] = float(optimizer_dict['learning_rate'])
        optimizer_dict['adam_beta1'] = float(optimizer_dict['adam_beta1'])
        optimizer_dict['adam_beta2'] = float(optimizer_dict['adam_beta2'])
        optimizer_dict['adam_weight_decay'] = float(optimizer_dict['adam_weight_decay'])
        optimizer_dict['adam_epsilon'] = float(optimizer_dict['adam_epsilon'])
        optimizer_dict['max_grad_norm'] = float(optimizer_dict['max_grad_norm'])
        optimizer_dict['conditioning_dropout_prob'] = float(optimizer_dict['conditioning_dropout_prob'])
        optimizer_dict['lr_warmup_steps'] = int(optimizer_dict['lr_warmup_steps'])

        # 确保training配置中的数值类型正确
        training_dict = config_dict['training']
        if training_dict['seed'] is not None:
            training_dict['seed'] = int(training_dict['seed'])
        training_dict['num_validation_images'] = int(training_dict['num_validation_images'])
        training_dict['validation_steps'] = int(training_dict['validation_steps'])
        training_dict['per_gpu_batch_size'] = int(training_dict['per_gpu_batch_size'])
        training_dict['num_train_epochs'] = int(training_dict['num_train_epochs'])
        if training_dict['max_train_steps'] is not None:
            training_dict['max_train_steps'] = int(training_dict['max_train_steps'])
        training_dict['gradient_accumulation_steps'] = int(training_dict['gradient_accumulation_steps'])

        # 确保data配置中的数值类型正确
        data_dict = config_dict['data']
        data_dict['num_frames'] = int(data_dict['num_frames'])
        data_dict['width'] = int(data_dict['width'])
        data_dict['height'] = int(data_dict['height'])
        data_dict['num_workers'] = int(data_dict['num_workers'])

        # 确保model配置中的数值类型正确
        model_dict = config_dict['model']
        model_dict['rank'] = int(model_dict['rank'])

        # 确保output配置中的数值类型正确
        output_dict = config_dict['output']
        output_dict['checkpointing_steps'] = int(output_dict['checkpointing_steps'])
        output_dict['checkpoints_total_limit'] = int(output_dict['checkpoints_total_limit'])

        return cls(
            data=DataConfig(**data_dict),
            model=ModelConfig(**model_dict),
            training=TrainingConfig(**training_dict),
            optimizer=OptimizerConfig(**optimizer_dict),
            system=SystemConfig(**config_dict['system']),
            output=OutputConfig(**output_dict),
            hub=HubConfig(**config_dict['hub'])
        )

    def update_from_args(self, args: Any) -> None:
        """从命令行参数更新配置"""
        for category in ['data', 'model', 'training', 'optimizer', 'system', 'output', 'hub']:
            config_category = getattr(self, category)
            for field in config_category.__dataclass_fields__:
                if hasattr(args, field) and getattr(args, field) is not None:
                    value = getattr(args, field)
                    
                    # 根据字段类型进行类型转换
                    field_type = config_category.__dataclass_fields__[field].type
                    if field_type == float:
                        value = float(value)
                    elif field_type == int:
                        value = int(value)
                    elif field_type == bool:
                        value = bool(value)
                    
                    setattr(config_category, field, value)