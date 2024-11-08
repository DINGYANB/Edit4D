import torch
import argparse
import logging
from svd_lora.configs.config_loader import Config
from svd_lora.data.dataset import VideoDataset
from svd_lora.trainer.trainer import SVDTrainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Stable Video Diffusion Training')
    parser.add_argument('--config', type=str, default='./svd_lora/configs/default.yaml',
                      help='配置文件路径')
    parser.add_argument('--base_folder', type=str, help='训练数据目录')
    parser.add_argument('--pretrained_model_name_or_path', type=str, help='预训练模型路径')
    
    return parser.parse_args()

def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 设置日志
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger = logging.getLogger(__name__)
        
        logger.info("开始加载配置文件...")
        config = Config.from_yaml(args.config)
        
        # 用命令行参数更新配置
        config.update_from_args(args)
        logger.info(f"数据目录: {config.data.base_folder}")
        logger.info(f"模型路径: {config.model.pretrained_model_name_or_path}")
        
        # 设置随机种子
        if config.training.seed is not None:
            set_seed(config.training.seed)
            logger.info(f"设置随机种子: {config.training.seed}")
            
        # 创建数据集
        logger.info("开始创建数据集...")
        train_dataset = VideoDataset(
            config.data.base_folder,
            width=config.data.width,
            height=config.data.height,
            sample_frames=config.data.num_frames
        )
        logger.info(f"数据集大小: {len(train_dataset)}")
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.training.per_gpu_batch_size,
            num_workers=config.data.num_workers,
        )
        
        # 创建训练器并开始训练
        logger.info("初始化训练器...")
        trainer = SVDTrainer(config)
        logger.info("开始训练...")
        trainer.train(train_dataloader)
        
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()