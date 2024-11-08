"""数据集相关代码"""
import os
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """视频数据集类"""
    def __init__(self, base_folder: str, width=1024, height=576, sample_frames=25):
        """
        初始化数据集
        Args:
            base_folder (str): 数据集根目录
            width (int): 图像宽度
            height (int): 图像高度 
            sample_frames (int): 采样帧数
        """
        logger.info(f"初始化数据集，基础目录: {base_folder}")
        
        self.base_folder = base_folder
        # 获取并排序文件夹列表
        self.folders = sorted(os.listdir(self.base_folder))
        logger.info(f"找到 {len(self.folders)} 个视频文件夹")
        
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        
        # 验证所有文件夹的帧数
        logger.info("验证视频文件夹...")
        valid_folders = []
        for folder in tqdm(self.folders, desc="验证视频文件夹"):
            frames = sorted(os.listdir(os.path.join(self.base_folder, folder)))
            if len(frames) >= self.sample_frames:
                valid_folders.append(folder)
        
        self.folders = valid_folders
        logger.info(f"有效视频文件夹数量: {len(self.folders)}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.folders)

    def __getitem__(self, idx):
        """获取数据样本
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含视频帧数据的字典
        """
        # 按索引顺序选择文件夹
        chosen_folder = self.folders[idx]
        folder_path = os.path.join(self.base_folder, chosen_folder)
        
        # 获取并排序帧文件列表
        frames = sorted(os.listdir(folder_path))

        if len(frames) < self.sample_frames:
            raise ValueError(f"视频'{chosen_folder}'帧数少于{self.sample_frames}帧")

        # 从第一帧开始采样固定数量的帧
        selected_frames = frames[:self.sample_frames]

        # 初始化tensor存储像素值
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))

        # 加载并处理每一帧
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                # 调整图像大小
                img_resized = img.resize((self.width, self.height))
                # 转换为tensor
                img_tensor = torch.from_numpy(np.array(img_resized)).float()
                # 归一化到[-1,1]范围
                img_normalized = img_tensor / 127.5 - 1
                
                # 调整通道顺序
                if self.channels == 3:
                    img_normalized = img_normalized.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(dim=2, keepdim=True)

                pixel_values[i] = img_normalized
        
        if idx % 100 == 0:  # 每100个样本输出一次日志
            logger.info(f"加载第 {idx} 个样本，从文件夹 {chosen_folder}")
                
        return {'pixel_values': pixel_values}