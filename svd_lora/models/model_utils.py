"""模型相关工具函数"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union

def tensor_to_vae_latent(t: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """将tensor转换为VAE潜在表示
    
    Args:
        t (torch.Tensor): 输入tensor，形状为 [B, T, C, H, W]
        vae (torch.nn.Module): VAE模型
        
    Returns:
        torch.Tensor: VAE潜在表示，形状为 [B, T, C', H', W']
        
    Example:
        >>> t = torch.randn(2, 16, 3, 256, 256)  # [batch_size, num_frames, channels, height, width]
        >>> latents = tensor_to_vae_latent(t, vae)
        >>> print(latents.shape)  # [2, 16, 4, 32, 32]
    """
    video_length = t.shape[1]
    
    # 重塑为 [B*T, C, H, W] 用于批处理编码
    t = t.reshape((-1,) + t.shape[2:])
    
    # VAE编码
    with torch.no_grad():
        latents = vae.encode(t).latent_dist.sample()
    
    # 重塑回 [B, T, C, H, W] 格式
    latents = latents.reshape((-1, video_length) + latents.shape[1:])
    
    # 应用缩放因子
    latents = latents * vae.config.scaling_factor
    
    return latents

def get_add_time_ids(
    fps: int, 
    motion_bucket_id: int, 
    noise_aug_strength: float,
    dtype: torch.dtype,
    batch_size: int,
    unet: torch.nn.Module
) -> torch.Tensor:
    """获取时间ID嵌入
    
    Args:
        fps (int): 帧率
        motion_bucket_id (int): 运动桶ID
        noise_aug_strength (float): 噪声增强强度
        dtype (torch.dtype): 数据类型
        batch_size (int): 批次大小
        unet (torch.nn.Module): UNet模型
        
    Returns:
        torch.Tensor: 时间ID嵌入张量
        
    Raises:
        ValueError: 当创建的嵌入维度与模型期望的维度不匹配时
        
    Example:
        >>> add_time_ids = get_add_time_ids(
        ...     fps=24,
        ...     motion_bucket_id=127,
        ...     noise_aug_strength=0.1,
        ...     dtype=torch.float32,
        ...     batch_size=2,
        ...     unet=unet_model
        ... )
    """
    # 创建时间ID列表
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    
    # 计算并验证嵌入维度
    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"模型期望的时间嵌入向量长度为{expected_add_embed_dim}, "
            f"但创建的向量长度为{passed_add_embed_dim}。"
        )

    # 创建并复制时间ID张量
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    
    return add_time_ids

def rand_log_normal(
    shape: Union[Tuple[int, ...], torch.Size],
    loc: float = 0.,
    scale: float = 1.,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """生成对数正态分布随机数
    
    Args:
        shape (Union[Tuple[int, ...], torch.Size]): 输出张量的形状
        loc (float, optional): 位置参数. 默认: 0.
        scale (float, optional): 尺度参数. 默认: 1.
        device (Union[str, torch.device], optional): 设备. 默认: 'cpu'
        dtype (torch.dtype, optional): 数据类型. 默认: torch.float32
        
    Returns:
        torch.Tensor: 服从对数正态分布的随机数张量
        
    Example:
        >>> samples = rand_log_normal((2, 3), loc=0, scale=1)
        >>> print(samples.shape)  # torch.Size([2, 3])
    """
    # 生成均匀分布随机数并限制在合理范围内
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    
    # 使用反函数方法生成对数正态分布
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def get_motion_unet_kwargs(
    motion_bucket_id: int,
    fps: int = 12,
    noise_aug_strength: float = 0.0,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    unet: Optional[torch.nn.Module] = None
) -> dict:
    """获取运动UNet的关键字参数
    
    Args:
        motion_bucket_id (int): 运动桶ID
        fps (int, optional): 帧率. 默认: 12
        noise_aug_strength (float, optional): 噪声增强强度. 默认: 0.0
        dtype (torch.dtype, optional): 数据类型. 默认: torch.float32
        batch_size (int, optional): 批次大小. 默认: 1
        unet (Optional[torch.nn.Module], optional): UNet模型. 默认: None
        
    Returns:
        dict: UNet的关键字参数
    """
    if unet is None:
        return {}
        
    add_time_ids = get_add_time_ids(
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        dtype=dtype,
        batch_size=batch_size,
        unet=unet
    )
    
    return {"add_time_ids": add_time_ids}