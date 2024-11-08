"""图像处理工具函数"""
import torch
import torch.nn.functional as F

def resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    """带抗锯齿的图像缩放
    Args:
        input: 输入图像
        size: 目标尺寸
        interpolation: 插值方法
        align_corners: 是否对齐角点
    Returns:
        缩放后的图像
    """
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = gaussian_blur2d(input, ks, sigmas)

    output = F.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def gaussian_blur2d(input, kernel_size, sigma):
    """2D高斯模糊
    Args:
        input: 输入图像
        kernel_size: 核大小
        sigma: 标准差
    Returns:
        模糊后的图像
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = kernel_size
    bs = sigma.shape[0]
    kernel_x = gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = gaussian(ky, sigma[:, 0].view(bs, 1))
    
    out_x = filter2d(input, kernel_x[..., None, :])
    out = filter2d(out_x, kernel_y[..., None])
    return out

# 其他辅助函数...