"""视频处理工具函数"""
import cv2
from PIL import Image
import numpy as np

def export_to_video(video_frames, output_video_path, fps):
    """导出视频文件
    Args:
        video_frames: 视频帧列表
        output_video_path: 输出路径
        fps: 帧率
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for frame in video_frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def export_to_gif(frames, output_gif_path, fps):
    """导出GIF文件
    Args:
        frames: 帧列表
        output_gif_path: 输出路径
        fps: 帧率
    """
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    pil_frames[0].save(
        output_gif_path.replace('.mp4', '.gif'),
        format='GIF',
        append_images=pil_frames[1:],
        save_all=True,
        duration=500,
        loop=0
    )