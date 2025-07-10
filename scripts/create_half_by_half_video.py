#!/usr/bin/env python3
"""
Create half-by-half diagonal split video from two image folders
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_matching_images(folder_a: Path, folder_b: Path) -> List[Tuple[Path, Path]]:
    """
    获取两个文件夹中匹配的图像对
    
    Args:
        folder_a: 图像文件夹A
        folder_b: 图像文件夹B
        
    Returns:
        匹配的图像路径对列表
    """
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取文件夹A中的所有图像
    images_a = {}
    for img_path in folder_a.iterdir():
        if img_path.suffix.lower() in image_extensions:
            images_a[img_path.stem] = img_path
    
    # 获取文件夹B中的所有图像
    images_b = {}
    for img_path in folder_b.iterdir():
        if img_path.suffix.lower() in image_extensions:
            images_b[img_path.stem] = img_path
    
    # 找到匹配的图像对
    matched_pairs = []
    for name in sorted(images_a.keys()):
        if name in images_b:
            matched_pairs.append((images_a[name], images_b[name]))
    
    logger.info(f"找到 {len(matched_pairs)} 对匹配图像")
    return matched_pairs

def create_diagonal_mask(width: int, height: int) -> np.ndarray:
    """
    创建对角线掩码，左下角为True，右上角为False
    
    Args:
        width: 图像宽度
        height: 图像高度
        
    Returns:
        对角线掩码数组
    """
    # 创建坐标网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 对角线：y = height - x * (height/width)
    # 左下角：y > height - x * (height/width)
    diagonal_line = height - x * (height / width)
    mask = y > diagonal_line
    
    return mask

def resize_and_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整图像大小并裁剪到目标尺寸，保持纵横比
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        
    Returns:
        调整后的图像
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    # 计算缩放比例，保持纵横比
    scale = max(target_w / w, target_h / h)
    
    # 调整大小
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # 居中裁剪
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]
    
    return cropped

def create_half_by_half_frame(image_a: np.ndarray, image_b: np.ndarray, 
                             target_size: Tuple[int, int],
                             add_labels: bool = False,
                             label_a: str = "A", label_b: str = "B") -> np.ndarray:
    """
    创建对角线分割的合成帧
    
    Args:
        image_a: 图像A（左下角）
        image_b: 图像B（右上角）
        target_size: 目标尺寸 (width, height)
        add_labels: 是否添加标签
        label_a: 图像A的标签
        label_b: 图像B的标签
        
    Returns:
        合成后的帧
    """
    target_w, target_h = target_size
    
    # 调整两个图像到目标尺寸
    img_a_resized = resize_and_crop(image_a, target_size)
    img_b_resized = resize_and_crop(image_b, target_size)
    
    # 创建对角线掩码
    mask = create_diagonal_mask(target_w, target_h)
    
    # 创建结果图像
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 应用掩码：左下角用图像A，右上角用图像B
    result[mask] = img_a_resized[mask]
    result[~mask] = img_b_resized[~mask]
    
    # 添加分割线（可选）
    # 在对角线上画一条细线
    for x in range(target_w):
        y = int(target_h - x * (target_h / target_w))
        if 0 <= y < target_h:
            cv2.line(result, (x, y), (x, y), (255, 255, 255), 1)
    
    # 添加标签
    if add_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 标签A放在左下角
        label_a_pos = (10, target_h - 10)
        cv2.putText(result, label_a, label_a_pos, font, font_scale, 
                   (255, 255, 255), thickness)
        cv2.putText(result, label_a, label_a_pos, font, font_scale, 
                   (0, 0, 0), 1)  # 黑色边框
        
        # 标签B放在右上角
        text_size = cv2.getTextSize(label_b, font, font_scale, thickness)[0]
        label_b_pos = (target_w - text_size[0] - 10, 25)
        cv2.putText(result, label_b, label_b_pos, font, font_scale, 
                   (255, 255, 255), thickness)
        cv2.putText(result, label_b, label_b_pos, font, font_scale, 
                   (0, 0, 0), 1)  # 黑色边框
    
    return result

def create_half_by_half_video(folder_a: Path, folder_b: Path, output_path: Path,
                             fps: int = 10, target_size: Tuple[int, int] = (848, 480),
                             add_labels: bool = True, label_a: str = "A", label_b: str = "B") -> bool:
    """
    创建对角线分割视频
    
    Args:
        folder_a: 图像文件夹A路径
        folder_b: 图像文件夹B路径
        output_path: 输出视频路径
        fps: 帧率
        target_size: 视频尺寸 (width, height)
        add_labels: 是否添加标签
        label_a: 图像A的标签
        label_b: 图像B的标签
        
    Returns:
        是否成功创建视频
    """
    try:
        # 获取匹配的图像对
        image_pairs = get_matching_images(folder_a, folder_b)
        
        if not image_pairs:
            logger.error("没有找到匹配的图像对")
            return False
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, target_size)
        
        logger.info(f"开始创建视频，共 {len(image_pairs)} 帧")
        
        for i, (img_a_path, img_b_path) in enumerate(image_pairs):
            # 读取图像
            img_a = cv2.imread(str(img_a_path))
            img_b = cv2.imread(str(img_b_path))
            
            if img_a is None or img_b is None:
                logger.warning(f"无法读取图像对: {img_a_path}, {img_b_path}")
                continue
            
            # 创建合成帧
            frame = create_half_by_half_frame(
                img_a, img_b, target_size, add_labels, label_a, label_b
            )
            
            # 写入视频
            out.write(frame)
            
            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i + 1}/{len(image_pairs)} 帧")
        
        # 释放资源
        out.release()
        
        logger.info(f"视频创建完成: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"创建视频时出错: {e}")
        return False

if __name__ == "__main__":
    # 测试代码
    folder_a = Path("test_images_a")
    folder_b = Path("test_images_b") 
    output_path = Path("test_half_by_half.mp4")
    
    success = create_half_by_half_video(
        folder_a=folder_a,
        folder_b=folder_b,
        output_path=output_path,
        fps=10,
        target_size=(848, 480),
        add_labels=True,
        label_a="Query",
        label_b="Result"
    )
    
    if success:
        print("✅ 对角线分割视频创建成功!")
    else:
        print("❌ 视频创建失败!") 