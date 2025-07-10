import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, image_format='jpg'):
    """
    从视频中逐帧提取图像并保存。

    Args:
        video_path (str): 输入视频文件的路径。
        output_dir (str): 保存提取帧的目录。
        image_format (str, optional): 保存图像的格式 (例如 'jpg', 'png')。 默认为 'jpg'。
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频文件
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    print(f"开始从 {video_path} 提取帧...")
    
    frame_count = 0
    success = True
    while success:
        # 读取一帧
        success, frame = video_capture.read()
        
        if success:
            # 构造输出文件名，例如: frame_00000.jpg, frame_00001.jpg, ...
            frame_filename = f"frame_{frame_count:05d}.{image_format}"
            output_path = output_dir / frame_filename
            
            # 保存帧
            cv2.imwrite(str(output_path), frame)
            
            frame_count += 1

    video_capture.release()
    print(f"提取完成！总共提取了 {frame_count} 帧图像到目录: {output_dir}")


if __name__ == '__main__':
    # --- 配置 ---
    input_video_path = Path("/home/phw/visual-localization/VPS/outputs/rgb/rgb.mp4")  # <--- 在这里修改你的视频文件路径
    output_frames_dir = Path("/home/phw/visual-localization/VPS/outputs/rgb")          # <--- 在这里修改你的输出目录
    # ------------

    if not input_video_path.exists():
        print(f"错误: 视频文件不存在于 {input_video_path}")
    else:
        extract_frames(input_video_path, output_frames_dir)

