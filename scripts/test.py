import sys
from pathlib import Path
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from vps.core import VisualPositioningSystem
import yaml
import time
import os
# 指定你的图片文件夹路径
# img_dir = Path("/home/phw/newdisk1/vpr_data/office619/ref")

# # 支持的图片后缀
# img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# for img_path in img_dir.iterdir():
#     if img_path.suffix.lower() in img_exts and img_path.stem.isdigit():
#         new_name = f"{int(img_path.stem) + 500}{img_path.suffix}"
#         new_path = img_path.with_name(new_name)
#         print(f"{img_path.name} -> {new_path.name}")
#         os.rename(img_path, new_path)

input_path = Path("/home/phw/newdisk1/vpr_data/office619/sift/images.txt")
output_path = Path("/home/phw/newdisk1/vpr_data/office619/sift/images_query.txt")

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
        # 去除行尾空白符再判断
        if line.strip().endswith(".jpg"):
            fout.write(line)
        else:
            fout.write("\n")  # 写入空白行

