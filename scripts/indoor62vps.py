import shutil
import numpy as np
from pathlib import Path

# 设置源路径和目标路径
source_dir = Path("/home/phw/newdisk1/vpr_data/indoor-6/scene1/images")             # 包含图像和原始pose文件的目录
target_image_dir = Path("data/ref/rgb") # 图像目标目录
target_pose_dir = Path("data/ref/poses")   # 处理后pose目标目录
query_dir = Path("data/query")
pair = Path("/home/phw/visual-localization/VPS/data/scene1_test.txt")

# 创建目标目录（如果不存在）
target_image_dir.mkdir(parents=True, exist_ok=True)
target_pose_dir.mkdir(parents=True, exist_ok=True)
query_dir.mkdir(parents=True, exist_ok=True)

# 遍历所有图像文件
for image_file in sorted(source_dir.glob("image-*.color.jpg")):
    number = image_file.stem.split("-")[1].split(".")[0]
    pose_file = source_dir / f"image-{number}.pose.txt"

    if pose_file.exists():
        # 生成新的文件名（移除.color后缀）
        new_image_name = f"image-{number}.jpg"
        # 复制图像文件
        shutil.copy(image_file, target_image_dir / new_image_name)

        # 读取 3x4 pose 矩阵
        pose_matrix = np.loadtxt(pose_file)
        if pose_matrix.shape == (3, 4):
            # 构造 4x4 矩阵
            bottom_row = np.array([[0, 0, 0, 1]])
            pose_matrix_4x4 = np.vstack([pose_matrix, bottom_row])
            pose_matrix_4x4 = np.linalg.inv(pose_matrix_4x4)
            # 生成新的pose文件名（移除.pose后缀）
            new_pose_name = f"image-{number}.txt"
            # 保存新的pose文件
            target_pose_path = target_pose_dir / new_pose_name
            np.savetxt(target_pose_path, pose_matrix_4x4, fmt="%.6f")
            print(f"✓ Copied image and converted pose: {image_file.name} -> {new_image_name}, {pose_file.name} -> {new_pose_name}")
        else:
            print(f"✗ Invalid pose shape in: {pose_file.name}, expected 3x4.")
    else:
        print(f"✗ Pose file not found for image: {image_file.name}")

# 读取测试文件并移动图像
print("\n开始处理测试集图像...")
if pair.exists():
    with open(pair, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]
    
    moved_count = 0
    for image_name in test_images:
        # 移除.color后缀（如果存在）
        if image_name.endswith('.color.jpg'):
            clean_name = image_name.replace('.color.jpg', '.jpg')
        else:
            clean_name = image_name
            
        source_path = target_image_dir / clean_name
        target_path = query_dir / clean_name
        
        if source_path.exists():
            # 移动文件到query目录
            shutil.move(str(source_path), str(target_path))
            print(f"✓ Moved: {clean_name} -> query/")
            moved_count += 1
        else:
            print(f"✗ Image not found: {clean_name}")
    
    print(f"\n总共移动了 {moved_count} 个图像文件到query目录")
else:
    print(f"✗ Test file not found: {pair}")
