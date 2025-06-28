from flask import Flask, request, jsonify
from pathlib import Path
import tempfile
import os
import numpy as np
from core import VisualPositioningSystem
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global VPS instance
vps = None

def create_app(config_path: str = "configs/default.yaml"):
    """Create and configure the Flask application."""
    global vps
    
    # Initialize VPS system
    vps = VisualPositioningSystem(config_path)
    logger.info("VPS system initialized successfully")
    
    return app

def transform_matrix_to_pose_2d(transform_matrix):
    """
    将4x4转换矩阵转换为2D位姿 (x, y, theta)
    
    Args:
        transform_matrix (np.ndarray): 4x4转换矩阵
        
    Returns:
        dict: 包含x, y, theta的字典
    """
    if transform_matrix is None:
        return None
    
    # 提取平移部分 (x, y, z)
    translation = transform_matrix[:3, 3]
    x, y, z = translation[0], translation[1], translation[2]
    
    # 提取旋转矩阵 (3x3)
    rotation_matrix = transform_matrix[:3, :3]
    
    # 从旋转矩阵计算yaw角 (绕z轴的旋转)
    # 使用atan2(R[1,0], R[0,0])来计算yaw角
    theta = np.arctan2(rotation_matrix[0, 0], -1*rotation_matrix[2, 0])
    
    # 转换为度数
    theta_degrees = np.degrees(theta)
    
    return {
        'x': float(x),
        'y': float(z),
        # 'z': float(z),
        # 'theta': float(theta_degrees),  # 以度为单位
        'theta': float(theta),      # 以弧度为单位
        # 'z': float(z)                   # 保留z值供参考
    }

def map2map(x, y, theta):
    scale = 1.0104234599509834
    R = np.array([[ 0.96054926, -0.2781099 ],[ 0.2781099, 0.96054926]])
    t = np.array([ 0.8427085, -5.45116856])
    src_pos = np.array([x, y])
    tgt_pos = scale * (R @ src_pos) + t

    dir_vec = np.array([np.cos(theta), np.sin(theta)])  
    new_dir = R @ dir_vec
    new_theta = np.arctan2(new_dir[1], new_dir[0])

    return tgt_pos[0], tgt_pos[1], new_theta
@app.route('/localize', methods=['POST'])
def localize():
    """Localization endpoint."""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        rgb_file = request.files['image']
        print(f"rgb_file: {rgb_file.filename}")
        if rgb_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        # Check file format
        allowed_extensions = vps.config['service']['supported_formats']
        if not any(rgb_file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': f'Unsupported file format. Supported: {allowed_extensions}'}), 400
        # Create temp directory if it doesn't exist
        os.makedirs(vps.config['service']['temp_dir'], exist_ok=True)
        query_image_path = os.path.join(vps.config['service']['temp_dir'], vps.config['service']['temp_rgb_name'])
        rgb_file.save(query_image_path)
        ##
        ##
        # 处理可选的深度文件
        depth_path = None
        if 'depth' in request.files:
            depth_file = request.files['depth']
            
            if depth_file and depth_file.filename:
                print(f"接收到深度文件: {depth_file.filename}")
                file_ext = os.path.splitext(depth_file.filename)[1].lower()
                depth_path = os.path.join(vps.config['service']['temp_dir'], vps.config['service']['temp_depth_name'])

                if file_ext == '.png':
                    # 从文件流读取、解码、转换并保存
                    filestr = depth_file.read()
                    npimg = np.frombuffer(filestr, np.uint8)
                    depth_data_png = cv2.imdecode(npimg, cv2.IMREAD_ANYDEPTH)
                    if depth_data_png is not None:
                        depth_data = depth_data_png.astype(np.float32) / 1000.0
                        np.save(depth_path, depth_data)
                        print(f"PNG深度文件已转换为NPY并保存到: {depth_path}")

                elif file_ext == '.npy':
                    # 直接保存npy文件
                    depth_file.save(depth_path)
                    print(f"NPY深度文件已保存到: {depth_path}")
        
        # 执行定位 (depth_path可能是None)
        pose3d= vps.localize(query_image_path,depth_path)
        
        # Convert numpy arrays to lists for JSON serialization
        if pose3d is not None:
            # 如果result包含pose字段且是4x4矩阵
            # 将4x4矩阵转换为x, y, theta格式
            pose_2d = transform_matrix_to_pose_2d(pose3d)
            ans = map2map(x= pose_2d['x'], y= pose_2d['y'], theta= pose_2d['theta'])
            ans = {
                'x': ans[0],
                'y': ans[1],
                'theta': ans[2]
            }
            print(f"ans: {ans}")
            return jsonify(ans), 200
        else:
            return jsonify({'error': 'No pose found'}), 400
    except Exception as e:
        logger.error(f"Error during localization: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False) 