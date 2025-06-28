#!/usr/bin/env python3
"""
测试4x4转换矩阵到x,y,theta格式的转换功能
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vps.service import transform_matrix_to_pose_2d

def create_test_transform_matrix(x=1.0, y=2.0, z=0.5, theta_degrees=45.0):
    """
    创建一个测试用的4x4转换矩阵
    
    Args:
        x, y, z: 平移分量
        theta_degrees: 绕z轴的旋转角度（度）
    
    Returns:
        np.ndarray: 4x4转换矩阵
    """
    # 将角度转换为弧度
    theta_rad = np.radians(theta_degrees)
    
    # 创建旋转矩阵（绕z轴）
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])
    
    # 创建4x4转换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [x, y, z]
    
    return transform_matrix

def test_pose_conversion():
    """测试位姿转换功能"""
    
    print("=" * 60)
    print("测试4x4转换矩阵到x,y,theta格式的转换")
    print("=" * 60)
    
    # 测试用例1: 简单的平移和旋转
    print("\n测试用例1: 简单平移和旋转")
    print("-" * 40)
    
    # 创建测试矩阵：x=1.0, y=2.0, z=0.5, theta=45度
    transform_matrix1 = create_test_transform_matrix(x=1.0, y=2.0, z=0.5, theta_degrees=45.0)
    
    print("原始4x4转换矩阵:")
    print(transform_matrix1)
    
    pose_2d_1 = transform_matrix_to_pose_2d(transform_matrix1)
    print("\n转换后的2D位姿:")
    print(f"x: {pose_2d_1['x']:.3f}")
    print(f"y: {pose_2d_1['y']:.3f}")
    print(f"z: {pose_2d_1['z']:.3f}")
    print(f"theta (度): {pose_2d_1['theta']:.3f}")
    print(f"theta (弧度): {pose_2d_1['theta_rad']:.3f}")
    
    # 验证结果
    expected_x, expected_y, expected_z = 1.0, 2.0, 0.5
    expected_theta = 45.0
    
    print(f"\n验证结果:")
    print(f"x 误差: {abs(pose_2d_1['x'] - expected_x):.6f}")
    print(f"y 误差: {abs(pose_2d_1['y'] - expected_y):.6f}")
    print(f"z 误差: {abs(pose_2d_1['z'] - expected_z):.6f}")
    print(f"theta 误差: {abs(pose_2d_1['theta'] - expected_theta):.6f}")
    
    # 测试用例2: 不同的角度
    print("\n\n测试用例2: 不同角度")
    print("-" * 40)
    
    angles_to_test = [0, 90, 180, 270, -45, -90]
    
    for angle in angles_to_test:
        transform_matrix = create_test_transform_matrix(x=0.0, y=0.0, z=0.0, theta_degrees=angle)
        pose_2d = transform_matrix_to_pose_2d(transform_matrix)
        
        print(f"角度 {angle:3d}° -> theta: {pose_2d['theta']:6.1f}°")
    
    # 测试用例3: 边界情况
    print("\n\n测试用例3: 边界情况")
    print("-" * 40)
    
    # 测试None输入
    result_none = transform_matrix_to_pose_2d(None)
    print(f"None输入: {result_none}")
    
    # 测试零矩阵
    zero_matrix = np.zeros((4, 4))
    zero_matrix[3, 3] = 1.0  # 设置齐次坐标
    pose_zero = transform_matrix_to_pose_2d(zero_matrix)
    print(f"零矩阵: x={pose_zero['x']:.3f}, y={pose_zero['y']:.3f}, theta={pose_zero['theta']:.3f}°")
    
    # 测试用例4: 实际场景模拟
    print("\n\n测试用例4: 实际场景模拟")
    print("-" * 40)
    
    # 模拟一个实际的定位结果
    real_transform = np.array([
        [ 0.8660, -0.5000,  0.0000,  2.5000],
        [ 0.5000,  0.8660,  0.0000,  1.8000],
        [ 0.0000,  0.0000,  1.0000,  0.3000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]
    ])
    
    print("实际场景转换矩阵:")
    print(real_transform)
    
    pose_real = transform_matrix_to_pose_2d(real_transform)
    print(f"\n转换结果:")
    print(f"位置: ({pose_real['x']:.3f}, {pose_real['y']:.3f}, {pose_real['z']:.3f})")
    print(f"朝向: {pose_real['theta']:.1f}° ({pose_real['theta_rad']:.3f} rad)")
    
    # 验证这个矩阵对应的角度（应该是30度）
    expected_angle = 30.0
    print(f"预期角度: {expected_angle}°")
    print(f"实际角度: {pose_real['theta']:.1f}°")
    print(f"角度误差: {abs(pose_real['theta'] - expected_angle):.3f}°")

def test_json_serialization():
    """测试JSON序列化"""
    print("\n\n" + "=" * 60)
    print("测试JSON序列化")
    print("=" * 60)
    
    import json
    
    # 创建一个模拟的VPS结果
    transform_matrix = create_test_transform_matrix(x=1.5, y=2.5, z=0.8, theta_degrees=60.0)
    
    # 模拟VPS返回的结果
    vps_result = {
        'pose': transform_matrix,
        'confidence': 0.85,
        'method': 'mast3r'
    }
    
    print("原始VPS结果:")
    print(f"pose shape: {vps_result['pose'].shape}")
    print(f"confidence: {vps_result['confidence']}")
    
    # 应用转换
    pose_2d = transform_matrix_to_pose_2d(vps_result['pose'])
    vps_result['pose_2d'] = pose_2d
    vps_result['pose'] = vps_result['pose'].tolist()
    
    print("\n转换后的结果:")
    print(f"pose_2d: {vps_result['pose_2d']}")
    print(f"pose (list): {len(vps_result['pose'])}x{len(vps_result['pose'][0])} 列表")
    
    # 测试JSON序列化
    try:
        json_str = json.dumps(vps_result, indent=2)
        print(f"\nJSON序列化成功，长度: {len(json_str)} 字符")
        print("JSON内容预览:")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
    except Exception as e:
        print(f"JSON序列化失败: {e}")

if __name__ == "__main__":
    test_pose_conversion()
    test_json_serialization()
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60) 