import numpy as np
import argparse
import os


def load_pose_txt(pose_path):
    """Load 4x4 pose matrix from txt file"""
    pose = np.loadtxt(pose_path)
    assert pose.shape == (4, 4), f"Pose file must be 4x4 matrix: {pose_path}"
    return pose


def compute_angle(t1, t2):
    """Compute angle between two translation vectors (in degrees)"""
    cos_theta = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


def compute_best_scale(t_gt, t_pred):
    """Compute the best scale to align t_pred to t_gt"""
    scale = np.dot(t_gt, t_pred) / np.dot(t_pred, t_pred)
    return scale


def main(gt_path, pred_path):
    T_gt = load_pose_txt(gt_path)
    T_pred = load_pose_txt(pred_path)

    t_gt = T_gt[:3, 3]
    t_pred = T_pred[:3, 3]

    print("=== Translation Vector Analysis ===")
    print(f"GT translation:    {t_gt}")
    print(f"Pred translation:  {t_pred}")

    angle_deg = compute_angle(t_gt, t_pred)
    scale = compute_best_scale(t_gt, t_pred)
    t_pred_scaled = scale * t_pred
    error = np.linalg.norm(t_pred_scaled - t_gt)*100

    print(f"\n[✓] 方向夹角(deg) : {angle_deg:.3f}°")
    print(f"[✓] 最佳尺度 s      : {scale:.4f}")
    print(f"[✓] 缩放后平移误差   : {error:.4f} cm")
    return angle_deg, scale, error


if __name__ == "__main__":

    gt_pose = '/home/phw/newdisk1/VPS_data/7/pgt_7scenes_chess/test/poses/seq-03-frame-000147.txt'
    pred_pose = '/home/phw/newdisk1/VPS_data/7/pgt_7scenes_chess/outputs/poses/seq-03-frame-000147.txt'
    main(gt_pose, pred_pose)
