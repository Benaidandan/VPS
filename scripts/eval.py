#!/usr/bin/env python3
"""
Test vps localization functionality
"""

import sys
from pathlib import Path
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vps.core import VisualPositioningSystem
import yaml
import time
def rotation_error(R1, R2):
    # R1, R2: 3x3旋转矩阵
    R = R1 @ R2.T
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)

def translation_error(t1, t2):
    # t1, t2: 3维平移向量
    return np.linalg.norm(t1 - t2) * 100  # m->cm
# 假设vps结果和gt都为4x4的txt
def evaluate(query_dir, result_dir, gt_dir, result_txt_path):
    thresholds = [
        (1, 1),(3,3), (3, 5),(3,10),(3,15),(3,20),(3,30) ,(5, 10), (5, 15)
    ]
    counts = [0] * len(thresholds)
    total = 0
    results = []

    t_errs = []
    r_errs = []

    for ext in ["*.jpg", "*.png"]:
        for q in Path(query_dir).glob(ext):
            name = q.stem
            pred_path = Path(result_dir) / f"{name}.txt"
            gt_path = Path(gt_dir) / f"{name}.txt"
            if not pred_path.exists() or not gt_path.exists():
                continue
            pred = np.loadtxt(pred_path)
            gt = np.loadtxt(gt_path)
            R_pred, t_pred = pred[:3, :3], pred[:3, 3]
            R_gt, t_gt = gt[:3, :3], gt[:3, 3]
            r_err = rotation_error(R_pred, R_gt)
            t_err = translation_error(t_pred, t_gt)
            t_errs.append(t_err)
            r_errs.append(r_err)
            results.append(f"{name}: {t_err:.4f} {r_err:.4f}")
            for i, (r_th, t_th) in enumerate(thresholds):
                if r_err < r_th and t_err < t_th:
                    counts[i] += 1
            total += 1

    with open(result_txt_path, 'w') as f:
        for line in results:
            f.write(line + '\n')
        f.write(f"总数: {total}\n")
        for i, (r_th, t_th) in enumerate(thresholds):
            percent = (counts[i]/total*100) if total > 0 else 0
            f.write(f"小于{r_th}度{t_th}cm: {percent:.2f}%\n")
        if total > 0:
            f.write(f"平移误差: 平均={np.mean(t_errs):.4f} 中位数={np.median(t_errs):.4f} 最大值={np.max(t_errs):.4f}\n")
            f.write(f"旋转误差: 平均={np.mean(r_errs):.4f} 中位数={np.median(r_errs):.4f} 最大值={np.max(r_errs):.4f}\n")


# # Load configuration
config_path = "configs/default.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize VPS
vps = VisualPositioningSystem(config_path=config_path)
start_time = time.time()
query_dir = Path("/home/phw/visual-localization/VPS/data/query")
for ext in ["*.jpg", "*.png"]:
    for query_image in query_dir.glob(ext):
        vps.localize(query_image)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
result_dir = "data/outputs/poses"
gt_dir = "data/ref/poses"
result_txt_path = "data/outputs/result.txt"
evaluate(query_dir, result_dir, gt_dir, result_txt_path)



    