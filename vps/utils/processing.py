import numpy as np
import cv2
from pathlib import Path
from typing import Union, Optional, List, Dict

def generate_ref_list(
    query_img: Union[str, Path], 
    ref_dir: Union[str, Path], 
    pairs_file: Union[str, Path]) -> List[str]:
    """
    根据给定的查询图像，从配对文件中生成参考图像列表。
    """
    query_name = Path(query_img).name
    ref_list = []
    with open(pairs_file, 'r') as f:
        for line in f:
            A, ref_name = line.strip().split()
            A = Path(A).name
            if A != query_name:
                continue
            ref_name = Path(ref_name).name
            ref_image = Path(ref_dir) / "rgb" / ref_name
            if Path(ref_image).exists():
                ref_list.append(str(ref_image))
            ref_render = Path(ref_dir) / "rgb_render" / ref_name
            if Path(ref_render).exists():
                ref_list.append(str(ref_render))
    
    return ref_list

def compute_scale_factor( 
                           pred_depth: np.ndarray, 
                           gt_depth: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> float:
    """
    计算模型预测深度和真值深度之间的尺度因子。
    
    Args:
        pred_depth: 预测的深度图 np.ndarray
        gt_depth: 真值深度图 np.ndarray
        mask: 可选的有效深度值掩码
        
    Returns:
        尺度因子
    """
    if pred_depth.shape != gt_depth.shape:
        gt_depth = cv2.resize(gt_depth, 
                                (pred_depth.shape[1], pred_depth.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    if mask is None:
        mask = (gt_depth > 1e-1) & (pred_depth > 1e-1)
    valid_pred = pred_depth[mask]
    valid_gt = gt_depth[mask]
    scale_factors = valid_gt / valid_pred
    if len(scale_factors) == 0:
        return 1.0
    else:
        return float(np.median(scale_factors))