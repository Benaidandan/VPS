import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vps.core.depth_pred import DepthPred
from vps.utils.processing import compute_scale_factor
import yaml
import numpy as np
import cv2
config_path = "/home/phw/visual-localization/VPS/configs/default.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
depth_model = DepthPred(config)
start_time = time.time()
for i in range(1):
    out = depth_model.infer("/home/phw/newdisk1/VPS_data/7/pgt_7scenes_chess/test/rgb/seq-03-frame-000147.png")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
depth_pred = out['depth'].cpu().numpy()
# np.save("/home/phw/newdisk1/VPS_data/7scenes_chess/outputs/seq-03-frame-000147.npy", depth_pred)
depth_png = cv2.imread("/home/phw/newdisk1/VPS_data/7scenes_source/chess/seq-01/frame-000002.depth.png", cv2.IMREAD_UNCHANGED)
depth_gt = depth_png.astype(np.float32) /1000.0
# depth_gt = np.load("/home/phw/visual-localization/VPS/outputs/rgb+depth/frame_0000.npy")
depth_scaled = compute_scale_factor(depth_pred, depth_gt)

print(depth_scaled)

