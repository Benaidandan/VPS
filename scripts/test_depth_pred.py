import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vps.core.depth_pred import DepthPred
from vps.utils.processing import compute_scale_factor
import yaml
import numpy as np
config_path = "/home/phw/visual-localization/VPS/configs/default.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
depth_model = DepthPred(config)
start_time = time.time()
for i in range(10):
    out = depth_model.infer("/home/phw/visual-localization/VPS/outputs/rgb+depth/frame_0000.jpg")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
depth_pred = out['depth'].cpu().numpy()
depth_gt = np.load("/home/phw/visual-localization/VPS/outputs/rgb+depth/frame_0000.npy")
depth_scaled = compute_scale_factor(depth_pred, depth_gt)

print(depth_scaled)

