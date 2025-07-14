import cv2
import torch
from typing import Dict
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2
class DepthPred:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['system']['device'])
        self.model_path = config['pose']['depth_pre_model_path']
        self.model = MoGeModel.from_pretrained(self.model_path).to(self.device)


    def infer(self, input_image_path: str) -> Dict:

        # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
        input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)                       
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)    

        # Infer 
        output = self.model.infer(input_image)
        return output
        """
        `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
        The maps are in the same size as the input image. 
        {
            "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
            "depth": (H, W),        # depth map
            "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
            "mask": (H, W),         # a binary mask for valid pixels. 
            "intrinsics": (3, 3),   # normalized camera intrinsics
        }
        """