import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import cv2
import json

import torch.nn.functional as F
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class PoseEstimator:
    """Pose estimation module using VGGT."""
    
    def __init__(self, config: Dict):
        """
        Initialize the pose estimator.
        
        Args:
            config: Configuration dictionary containing pose estimation settings
        """
        self.config = config
        self.device = torch.device(config['system']['device'])
        
        # Set up dtype
        if config['system']['dtype'] == 'float16':
            self.dtype = torch.float16
        elif config['system']['dtype'] == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
            
        # Initialize VGGT model
        self.model = VGGT()
        model_path = config['pose']['vggt']['model_path']
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        else:
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Image preprocessing settings
        self.image_size = config['pose']['vggt']['image_size']


    def gengerate_ref_list(self, query_img: Union[str, Path]) -> list[str]:
        """
        Generate a list of reference images and their corresponding poses.
        """
        query_name = Path(query_img).name
        pairs_file = self.config['vpr']['pairs_file_path']
        ref_list = []
        with open(pairs_file, 'r') as f:
            for line in f:
                A, ref_name = line.strip().split()
                A = Path(A).name
                if A != query_name:
                    continue
                ref_name = Path(ref_name).name
                ref_image = Path(self.config['pose']['vggt']['ref_dir']) / "rgb" / ref_name
                if Path(ref_image).exists():
                    ref_list.append(ref_image)
                ref_render = Path(self.config['pose']['vggt']['ref_dir']) / "rgb_render" / ref_name
                if Path(ref_render).exists():
                    ref_list.append(ref_render)
        
        return ref_list

    def compute_scale_factor(self, 
                           vggt_depth: np.ndarray, 
                           gt_depth: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> float:
        """
        Compute scale factor between VGGT depth and ground truth depth.
        
        Args:
            vggt_depth: Depth map from VGGT
            gt_depth: Ground truth depth map
            mask: Optional mask for valid depth values
            
        Returns:
            Scale factor
        """
        if vggt_depth.shape != gt_depth.shape:
            gt_depth = cv2.resize(gt_depth, 
                                (vggt_depth.shape[1], vggt_depth.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        if mask is None:
            mask = (gt_depth > 1e-1) & (vggt_depth > 1e-1)
        valid_vggt = vggt_depth[mask]
        valid_gt = gt_depth[mask]
        scale_factors = valid_gt / valid_vggt
        if len(scale_factors) == 0:
            scale_factors = 1
        else:
            scale_factors = np.median(scale_factors)
        return scale_factors

    def run_VGGT(self, model, images, dtype, resolution=518):
    # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        # hard-coded to use 518 for VGGT
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        return extrinsic, intrinsic, depth_map, depth_conf
    
    def estimate_pose(self, query_img: Union[str, Path]) -> np.ndarray:
        """
        Estimate relative pose between query and reference images.
        
        Args:
            query_img: Path to query image
            ref_img: Path to reference image
            gt_depth: Optional ground truth depth for scale recovery
            
        Returns:
            Tuple containing:
            - Absoulte pose matrix (4x4)  camera2w
            - Depth map
            - Depth confidence map
            - Scale factor (if gt_depth provided, else 1.0)
        """
        # Preprocess images

        # 加载和预处理图像
        query_img = Path(query_img)
        ref_imgs = self.gengerate_ref_list(query_img)
        ref_imgs.append(query_img)
        image_paths = ref_imgs
        assert len(image_paths) >=2
        images, original_coords = load_and_preprocess_images_square(image_paths, self.image_size)
        images = images.to(self.device)
        original_coords = original_coords.to(original_coords.device)
        
        # 运行VGGT获取相机参数和深度图
        extrinsic, intrinsic, depth_map, depth_conf = self.run_VGGT(self.model, images, self.dtype, 518)
        
        # Compute relative pose
        P_query = np.concatenate([extrinsic[-1], np.array([[0, 0, 0, 1]])], axis=0)
        P_ref = np.concatenate([extrinsic[0], np.array([[0, 0, 0, 1]])], axis=0)
        query2ref = P_ref @ np.linalg.inv(P_query)
        # 读取第一张ref图像的pose和深度
        scale_factor = 1.0
        ref_img = Path(ref_imgs[0])
        ref_pose = np.loadtxt(ref_img.parent.parent / "poses" / f"{ref_img.stem}.txt").reshape(4, 4)
        ref_depth = None
        if Path(ref_img.parent.parent/"depth"/f"{ref_img.stem}.npy").exists():
            ref_depth = np.load(ref_img.parent.parent / "depth" / f"{ref_img.stem}.npy")
        if Path(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy").exists():
            ref_depth = np.load(ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy")
        

        if ref_depth is not None:
            vggt_depth = depth_map[0].squeeze()
            ref_depth = cv2.resize(ref_depth, (vggt_depth.shape[1], vggt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            scale_factor = self.compute_scale_factor(vggt_depth, ref_depth)
            query2ref[:3, 3] *= scale_factor
            print(f"scale_factor: {scale_factor}")
        
         # 计算最终的位姿
        final_pose = ref_pose @ query2ref
        result_path = Path(self.config['pose']['vggt']['results_dir']) / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)   

        return final_pose 