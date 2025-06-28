import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import cv2
import json
import logging
import sys
sys.path.append("third_party/mast3r")
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from scipy.spatial.transform import Rotation as R

_logger = logging.getLogger(__name__)

def getScale(depth1: np.ndarray, depth2: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute scale factor between two depth maps.
    
    Args:
        depth1: First depth map
        depth2: Second depth map  
        mask: Optional mask for valid depth values
        
    Returns:
        Scale factor
    """
    if depth1.shape != depth2.shape:
        depth2 = cv2.resize(depth2, 
                           (depth1.shape[1], depth1.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    
    if mask is None:
        mask = (depth1 > 1e-1) & (depth2 > 1e-1)
    
    valid_depth1 = depth1[mask]
    valid_depth2 = depth2[mask]
    
    if len(valid_depth1) == 0:
        return 1.0
    
    scale_factors = valid_depth2 / valid_depth1
    scale_factor = np.median(scale_factors)
    
    return scale_factor

class PoseEstimatorMASt3R:
    """Pose estimation module using MASt3R."""
    
    def __init__(self, config: Dict):
        """
        Initialize the MASt3R pose estimator.
        
        Args:
            config: Configuration dictionary containing pose estimation settings
        """
        self.config = config
        self.device = torch.device(config['system']['device'])
        
        # Initialize MASt3R model
        model_path = config['pose']['mast3r']['model_path']
        self.model = AsymmetricMASt3R.from_pretrained(model_path).to(self.device).eval()
        
        # MASt3R specific settings
        self.image_size = config['pose']['mast3r']['image_size']
        self.batch_size = config['pose']['mast3r']['batch_size']

    def generate_ref_list(self, query_img: Union[str, Path]) -> list[str]:
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
                ref_image = Path(self.config['pose']['mast3r']['ref_dir']) / "rgb" / ref_name
                if Path(ref_image).exists():
                    ref_list.append(str(ref_image))
                ref_render = Path(self.config['pose']['mast3r']['ref_dir']) / "rgb_render" / ref_name
                if Path(ref_render).exists():
                    ref_list.append(str(ref_render))
        
        return ref_list

    def run_MASt3R(self, ref_img: Union[str, Path], query_img: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run MASt3R inference on a pair of images.
        
        Args:
            ref_img: Path to reference image
            query_img: Path to query image
            
        Returns:
            Tuple containing:
            - Relative pose matrix (4x4) query2ref
            - Depth map from MASt3R
        """
        # Load and preprocess images
        images = load_images([str(ref_img), str(query_img)], size=self.image_size)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        
        # Run inference
        output = inference(pairs, self.model, self.device, batch_size=self.batch_size)
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer)
        
        # Get poses
        poses = scene.get_im_poses()
        
        # Determine relative pose (query2ref)
        P_ini = np.eye(4)
        flag = poses[0].detach().cpu().numpy()
        
        if np.allclose(flag, P_ini, atol=1e-6):
            P_rel = poses[1].detach().cpu().numpy()  # query2ref
        else:
            P_rel = poses[0].detach().cpu().numpy()
            P_rel = np.linalg.inv(P_rel)  # query2ref
        
        # Get depth map
        depth_map = scene.get_depthmaps()[0].cpu().numpy()
        
        return P_rel, depth_map

    def estimate_pose(self, query_img: Union[str, Path]) -> np.ndarray:
        """
        Estimate absolute pose for query image using MASt3R.
        
        Args:
            query_img: Path to query image
            
        Returns:
            Absolute pose matrix (4x4) camera2world
        """
        query_img = Path(query_img)
        ref_imgs = self.generate_ref_list(query_img)
        
        if not ref_imgs:
            raise ValueError(f"No reference images found for query {query_img.name}")
        
        # Use the first reference image
        ref_img = Path(ref_imgs[0])
        
        # Run MASt3R to get relative pose and depth
        P_rel, depth_map_mast3r = self.run_MASt3R(ref_img, query_img)
        
        # Load reference pose and depth for scale recovery
        ref_pose = np.loadtxt(ref_img.parent.parent / "poses" / f"{ref_img.stem}.txt").reshape(4, 4)
        
        # Try to load reference depth for scale recovery
        ref_depth = None
        depth_path = ref_img.parent.parent / "depth" / f"{ref_img.stem}.npy"
        depth_render_path = ref_img.parent.parent / "depth_render" / f"{ref_img.stem}.npy"
        
        if depth_path.exists():
            ref_depth = np.load(depth_path)
        elif depth_render_path.exists():
            ref_depth = np.load(depth_render_path)
        
        # Apply scale recovery if reference depth is available
        scale_factor = 1.0
        if ref_depth is not None:
            depth_map_resized = cv2.resize(ref_depth, 
                                         (depth_map_mast3r.shape[1], depth_map_mast3r.shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
            scale_factor = getScale(depth_map_mast3r, depth_map_resized)
            P_rel[:3, 3] *= scale_factor
            _logger.info(f"Scale factor: {scale_factor}")
        
        # Compute final absolute pose
        final_pose = ref_pose @ P_rel
        
        # Save result
        result_path = Path(self.config['pose']['mast3r']['results_dir']) / f"{query_img.stem}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(result_path, final_pose)
        
        return final_pose

