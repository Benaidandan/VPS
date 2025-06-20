from .vpr import VisualPlaceRecognition
from .pose import PoseEstimator
import yaml
from pathlib import Path
from typing import Dict, Union, Optional
import numpy as np
import json
import time

class VisualPositioningSystem:
    """Main class for the Visual Positioning System."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the VPS system.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.vpr = VisualPlaceRecognition(self.config)
        self.pose_estimator = PoseEstimator(self.config)
        

    def localize(self,
                 query_image: Union[str, Path],
                ) -> np.ndarray:
        """
        Perform visual localization for a single query image.
        
        Args:
            query_image: Path to query image
            
        Returns:
            Dictionary containing:
            - pose: 4x4 transformation matrix c2w
            - processing_time: processing time in seconds
        """
        # Find similar images using VPR
        a = time.time()
        similar_pairs = self.vpr.find_similar_images(
            query_image
        )
        b = time.time()
        print(f"VPR time: {b - a} seconds")
        pose_answer = self.pose_estimator.estimate_pose(query_image)
        c = time.time()
        print(f"Pose estimation time: {c - b} seconds")
        return pose_answer

