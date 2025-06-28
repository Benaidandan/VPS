#!/usr/bin/env python3
"""
Test script for MASt3R pose estimation integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
from vps.core import VisualPositioningSystem
import numpy as np
from scipy.spatial.transform import Rotation as R

def test_mast3r_pose():
    """Test MASt3R pose estimation."""
    
    # Load configuration
    config_path = "configs/default.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Change pose method to mast3r
    config['pose']['method'] = 'mast3r'
    
    # Save temporary config
    temp_config_path = "configs/test_mast3r.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Initialize VPS with MASt3R
        vps = VisualPositioningSystem(temp_config_path)
        
        # Test with a sample query image
        query_image = "data/query/554.jpg"
        
        if not Path(query_image).exists():
            print(f"Query image {query_image} not found. Please check the path.")
            return
        
        print(f"Testing MASt3R pose estimation with query image: {query_image}")
        
        # Perform localization
        estimated_pose = vps.localize(query_image)
        
        print("Estimated pose matrix:")
        print(estimated_pose)
        
        # Load ground truth pose for comparison
        gt_pose_path = "data/ref/poses/554.txt"
        if Path(gt_pose_path).exists():
            gt_pose = np.loadtxt(gt_pose_path).reshape(4, 4)
            
            # Calculate errors
            T1 = gt_pose
            T2 = estimated_pose
            
            # Rotation and translation parts
            R1, t1 = T1[:3, :3], T1[:3, 3]
            R2, t2 = T2[:3, :3], T2[:3, 3]
            
            # Translation error (L2 distance)
            trans_error = np.linalg.norm(t1 - t2)
            
            # Rotation error (degrees)
            R_rel = R1.T @ R2
            rotvec = R.from_matrix(R_rel).as_rotvec()
            rot_error_deg = np.linalg.norm(rotvec) * 180 / np.pi
            
            print(f"\nError Analysis:")
            print(f"Rotation error: {rot_error_deg:.4f} degrees")
            print(f"Translation error: {trans_error:.4f} meters")
        
        print("\nMASt3R pose estimation test completed successfully!")
        
    except Exception as e:
        print(f"Error during MASt3R test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary config
        if Path(temp_config_path).exists():
            os.remove(temp_config_path)

if __name__ == "__main__":
    test_mast3r_pose() 