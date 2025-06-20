#!/usr/bin/env python3
"""
Test Pose functionality
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vps.core.pose import PoseEstimator
import yaml

# Load configuration
config_path = "configs/default.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize VPR
pose = PoseEstimator(config)

# Test with a query image
query_image = "/home/phw/visual-localization/VPS/data/ref/rgb/53.jpg"
print(f"Testing VPR with query image: {query_image}")
result = pose.estimate_pose(query_image)
print(f"✓ Pose test successful!")
print(f"Result: {result}")
