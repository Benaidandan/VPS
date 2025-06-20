#!/usr/bin/env python3
"""
Test VPR functionality
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vps.core.vpr import VisualPlaceRecognition
import yaml

# Load configuration
config_path = "configs/default.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize VPR
vpr = VisualPlaceRecognition(config)

# Test with a query image
query_image = "/home/phw/visual-localization/VPS/data/ref/rgb/52.jpg"
print(f"Testing VPR with query image: {query_image}")


pairs_file = vpr.find_similar_images(query_image)
print(f"✓ VPR test successful!")
print(f"Pairs file: {pairs_file}")
