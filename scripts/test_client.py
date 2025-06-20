#!/usr/bin/env python3
"""
VPS Test Client

This script demonstrates how to use the VPS service.
"""

import requests
import json
from pathlib import Path
import argparse

def test_health_check(base_url: str):
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        print("✓ Health check passed")
        print(f"Reference database info: {response.json()['reference_info']}")
    else:
        print(f"✗ Health check failed: {response.status_code}")

def test_localization(base_url: str, image_path: str, use_gt_depth: bool = True):
    """Test the localization endpoint."""
    print(f"Testing localization with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'use_gt_depth': str(use_gt_depth).lower()}
        
        response = requests.post(f"{base_url}/localize", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Localization successful")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Best reference image: {result['best_ref_image']}")
            print(f"Number of similar images: {len(result['similar_images'])}")
            
            if result['pose']:
                print("Estimated pose matrix:")
                for row in result['pose']:
                    print(f"  {row}")
        else:
            print(f"✗ Localization failed: {response.status_code}")
            print(f"Error: {response.text}")

def main():
    parser = argparse.ArgumentParser(description='Test VPS service')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                       help='Base URL of the VPS service')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--no-gt-depth', action='store_true',
                       help='Disable ground truth depth usage')
    
    args = parser.parse_args()
    
    # Test health check
    test_health_check(args.url)
    print()
    
    # Test localization
    test_localization(args.url, args.image, not args.no_gt_depth)

if __name__ == '__main__':
    main() 