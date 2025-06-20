#!/usr/bin/env python3
"""
Complete VPS Workflow

This script demonstrates the complete workflow:
1. Convert COLMAP data to basic VPS format
2. Generate render data (optional)
3. Use VPS for localization
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.colmap_to_vps import COLMAPToVPSConverter
from scripts.generate_renders import RenderGenerator
from vps.core import VisualPositioningSystem

def step1_convert_colmap_data(colmap_dir: str, output_dir: str):
    """Step 1: Convert COLMAP data to basic VPS format."""
    print("=" * 60)
    print("STEP 1: Converting COLMAP data to VPS format")
    print("=" * 60)
    
    colmap_path = Path(colmap_dir)
    output_path = Path(output_dir)
    
    if not colmap_path.exists():
        print(f"❌ COLMAP directory not found: {colmap_path}")
        return False
    
    # Create converter and run conversion
    converter = COLMAPToVPSConverter(
        colmap_dir=colmap_path,
        output_dir=output_path
    )
    
    converter.convert_basic_data()
    return True

def step2_generate_renders(ref_data_dir: str, mesh_path: str):
    """Step 2: Generate render data from mesh (optional)."""
    print("\n" + "=" * 60)
    print("STEP 2: Generating render data from mesh")
    print("=" * 60)
    
    ref_path = Path(ref_data_dir)
    mesh_file = Path(mesh_path)
    
    if not mesh_file.exists():
        print(f"⚠️  Mesh file not found: {mesh_file}")
        print("Skipping render generation...")
        return True
    
    # Create render generator and run
    generator = RenderGenerator(
        ref_data_dir=ref_path,
        mesh_path=mesh_file
    )
    
    generator.generate_renders()
    return True

def step3_test_vps_localization(ref_data_dir: str, query_image: str):
    """Step 3: Test VPS localization."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing VPS localization")
    print("=" * 60)
    
    ref_path = Path(ref_data_dir)
    query_path = Path(query_image)
    
    if not query_path.exists():
        print(f"❌ Query image not found: {query_path}")
        return False
    
    # Initialize VPS system
    config_path = "configs/default.yaml"
    vps = VisualPositioningSystem(config_path)
    
    # Get reference database info
    ref_info = vps.get_reference_info()
    print(f"Reference database info: {ref_info}")
    
    # Perform localization
    print(f"Localizing query image: {query_path}")
    result = vps.localize(query_path, use_gt_depth=True)
    
    # Print results
    print("Localization Results:")
    print(f"  - Confidence: {result['confidence']:.4f}")
    print(f"  - Processing time: {result['processing_time']:.2f}s")
    print(f"  - Best reference image: {result['best_ref_image']}")
    print(f"  - Number of similar images: {len(result['similar_images'])}")
    
    if result['pose'] is not None:
        print("  - Estimated pose matrix:")
        for i, row in enumerate(result['pose']):
            print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")
    
    return True

def main():
    """Main workflow function."""
    print("VPS Complete Workflow")
    print("This script demonstrates the complete VPS pipeline")
    
    # Configuration (modify these paths for your data)
    colmap_dir = "/path/to/your/colmap/data"  # COLMAP directory with images/ and sparse/
    output_dir = "data/ref"  # VPS reference database output
    mesh_path = "/path/to/your/mesh.obj"  # Optional mesh for rendering
    query_image = "/path/to/your/query.jpg"  # Query image for testing
    
    print(f"\nConfiguration:")
    print(f"  COLMAP directory: {colmap_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mesh file: {mesh_path}")
    print(f"  Query image: {query_image}")
    
    # Check if paths exist
    if not Path(colmap_dir).exists():
        print(f"\n❌ Please modify the colmap_dir path in this script.")
        return
    
    if not Path(query_image).exists():
        print(f"\n❌ Please modify the query_image path in this script.")
        return
    
    # Run the complete workflow
    success = True
    
    # Step 1: Convert COLMAP data
    success &= step1_convert_colmap_data(colmap_dir, output_dir)
    
    if success:
        # Step 2: Generate renders (optional)
        step2_generate_renders(output_dir, mesh_path)
        
        # Step 3: Test localization
        step3_test_vps_localization(output_dir, query_image)
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Complete workflow finished successfully!")
    else:
        print("❌ Workflow failed. Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    main() 