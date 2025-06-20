#!/usr/bin/env python3
"""
Generate Render Data for VPS

This script generates RGB and depth render data from mesh using existing poses and calibration.
"""

import os
import numpy as np
from pathlib import Path
import argparse
import time
import json
import yaml

# 设置环境变量用于渲染
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["DISPLAY"] = ":0"

import trimesh
import pyrender
import imageio

class RenderGenerator:
    def __init__(self, ref_data_dir: Path, mesh_path: Path):
        """
        Initialize the render generator.
        
        Args:
            ref_data_dir: Path to VPS reference database (containing rgb/, poses/, calibration/)
            mesh_path: Path to mesh file for rendering
        """
        self.ref_data_dir = Path(ref_data_dir)
        self.mesh_path = Path(mesh_path)
        
        # Input paths
        self.rgb_dir = self.ref_data_dir / "rgb"
        self.poses_dir = self.ref_data_dir / "poses"
        self.calibration_dir = self.ref_data_dir / "calibration"
        
        # Output paths
        self.rgb_render_dir = self.ref_data_dir / "rgb_render"
        self.depth_render_dir = self.ref_data_dir / "depth_render"
        
        # Create output directories
        for dir_path in [self.rgb_render_dir, self.depth_render_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Rendering scene
        self.render_scene = None
        self._setup_rendering()

    def _setup_rendering(self):
        """Setup rendering scene from mesh."""
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
        
        print(f"Loading mesh from: {self.mesh_path}")
        self.render_scene = self._build_base_scene(self.mesh_path)
        print("✓ Rendering scene setup complete")

    def _build_base_scene(self, mesh_path):
        """Build pyrender scene from mesh file."""
        mesh_obj = trimesh.load(mesh_path)
        pyr_scene = pyrender.Scene()
        
        if isinstance(mesh_obj, trimesh.Scene):
            for _, geom in mesh_obj.geometry.items():
                if not hasattr(geom.visual, 'vertex_colors') or geom.visual.vertex_colors is None:
                    geom.visual.vertex_colors = [200, 200, 200, 255]
                pyr_scene.add(pyrender.Mesh.from_trimesh(geom))
        else:
            if not hasattr(mesh_obj.visual, 'vertex_colors') or mesh_obj.visual.vertex_colors is None:
                mesh_obj.visual.vertex_colors = [200, 200, 200, 255]
            pyr_scene.add(pyrender.Mesh.from_trimesh(mesh_obj))
        
        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        pyr_scene.add(light, pose=np.eye(4))
        
        return pyr_scene

    def _create_camera_intrinsics(self, fx, fy, cx, cy, width, height):
        """Create pyrender camera intrinsics."""
        return pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=1000.0)

    def _ros_to_pyrender_pose(self, ros_pose):
        """Convert ROS coordinate system pose to pyrender pose."""
        R = ros_pose[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        t = ros_pose[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        return pose

    def _render_scene(self, scene, camera_pose, intrinsics_camera, width, height):
        """Render RGB and depth from scene."""
        cam_node = scene.add(intrinsics_camera, pose=camera_pose)
        light_node = scene.add(pyrender.PointLight(color=np.ones(3), intensity=50.0), pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        color, depth = renderer.render(scene)
        renderer.delete()

        scene.remove_node(cam_node)
        scene.remove_node(light_node)
        return color, depth

    def _get_image_resolution(self, image_path):
        """Get image resolution from RGB image."""
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            return img.shape[1], img.shape[0]  # width, height
        return None, None

    def generate_renders(self):
        """Generate render data for all images."""
        print(f"Generating render data from: {self.mesh_path}")
        print(f"Output directory: {self.ref_data_dir}")
        
        # Get all RGB images
        rgb_images = list(self.rgb_dir.glob("*.jpg")) + list(self.rgb_dir.glob("*.png"))
        total_images = len(rgb_images)
        
        if total_images == 0:
            print("❌ No RGB images found!")
            return
        
        print(f"Found {total_images} images to render")
        
        rendered_count = 0
        failed_count = 0
        start_time = time.time()
        for i, rgb_image_path in enumerate(rgb_images):
            image_name = rgb_image_path.name
            image_stem = rgb_image_path.stem
            
            print(f"Rendering {i+1}/{total_images}: {image_name}")
            
            try:
                # Load pose (4x4 matrix)
                pose_path = self.poses_dir / f"{image_stem}.txt"
                if not pose_path.exists():
                    print(f"⚠️  Pose file not found: {pose_path}")
                    failed_count += 1
                    continue
                
                pose_matrix = np.loadtxt(pose_path)
                
                # Load calibration (3x3 matrix)
                calib_path = self.calibration_dir / f"{image_stem}.txt"
                if not calib_path.exists():
                    print(f"⚠️  Calibration file not found: {calib_path}")
                    failed_count += 1
                    continue
                
                K = np.loadtxt(calib_path)
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                
                # Get image resolution
                width, height = self._get_image_resolution(rgb_image_path)
                if width is None or height is None:
                    print(f"⚠️  Could not read image resolution: {rgb_image_path}")
                    failed_count += 1
                    continue
                
                # Create camera intrinsics for rendering
                intrinsics_camera = self._create_camera_intrinsics(fx, fy, cx, cy, width, height)
                
                # Convert pose for rendering
                render_pose = self._ros_to_pyrender_pose(pose_matrix)
                
                # Render
                rgb, depth = self._render_scene(self.render_scene, render_pose, 
                                              intrinsics_camera, width, height)
                
                # Save rendered images
                rgb_render_path = self.rgb_render_dir / image_name
                depth_render_path = self.depth_render_dir / f"{image_stem}.npy"
                
                imageio.imwrite(rgb_render_path, rgb.astype(np.uint8))
                np.save(depth_render_path, depth)
                
                rendered_count += 1
                
            except Exception as e:
                print(f"❌ Rendering failed for {image_name}: {e}")
                failed_count += 1
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Render generation time: {total_time} seconds")
        print(f"✓ Render generation complete!")
        print(f"  - Successfully rendered: {rendered_count}")
        print(f"  - Failed: {failed_count}")
        print(f"  - Total: {total_images}")
        
        # Create render summary
        self._create_render_summary(rendered_count, failed_count, total_images,total_time)

    def _create_render_summary(self, rendered_count, failed_count, total_images,total_time):
        """Create a summary of the render generation."""
        summary = {
            "render_info": {
                "mesh_path": str(self.mesh_path),
                "ref_data_dir": str(self.ref_data_dir),
                "time": total_time
            },
            "render_counts": {
                "total_images": total_images,
                "successfully_rendered": rendered_count,
                "failed": failed_count,
                "rgb_renders": len(list(self.rgb_render_dir.glob("*.jpg"))) + len(list(self.rgb_render_dir.glob("*.png"))),
                "depth_renders": len(list(self.depth_render_dir.glob("*.npy")))
            }
        }
        
        summary_path = self.ref_data_dir / "render_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Render summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate render data for VPS')
    parser.add_argument('--config', type=str, default='configs/data.yaml')
    parser.add_argument('--ref_data_dir', type=str,default=None,
                       help='Path to VPS reference database (containing rgb/, poses/, calibration/)')
    parser.add_argument('--mesh_path', type=str, default=None,
                       help='Path to mesh file for rendering')
    
    args = parser.parse_args()
        # 读取配置文件
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    # 命令行参数优先生效
    ref_data_dir = Path(args.ref_data_dir) if args.ref_data_dir else Path(cfg['output_dir'])
    mesh_path = Path(args.mesh_path) if args.mesh_path else Path(cfg['mesh_path'])
    # Validate inputs
    
    if not ref_data_dir.exists():
        print(f"❌ Reference data directory not found: {ref_data_dir}")
        return
    
    if not (ref_data_dir / "rgb").exists() or not (ref_data_dir / "poses").exists():
        print(f"❌ Invalid reference data directory structure. Expected rgb/ and poses/ subdirectories.")
        return
    
    if not mesh_path.exists():
        print(f"❌ Mesh file not found: {mesh_path}")
        return
    
    # Create render generator and run
    generator = RenderGenerator(
        ref_data_dir=ref_data_dir,
        mesh_path=mesh_path
    )
    
    generator.generate_renders()

if __name__ == "__main__":
    main() 