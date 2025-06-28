#!/usr/bin/env python3
"""
Create side-by-side video from two folders with same-named images
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

def get_common_images(folder_a: Path, folder_b: Path) -> List[Tuple[Path, Path]]:
    """
    Get pairs of images with the same name from two folders
    
    Args:
        folder_a: Path to first folder
        folder_b: Path to second folder
        
    Returns:
        List of tuples (image_a_path, image_b_path)
    """
    # Get all image files from both folders
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    images_a = {}
    images_b = {}
    
    # Load images from folder A
    for ext in image_exts:
        for img_path in folder_a.glob(f"*{ext}"):
            images_a[img_path.stem] = img_path
    
    # Load images from folder B
    for ext in image_exts:
        for img_path in folder_b.glob(f"*{ext}"):
            images_b[img_path.stem] = img_path
    
    # Find common image names
    common_names = set(images_a.keys()) & set(images_b.keys())
    
    # Create pairs
    pairs = []
    for name in sorted(common_names):
        pairs.append((images_a[name], images_b[name]))
    
    print(f"Found {len(pairs)} common images")
    return pairs

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor to fit in target size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create canvas with target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def create_side_by_side_frame(img_a: np.ndarray, img_b: np.ndarray, 
                            target_size: Tuple[int, int]) -> np.ndarray:
    """
    Create a side-by-side frame from two images
    
    Args:
        img_a: Left image
        img_b: Right image
        target_size: Target size for each half (width, height)
        
    Returns:
        Combined frame
    """
    # Resize both images to target size
    img_a_resized = resize_image(img_a, target_size)
    img_b_resized = resize_image(img_b, target_size)
    
    # Create combined frame
    combined_width = target_size[0] * 2
    combined_height = target_size[1]
    
    combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Place images side by side
    combined_frame[:, :target_size[0]] = img_a_resized
    combined_frame[:, target_size[0]:] = img_b_resized
    
    return combined_frame

def create_side_by_side_video(folder_a: Path, folder_b: Path, output_path: Path,
                            fps: int = 30, target_size: Tuple[int, int] = (640, 480),
                            add_labels: bool = True, label_a: str = "Folder A", 
                            label_b: str = "Folder B"):
    """
    Create side-by-side video from two folders
    
    Args:
        folder_a: Path to first folder
        folder_b: Path to second folder
        output_path: Output video path
        fps: Frames per second
        target_size: Target size for each half (width, height)
        add_labels: Whether to add text labels
        label_a: Label for left side
        label_b: Label for right side
    """
    # Get common images
    image_pairs = get_common_images(folder_a, folder_b)
    
    if not image_pairs:
        print("❌ No common images found!")
        return False
    
    # Calculate video dimensions
    video_width = target_size[0] * 2
    video_height = target_size[1]
    
    print(f"Video dimensions: {video_width}x{video_height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {len(image_pairs)}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (video_width, video_height))
    
    if not out.isOpened():
        print("❌ Error: Could not open video writer")
        return False
    
    # Process each image pair
    for i, (img_a_path, img_b_path) in enumerate(image_pairs):
        print(f"Processing frame {i+1}/{len(image_pairs)}: {img_a_path.name}")
        
        # Load images
        img_a = cv2.imread(str(img_a_path))
        img_b = cv2.imread(str(img_b_path))
        
        if img_a is None or img_b is None:
            print(f"⚠️  Skipping {img_a_path.name} - could not load images")
            continue
        
        # Create side-by-side frame
        combined_frame = create_side_by_side_frame(img_a, img_b, target_size)
        
        # Add labels if requested
        if add_labels:
            # Add left label
            cv2.putText(combined_frame, label_a, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add right label
            cv2.putText(combined_frame, label_b, (target_size[0] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add frame number
            cv2.putText(combined_frame, f"Frame {i+1}/{len(image_pairs)}", 
                       (10, video_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(combined_frame)
    
    # Release video writer
    out.release()
    
    print(f"✓ Video created successfully: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create side-by-side video from two image folders")
    parser.add_argument("folder_a", help="Path to first image folder")
    parser.add_argument("folder_b", help="Path to second image folder")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Width of each half (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Height of each half (default: 480)")
    parser.add_argument("--no-labels", action="store_true", help="Disable text labels")
    parser.add_argument("--label-a", default="Folder A", help="Label for left side")
    parser.add_argument("--label-b", default="Folder B", help="Label for right side")
    
    args = parser.parse_args()
    
    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)
    output_path = Path(args.output)
    
    if not folder_a.exists():
        print(f"❌ Error: Folder A does not exist: {folder_a}")
        return
    
    if not folder_b.exists():
        print(f"❌ Error: Folder B does not exist: {folder_b}")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video
    success = create_side_by_side_video(
        folder_a=folder_a,
        folder_b=folder_b,
        output_path=output_path,
        fps=args.fps,
        target_size=(args.width, args.height),
        add_labels=not args.no_labels,
        label_a=args.label_a,
        label_b=args.label_b
    )
    
    if success:
        print("🎉 Video creation completed!")
    else:
        print("❌ Video creation failed!")

if __name__ == "__main__":
    # Example usage (uncomment to use directly)
    # folder_a = Path("data/query")
    # folder_b = Path("data/ref/rgb")
    # output_path = Path("outputs/side_by_side.mp4")
    # 
    # create_side_by_side_video(folder_a, folder_b, output_path, fps=10)
    
    main() 