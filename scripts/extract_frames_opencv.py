#!/usr/bin/env python3
"""
Extract frames from MP4 video using OpenCV
"""

import cv2
from pathlib import Path
import numpy as np

def extract_frames_opencv(video_path, output_dir, fps=None, start_time=None, duration=None):
    """
    Extract frames from video using OpenCV
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frame rate (optional, if None uses original fps)
        start_time: Start time in seconds (optional)
        duration: Duration to extract in seconds (optional)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / original_fps
    
    print(f"Video info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {original_fps}")
    print(f"  - Duration: {video_duration:.2f} seconds")
    
    # Calculate frame interval for desired fps
    if fps is not None:
        frame_interval = int(original_fps / fps)
        print(f"  - Extracting every {frame_interval} frames (target fps: {fps})")
    else:
        frame_interval = 1
        print(f"  - Extracting all frames")
    
    # Calculate start and end frames
    start_frame = 0
    end_frame = total_frames
    
    if start_time is not None:
        start_frame = int(start_time * original_fps)
        print(f"  - Start frame: {start_frame}")
    
    if duration is not None:
        end_frame = min(total_frames, start_frame + int(duration * original_fps))
        print(f"  - End frame: {end_frame}")
    
    # Set start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    while frame_count < end_frame - start_frame:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame if it matches the interval
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = output_dir / frame_filename
            
            # Save frame
            success = cv2.imwrite(str(frame_path), frame)
            if success:
                saved_count += 1
                if saved_count % 100 == 0:
                    print(f"Saved {saved_count} frames...")
            else:
                print(f"Failed to save frame {frame_filename}")
        
        frame_count += 1
    
    # Clean up
    cap.release()
    
    print(f"✓ Frame extraction completed!")
    print(f"  - Extracted {saved_count} frames")
    print(f"  - Output directory: {output_dir}")
    
    return True

def extract_frames_with_timestamp(video_path, output_dir, save_timestamps=True):
    """
    Extract frames with timestamp information
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Extracting {total_frames} frames at {fps} fps")
    
    # Extract frames with timestamps
    frame_count = 0
    timestamps = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = frame_count / fps
        
        # Save frame
        frame_filename = f"frame_{frame_count:06d}.jpg"
        frame_path = output_dir / frame_filename
        
        success = cv2.imwrite(str(frame_path), frame)
        if success:
            timestamps.append((frame_filename, timestamp))
            
            if frame_count % 100 == 0:
                print(f"Saved {frame_count} frames...")
        else:
            print(f"Failed to save frame {frame_filename}")
        
        frame_count += 1
    
    # Save timestamps if requested
    if save_timestamps:
        timestamp_file = output_dir / "timestamps.txt"
        with open(timestamp_file, 'w') as f:
            f.write("frame_file,timestamp_seconds\n")
            for filename, timestamp in timestamps:
                f.write(f"{filename},{timestamp:.6f}\n")
        print(f"Saved timestamps to: {timestamp_file}")
    
    # Clean up
    cap.release()
    
    print(f"✓ Frame extraction completed!")
    print(f"  - Extracted {frame_count} frames")
    
    return True

if __name__ == "__main__":
    # Example usage
    video_path = "/home/phw/Downloads/output.mp4"
    output_dir = "data/query"
    
    # Extract all frames
    extract_frames_opencv(video_path, output_dir)
    
    # Or extract with specific parameters
    # extract_frames_opencv(video_path, output_dir, fps=1, start_time=10, duration=30)
    
    # Or extract with timestamps
    # extract_frames_with_timestamp(video_path, output_dir) 