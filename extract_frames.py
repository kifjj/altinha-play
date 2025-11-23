# usage: python extract_frames.py <video_path> <output_folder>
# uv run --with <library> <script_name>

# uv run extract_frames.py .\data\videos\altinha-beach-dark-mq-20s.mp4 .\data\videos\altinha-beach-dark-mq-20s-frames
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python",
#     "pandas",
# ]
# ///


import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_folder):
    """
    Extract all frames from a video and save them to the output folder.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the output folder where frames will be saved
    
    Returns:
        int: Number of frames extracted, or -1 if there was an error
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return -1
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        # Read the next frame
        success, frame = video.read()
        
        if not success:
            break
        
        # Generate output filename with zero-padded frame number
        # e.g., frame_00001.jpg, frame_00002.jpg, etc.
        frame_filename = f"frame_{frame_count:05d}.jpg"
        frame_path = output_path / frame_filename
        
        # Save the frame
        cv2.imwrite(str(frame_path), frame)
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count}/{total_frames} frames...")
    
    # Release the video capture object
    video.release()
    
    print(f"\nCompleted! Extracted {frame_count} frames to '{output_folder}'")
    return frame_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract all frames from a video file and save them as images."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder where frames will be saved"
    )
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' does not exist.")
        return
    
    # Extract frames
    extract_frames(args.video_path, args.output_folder)


if __name__ == "__main__":
    main()

