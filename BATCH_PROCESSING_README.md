# Altinha Batch Processing Guide

This guide explains how to use the refactored altinha video processing system to process multiple videos in batch.

## Files Overview

### 1. `altinha_processor.py`
A Python module containing all the ball tracking and hit detection logic. This can be imported and reused across different notebooks.

**Key Function:**
```python
process_altinha_video(
    video_path,
    model_path,
    output_path,
    pose_model_path='/kaggle/input/yolo11-pose/pytorch/default/1/yolo11l-pose.pt',
    debug_frames_dir=None,
    confidence_threshold=0.05,
    min_confidence=0.08,
    iou_nms=0.5,
    min_vertical_amplitude=3.0,
    min_frames_between_hits=8,
    gap_reset_frames=30,
    verbose=True
)
```

**Returns:** A dictionary with:
- `video_name`: Input video filename
- `output_path`: Path to annotated output video
- `json_path`: Path to JSON results file
- `ball_detection_percentage`: % of frames with ball detected
- `total_hits`, `head_hits`, `foot_hits`, `unknown_hits`: Hit counts
- `hit_detections`: List of hit metadata (frame number, type, player_id, timestamp)

### 2. `alta_infer_function.ipynb`
A simple notebook demonstrating how to process a single video using the module. Use this for testing or single video processing.

### 3. `batch_process_videos.ipynb` 
The main batch processing notebook that processes multiple videos in a loop.

## How to Use on Kaggle

### Step 1: Prepare Your Environment

1. Create a new Kaggle notebook
2. Add your video dataset as input data source
3. Add the YOLO models as input data sources:
   - Ball detection model: `altinha_best.pt`
   - Pose detection model: `yolo11l-pose.pt`

### Step 2: Upload the Processing Module

**Option A: Upload directly**
1. In Kaggle, click "Add data" → "Upload"
2. Upload `altinha_processor.py`

**Option B: From GitHub** (if you commit these files)
```python
!wget https://raw.githubusercontent.com/YOUR_USERNAME/altinha-play/main/altinha_processor.py
```

### Step 3: Import the Batch Processing Notebook

1. Upload `batch_process_videos.ipynb` to Kaggle
2. Or copy/paste its cells into a new notebook

### Step 4: Configure Video Paths

Edit the `VIDEO_PATHS` array in the batch processing notebook:

```python
VIDEO_PATHS = [
    '/kaggle/input/alta-videos/video1.mp4',
    '/kaggle/input/alta-videos/video2.mp4',
    '/kaggle/input/alta-videos/video3.mp4',
    # Add all your video paths here...
]
```

### Step 5: Run the Batch Processing

Run all cells in `batch_process_videos.ipynb`. The notebook will:

1. Install dependencies
2. Import the processing module
3. Loop through all videos
4. Process each video with ball tracking and hit detection
5. Generate annotated videos
6. Save JSON results for each video
7. Create a comprehensive batch summary

## Output Files

For each video processed, you'll get:

### Individual Video Outputs
- `{video_name}_annotated.mp4` - Annotated video with ball detection and hit counter
- `{video_name}_results.json` - JSON file with detailed results
- `debug_frames/{video_name}/` - Debug frames showing detected hits (if enabled)

### Batch Summary Outputs
- `batch_summary.json` - Comprehensive JSON with all results
- `batch_summary.csv` - CSV table for easy viewing in spreadsheet software
- `batch_results.zip` - Zip file containing all outputs (optional)

### Example `batch_summary.json` Structure

```json
{
  "batch_info": {
    "timestamp": "2025-11-27T10:30:00",
    "total_videos": 3,
    "successful": 3,
    "failed": 0,
    "total_processing_time_sec": 245.3
  },
  "aggregate_stats": {
    "total_hits": 127,
    "average_detection_percentage": 0.8543
  },
  "videos": [
    {
      "video_name": "altinha-beach-green-mq-13s.mp4",
      "status": "success",
      "output_path": "/kaggle/working/batch_results/altinha-beach-green-mq-13s_annotated.mp4",
      "json_path": "/kaggle/working/batch_results/altinha-beach-green-mq-13s_results.json",
      "ball_detection_percentage": 0.8721,
      "total_hits": 42,
      "head_hits": 28,
      "foot_hits": 12,
      "unknown_hits": 2,
      "hit_frames": [15, 32, 48, 67, ...]
    },
    ...
  ]
}
```

## How to Use on Google Colab

The process is similar to Kaggle:

1. Upload `altinha_processor.py` to Colab files
2. Mount your Google Drive if videos are stored there:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update paths to point to your Drive:
   ```python
   VIDEO_PATHS = [
       '/content/drive/MyDrive/videos/video1.mp4',
       '/content/drive/MyDrive/videos/video2.mp4',
   ]
   ```
4. Run the batch processing notebook

## Customizing Processing Parameters

You can customize the processing by passing parameters to `process_altinha_video()`:

```python
result = process_altinha_video(
    video_path=video_path,
    model_path=BALL_MODEL_PATH,
    output_path=output_video_path,
    pose_model_path=POSE_MODEL_PATH,
    debug_frames_dir=video_debug_dir,
    
    # Adjust these parameters:
    confidence_threshold=0.05,      # Lower = more sensitive ball detection
    min_confidence=0.08,            # Minimum confidence to keep detection
    min_vertical_amplitude=3.0,     # Minimum pixels for valid hit
    min_frames_between_hits=8,      # Prevent duplicate hit detection
    gap_reset_frames=30,            # Reset trajectory after gap
    verbose=True                    # Print progress messages
)
```

## Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'altinha_processor'`
- Ensure `altinha_processor.py` is in the working directory
- Check the file uploaded successfully
- Try restarting the kernel

### GPU/Memory Issues
- Process fewer videos at once
- Set `debug_frames_dir=None` to save memory
- Set `verbose=False` to reduce output

### Video Path Not Found
- Verify the video paths are correct
- Check if dataset is properly mounted in Kaggle/Colab
- Use absolute paths

## Example: Processing Videos from Different Sources

```python
# Mix of different video sources
VIDEO_PATHS = [
    '/kaggle/input/dataset1/video1.mp4',
    '/kaggle/input/dataset2/video2.mp4',
    '/content/drive/MyDrive/videos/video3.mp4',
]

# You can also generate paths dynamically:
import glob
VIDEO_PATHS = glob.glob('/kaggle/input/alta-videos/*.mp4')
```

## Next Steps

1. Review the annotated videos
2. Analyze the JSON results
3. Examine hit detection frames in the debug folders
4. Adjust parameters if needed and reprocess
5. Use the CSV summary for further analysis

## Support

For issues or questions:
- Check the original `alta-infer.ipynb` for reference
- Review the helper functions in `altinha_processor.py`
- Examine the debug frames to understand hit detection behavior

