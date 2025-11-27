# Altinha Video Processing Refactoring - Summary

## What Was Done

Your altinha video processing notebook has been successfully refactored to support batch processing of multiple videos. Here's what was created:

## New Files Created

### 1. **`altinha_processor.py`** - Core Processing Module
A standalone Python module containing all the ball tracking and hit detection logic. This can be imported and reused across different notebooks.

**Key Features:**
- All helper functions from the original notebook
- Main `process_altinha_video()` function that processes a single video
- Returns comprehensive results including ball detection percentage and hit frames
- Saves JSON output with detailed results
- Optional debug frame saving

### 2. **`alta_infer_function.ipynb`** - Single Video Processing Notebook
A streamlined notebook for processing individual videos.

**Use Cases:**
- Testing the processing on a single video
- Quick processing without batch setup
- Understanding how to use the module

### 3. **`batch_process_videos.ipynb`** - Batch Processing Notebook ⭐
The main batch processing notebook that handles multiple videos.

**Features:**
- Define an array of video paths
- Processes all videos in a loop
- Handles errors gracefully (continues if one video fails)
- Generates comprehensive summary reports
- Creates both JSON and CSV output
- Shows processing time and statistics
- Optional: Creates zip file for easy download

### 4. **`BATCH_PROCESSING_README.md`** - User Guide
Complete documentation on how to use the batch processing system in Kaggle/Colab.

### 5. **`REFACTORING_SUMMARY.md`** - This File
Summary of what was changed and how to use it.

## How the Refactoring Works

### Original Workflow (alta-infer.ipynb)
```
[Single Video] → [Hardcoded Config] → [Process] → [One Output]
```

### New Workflow
```
[Multiple Videos Array] → [Loop] → [Shared Function] → [Multiple Outputs + Summary]
```

## Key Changes from Original

1. **Modular Design**: All processing logic extracted to `altinha_processor.py`
2. **Parameterized Function**: Config values are now function parameters
3. **Reusable**: Can be called from any notebook or script
4. **Batch Support**: Easy to process multiple videos in sequence
5. **Better Output**: Returns structured data + saves JSON
6. **Error Handling**: Batch processing continues even if one video fails
7. **Summary Reports**: Aggregate statistics across all videos

## Quick Start Guide

### Step 1: Upload to Kaggle

Upload these files to your Kaggle notebook:
- `altinha_processor.py` (required)
- `batch_process_videos.ipynb` (or copy cells into new notebook)

### Step 2: Define Your Videos

In the batch notebook, update the `VIDEO_PATHS` array:

```python
VIDEO_PATHS = [
    '/kaggle/input/alta-videos/video1.mp4',
    '/kaggle/input/alta-videos/video2.mp4',
    '/kaggle/input/alta-videos/video3.mp4',
    # Add more...
]
```

### Step 3: Run All Cells

The notebook will automatically:
1. Install dependencies
2. Import the processing module
3. Process all videos
4. Save outputs for each video
5. Create batch summary (JSON + CSV)

### Step 4: Review Results

**Per-Video Outputs:**
- `{video_name}_annotated.mp4` - Annotated video
- `{video_name}_results.json` - Detailed results

**Batch Outputs:**
- `batch_summary.json` - All results combined
- `batch_summary.csv` - Spreadsheet-friendly summary
- `batch_results.zip` - All files in one download

## What You Get from Each Video

The `process_altinha_video()` function returns:

```python
{
    'video_name': 'altinha-beach-green-mq-13s.mp4',
    'output_path': '/kaggle/working/batch_results/..._annotated.mp4',
    'json_path': '/kaggle/working/batch_results/..._results.json',
    'ball_detection_percentage': 0.8721,  # 87.21% of frames detected
    'total_hits': 42,
    'head_hits': 28,
    'foot_hits': 12,
    'unknown_hits': 2,
    'hit_detections': [
        {'frame': 15, 'type': 'Head', 'player_id': 0, 'timestamp_sec': 0.5},
        {'frame': 32, 'type': 'Foot', 'player_id': 1, 'timestamp_sec': 1.07},
        ...
    ],
    'debug_frames_dir': '/kaggle/working/debug_frames/video_name'
}
```

## Comparison: Before vs After

### Before (Original alta-infer.ipynb)
- ❌ Hardcoded video path
- ❌ Must manually change path for each video
- ❌ Must re-run entire notebook for each video
- ❌ No aggregate statistics across videos
- ✅ Complete, working solution

### After (New System)
- ✅ Array of video paths
- ✅ Automatic loop through all videos
- ✅ Process multiple videos with one run
- ✅ Batch summary with aggregate stats
- ✅ Reusable module for other projects
- ✅ Error handling (one failure doesn't stop batch)
- ✅ Same complete functionality

## Example Usage

### Process Single Video
```python
from altinha_processor import process_altinha_video

result = process_altinha_video(
    video_path='/kaggle/input/alta-videos/video1.mp4',
    model_path='/kaggle/input/models/altinha_best.pt',
    output_path='/kaggle/working/output.mp4',
    pose_model_path='/kaggle/input/models/yolo11l-pose.pt'
)

print(f"Detected {result['total_hits']} hits!")
print(f"Ball detection: {result['ball_detection_percentage']:.1%}")
```

### Process Multiple Videos
```python
videos = [
    'video1.mp4',
    'video2.mp4',
    'video3.mp4'
]

for video in videos:
    result = process_altinha_video(
        video_path=f'/kaggle/input/alta-videos/{video}',
        model_path=MODEL_PATH,
        output_path=f'/kaggle/working/{video}',
        pose_model_path=POSE_MODEL_PATH
    )
    print(f"{video}: {result['total_hits']} hits")
```

## File Structure

```
alta-play/
├── alta-infer.ipynb                 # Original (kept for reference)
├── altinha_processor.py             # ⭐ NEW: Core processing module
├── alta_infer_function.ipynb        # ⭐ NEW: Single video notebook
├── batch_process_videos.ipynb       # ⭐ NEW: Batch processing notebook
├── BATCH_PROCESSING_README.md       # ⭐ NEW: User guide
├── REFACTORING_SUMMARY.md           # ⭐ NEW: This summary
└── [other existing files...]
```

## Migration Path

You have two options:

### Option 1: Keep Using Original (No Changes Needed)
- Continue using `alta-infer.ipynb` as before
- Nothing changes for you
- Good for: Single video processing

### Option 2: Switch to Batch Processing (Recommended for Multiple Videos)
- Use `batch_process_videos.ipynb` for multiple videos
- Define video array once
- Process all videos in one run
- Get aggregate statistics
- Good for: Processing multiple videos efficiently

## Backwards Compatibility

✅ The original `alta-infer.ipynb` still works exactly as before  
✅ No breaking changes to existing workflows  
✅ New system is additive - you can use both  

## Next Steps

1. **Test with Single Video**: Try `alta_infer_function.ipynb` first
2. **Try Batch Processing**: Use `batch_process_videos.ipynb` with 2-3 videos
3. **Scale Up**: Once working, add all your video paths
4. **Analyze Results**: Use the CSV summary for analysis

## Technical Details

### Function Signature
```python
def process_altinha_video(
    video_path: str,                          # Input video path
    model_path: str,                          # Ball detection model
    output_path: str,                         # Output video path
    pose_model_path: str = '...',            # Pose detection model
    debug_frames_dir: Optional[str] = None,  # Debug output (optional)
    confidence_threshold: float = 0.05,       # Detection sensitivity
    min_confidence: float = 0.08,            # Minimum confidence
    iou_nms: float = 0.5,                    # NMS threshold
    min_vertical_amplitude: float = 3.0,     # Hit detection threshold
    min_frames_between_hits: int = 8,        # Hit spacing
    gap_reset_frames: int = 30,              # Trajectory reset
    verbose: bool = True                     # Print progress
) -> Dict[str, object]:
    ...
```

### Return Value
```python
{
    'video_name': str,
    'output_path': str,
    'json_path': str,
    'ball_detection_percentage': float,      # 0.0 to 1.0
    'total_hits': int,
    'head_hits': int,
    'foot_hits': int,
    'unknown_hits': int,
    'hit_detections': List[Dict],           # Detailed hit data
    'debug_frames_dir': Optional[str]
}
```

## Support & Troubleshooting

See `BATCH_PROCESSING_README.md` for:
- Detailed setup instructions
- Troubleshooting common issues
- Parameter tuning guide
- Example configurations

## Summary

✅ **Created**: Reusable processing module  
✅ **Created**: Single video processing notebook  
✅ **Created**: Batch processing notebook  
✅ **Created**: Comprehensive documentation  
✅ **Maintained**: Original notebook still works  
✅ **Added**: Batch summary reports (JSON + CSV)  
✅ **Added**: Error handling for batch processing  
✅ **Added**: Frame-level hit detection data in output  

You now have a complete system for processing both individual videos and batches of videos efficiently!

