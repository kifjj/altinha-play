"""
Altinha Ball Tracking and Hit Detection - Reusable Module

This module provides functions for ball tracking and hit detection
that can be imported and used from Jupyter notebooks.
"""

import json
import os
import shutil
import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Optional, Sequence, Tuple
from ultralytics import YOLO


# Type aliases
BallPosition = Tuple[int, float, float, np.ndarray]
HitDetections = List[Dict[str, object]]

# Global parameter (can be overridden)
MIN_VERTICAL_AMPLITUDE = 3


def filter_best_ball_detection(detections: sv.Detections, min_confidence: float) -> sv.Detections:
    """Filter detections to keep only the best one (highest confidence)."""
    if len(detections) == 0:
        return detections
    
    best_idx = int(np.argmax(detections.confidence))
    best_conf = float(detections.confidence[best_idx])
    
    if best_conf >= min_confidence:
        return detections[best_idx:best_idx+1]
    
    return detections[0:0]


def update_ball_tracking_state(
    detections: sv.Detections,
    n_frame: int,
    last_ball_positions: List[BallPosition],
    last_ball_detection_n_frame: Optional[int],
    gap_reset_frames: int,
) -> Tuple[List[BallPosition], Optional[int]]:
    """Update ball tracking state with new detection."""
    if len(detections) == 0:
        return last_ball_positions, last_ball_detection_n_frame
    
    if last_ball_detection_n_frame is not None:
        gap = n_frame - last_ball_detection_n_frame
        if gap > gap_reset_frames:
            last_ball_positions = []
    
    bbox = detections.xyxy[0]
    x1, y1, x2, y2 = bbox.tolist()
    x_center = 0.5 * (x1 + x2)
    y_center = 0.5 * (y1 + y2)
    
    last_ball_positions = last_ball_positions.copy()
    last_ball_positions.append((n_frame, x_center, y_center, bbox))
    if len(last_ball_positions) > 3:
        last_ball_positions.pop(0)
    
    return last_ball_positions, n_frame


def detect_hit(last_positions: List[BallPosition]) -> Tuple[bool, Optional[int], Optional[float], Optional[float], Optional[np.ndarray]]:
    """Detect if a hit occurred based on ball trajectory."""
    if len(last_positions) != 3:
        return False, None, None, None, None
    
    (f0, x0, y0, bbox0), (f1, x1c, y1c, bbox1c), (f2, x2c, y2c, bbox2c) = last_positions
    
    going_down_then_up = (y0 < y1c) and (y2c < y1c)
    vertical_span = y1c - min(y0, y2c)
    
    if going_down_then_up and vertical_span >= MIN_VERTICAL_AMPLITUDE:
        return True, f1, y1c, vertical_span, bbox1c
    
    return False, None, None, None, None


def get_pose_keypoints(frame: np.ndarray, model_pose: YOLO) -> List[np.ndarray]:
    """Run YOLO-pose model to detect player keypoints."""
    results = model_pose(frame, verbose=False, conf=0.3)[0]
    
    if results.keypoints is None or len(results.keypoints.data) == 0:
        return []
    
    poses = []
    for person_keypoints in results.keypoints.data:
        poses.append(person_keypoints.cpu().numpy())
    
    return poses


def find_closest_player(ball_center: Tuple[float, float], poses: Sequence[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
    """Find the player closest to the ball."""
    if not poses:
        return None, -1
    
    ball_x, ball_y = ball_center
    min_distance = float('inf')
    closest_player_idx = -1

    for i, pose in enumerate(poses):
        valid_keypoints = pose[pose[:, 2] > 0.3]
        if len(valid_keypoints) == 0:
            continue
        
        player_x = np.mean(valid_keypoints[:, 0])
        player_y = np.mean(valid_keypoints[:, 1])
        
        distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_player_idx = i
    
    if closest_player_idx == -1:
        return None, -1
    
    return poses[closest_player_idx], closest_player_idx


def classify_hit_type(ball_bbox: np.ndarray, player_pose: Optional[np.ndarray], verbose: bool = False) -> str:
    """Classify hit type (Head/Foot/Unknown) based on ball position and player keypoints."""
    if player_pose is None:
        if verbose:
            print("  [CLASSIFY] No player pose detected -> Unknown")
        return 'Unknown'
    
    ball_x = (ball_bbox[0] + ball_bbox[2]) / 2
    ball_y = (ball_bbox[1] + ball_bbox[3]) / 2
    
    head_indices = [0, 3, 4]
    foot_indices = [15, 16]
    
    head_distances = []
    for idx in head_indices:
        if idx < len(player_pose) and player_pose[idx, 2] > 0.3:
            kp_x, kp_y = player_pose[idx, 0], player_pose[idx, 1]
            dist = np.sqrt((ball_x - kp_x)**2 + (ball_y - kp_y)**2)
            head_distances.append(dist)
    
    foot_distances = []
    for idx in foot_indices:
        if idx < len(player_pose) and player_pose[idx, 2] > 0.3:
            kp_x, kp_y = player_pose[idx, 0], player_pose[idx, 1]
            dist = np.sqrt((ball_x - kp_x)**2 + (ball_y - kp_y)**2)
            foot_distances.append(dist)
    
    min_head_dist = min(head_distances) if head_distances else float('inf')
    min_foot_dist = min(foot_distances) if foot_distances else float('inf')
    
    if min_head_dist == float('inf') and min_foot_dist == float('inf'):
        return 'Unknown'
    
    distance_threshold = 80
    
    if min_head_dist < distance_threshold and min_head_dist < min_foot_dist:
        return 'Head'
    elif min_foot_dist < distance_threshold:
        return 'Foot'
    else:
        return 'Unknown'


def draw_debug_keypoints(frame: np.ndarray, player_pose: Optional[np.ndarray]) -> np.ndarray:
    """Draw debug keypoints on frame."""
    if player_pose is None:
        return frame
    
    keypoint_indices = [0, 3, 4, 15, 16]
    keypoint_names = ['nose', 'left_ear', 'right_ear', 'left_ankle', 'right_ankle']
    colors = [(0, 0, 255)] * 5
    
    for idx, name, color in zip(keypoint_indices, keypoint_names, colors):
        if idx < len(player_pose) and player_pose[idx, 2] > 0.3:
            kp_x, kp_y = int(player_pose[idx, 0]), int(player_pose[idx, 1])
            
            box_size = 5
            cv2.rectangle(
                frame,
                (kp_x - box_size, kp_y - box_size),
                (kp_x + box_size, kp_y + box_size),
                color,
                thickness=2
            )
            
            cv2.putText(
                frame, name, (kp_x + box_size + 2, kp_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )
    
    return frame


def save_debug_frames(
    frame_buffer: Sequence[Tuple[int, np.ndarray]],
    hit_frame: int,
    hit_number: int,
    debug_dir: str,
) -> None:
    """Save frames around a detected hit for debugging."""
    os.makedirs(debug_dir, exist_ok=True)
    
    frames_to_save = [hit_frame - 1, hit_frame, hit_frame + 1]
    
    for frame_idx, frame_image in frame_buffer:
        if frame_idx in frames_to_save:
            filename = f"hit-{hit_number}-frame-{frame_idx}.png"
            filepath = os.path.join(debug_dir, filename)
            cv2.imwrite(filepath, frame_image)


def check_and_record_hit(
    last_ball_positions: List[BallPosition],
    hit_detections: HitDetections,
    fps: float,
    min_frames_between_hits: int,
    frame: np.ndarray,
    model_pose: YOLO,
    frame_buffer: Optional[Sequence[Tuple[int, np.ndarray]]] = None,
    debug_dir: Optional[str] = None,
    verbose: bool = True,
) -> HitDetections:
    """Check for hit and record it if valid."""
    is_hit, hit_n_frame, hit_y, span, hit_bbox = detect_hit(last_ball_positions)
    
    if is_hit:
        last_hit_frame = hit_detections[-1]['frame'] if hit_detections else None
        if not hit_detections or (hit_n_frame - last_hit_frame) >= min_frames_between_hits:
            hit_frame = frame
            if frame_buffer is not None:
                for frame_idx, frame_image in frame_buffer:
                    if frame_idx == hit_n_frame:
                        hit_frame = frame_image
                        break
            
            poses = get_pose_keypoints(hit_frame, model_pose)
            
            ball_center = ((hit_bbox[0] + hit_bbox[2]) / 2, (hit_bbox[1] + hit_bbox[3]) / 2)
            
            player_pose, player_id = find_closest_player(ball_center, poses)
            
            hit_type = classify_hit_type(hit_bbox, player_pose, verbose=verbose)
            
            hit_detections = hit_detections.copy()
            hit_detections.append({
                'frame': hit_n_frame,
                'type': hit_type,
                'player_id': player_id,
                'player_pose': player_pose
            })
            t_sec = hit_n_frame / fps
            hit_number = len(hit_detections)
            
            if verbose:
                print(f"HIT #{hit_number} at frame {hit_n_frame} (t={t_sec:.2f}s), Type: {hit_type}, Player: {player_id}")
            
            if frame_buffer is not None and debug_dir is not None:
                save_debug_frames(frame_buffer, hit_n_frame, hit_number, debug_dir)
            
            return hit_detections
    
    return hit_detections


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    hit_detections: HitDetections,
    n_frame: int,
) -> np.ndarray:
    """Annotate frame with ball detection and hit counter."""
    annotated_frame = frame.copy()
    
    if len(detections) > 0:
        conf = float(detections.confidence[0])
        labels = [f"Ball {conf:.2f}"]
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )
    
    if hit_detections:
        last_hit = hit_detections[-1]
        frames_since_hit = n_frame - last_hit['frame']
        if 0 <= frames_since_hit <= 10 and 'player_pose' in last_hit:
            annotated_frame = draw_debug_keypoints(annotated_frame, last_hit['player_pose'])
    
    annotated_frame = draw_hit_counter(annotated_frame, hit_detections)
    
    return annotated_frame


def draw_hit_counter(frame: np.ndarray, hit_detections: Sequence[Dict[str, object]]) -> np.ndarray:
    """Draw hit counter HUD on frame."""
    total_hits = len(hit_detections)
    head_hits = sum(1 for h in hit_detections if h['type'] == 'Head')
    foot_hits = sum(1 for h in hit_detections if h['type'] == 'Foot')
    unknown_hits = sum(1 for h in hit_detections if h['type'] == 'Unknown')
    
    hit_text = f"Hits: {total_hits} | Head: {head_hits} | Foot: {foot_hits} | Unknown: {unknown_hits}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(
        hit_text, font, font_scale, thickness
    )
    
    pad_x, pad_y = 10, 10
    x1, y1 = 10, 10
    x2 = x1 + text_width + 2 * pad_x
    y2 = y1 + text_height + 2 * pad_y
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    
    text_x = x1 + pad_x
    text_y = y1 + pad_y + text_height
    cv2.putText(
        frame, hit_text, (text_x, text_y),
        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
    )
    
    return frame


def process_altinha_video(
    video_path: str,
    model_path: str,
    output_path: str,
    pose_model_path: str = '/kaggle/input/yolo11-pose/pytorch/default/1/yolo11l-pose.pt',
    debug_frames_dir: Optional[str] = None,
    confidence_threshold: float = 0.05,
    min_confidence: float = 0.08,
    iou_nms: float = 0.5,
    min_vertical_amplitude: float = 3.0,
    min_frames_between_hits: int = 8,
    gap_reset_frames: int = 30,
    verbose: bool = True
) -> Dict[str, object]:
    """
    Process an altinha video for ball tracking and hit detection.
    
    Args:
        video_path: Path to input video
        model_path: Path to ball detection YOLO model
        output_path: Path for output annotated video
        pose_model_path: Path to YOLO pose model
        debug_frames_dir: Directory to save debug frames (None to skip)
        confidence_threshold: Minimum confidence for initial detection
        min_confidence: Minimum confidence to keep a detection
        iou_nms: NMS IoU threshold
        min_vertical_amplitude: Minimum pixels for a valid hit
        min_frames_between_hits: Minimum frames between consecutive hits
        gap_reset_frames: Frames without detection before resetting trajectory
        verbose: Print progress messages
        
    Returns:
        Dictionary containing:
            - 'video_name': Input video filename
            - 'output_path': Path to output video
            - 'json_path': Path to JSON summary
            - 'ball_detection_percentage': Percentage of frames with ball detected
            - 'total_hits': Total number of hits detected
            - 'head_hits': Number of head hits
            - 'foot_hits': Number of foot hits
            - 'unknown_hits': Number of unknown hits
            - 'hit_detections': List of hit metadata dicts with 'frame', 'type', 'player_id'
            - 'debug_frames_dir': Directory with debug frames (if enabled)
    """
    global MIN_VERTICAL_AMPLITUDE
    MIN_VERTICAL_AMPLITUDE = min_vertical_amplitude
    
    # Copy pose model to writable directory
    pose_model_writable = '/kaggle/working/yolo11l-pose.pt'
    if not os.path.exists(pose_model_writable):
        shutil.copy(pose_model_path, pose_model_writable)
    
    # Load models
    model = YOLO(model_path)
    model_pose = YOLO(pose_model_writable)
    
    # Setup annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.Color.from_hex("#00FF00")
    )
    
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=2,
        text_position=sv.Position.TOP_CENTER,
    )
    
    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path)
    fps = video_info.fps
    frames_generator = sv.get_video_frames_generator(video_path)
    
    if verbose:
        print(f"\n" + "="*60)
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"FPS: {fps}, Resolution: {video_info.width}x{video_info.height}")
        print("="*60)
    
    # Initialize tracking state
    last_ball_positions = []
    hit_detections = []
    last_ball_detection_n_frame = None
    frame_buffer = []
    previous_frame = None
    
    processed_frames = 0
    frames_with_ball_detection = 0
    
    # Process video
    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for n_frame, frame in enumerate(frames_generator, start=1):
            processed_frames += 1
            
            if previous_frame is None:
                previous_frame = frame
            
            # Add current frame to buffer
            if debug_frames_dir is not None:
                frame_buffer.append((n_frame, frame.copy()))
                if len(frame_buffer) > 3:
                    frame_buffer.pop(0)
            
            # Run YOLO detection
            results = model(
                frame,
                verbose=False,
                conf=confidence_threshold,
                iou=iou_nms,
            )[0]
            
            ball_detections = sv.Detections.from_ultralytics(results)
            ball_detections = filter_best_ball_detection(ball_detections, min_confidence)
            
            if len(ball_detections) > 0:
                frames_with_ball_detection += 1
            
            # Update tracking state
            last_ball_positions, last_ball_detection_n_frame = update_ball_tracking_state(
                ball_detections,
                n_frame,
                last_ball_positions,
                last_ball_detection_n_frame,
                gap_reset_frames
            )
            
            # Check for hit
            if len(ball_detections) > 0:
                hit_detections = check_and_record_hit(
                    last_ball_positions,
                    hit_detections,
                    fps,
                    min_frames_between_hits,
                    previous_frame,
                    model_pose,
                    frame_buffer=frame_buffer if debug_frames_dir else None,
                    debug_dir=debug_frames_dir,
                    verbose=verbose
                )
            
            # Annotate frame
            annotated_frame = annotate_frame(
                frame, ball_detections, box_annotator, label_annotator, hit_detections, n_frame
            )
            
            sink.write_frame(annotated_frame)
            
            previous_frame = frame
    
    detection_percentage = frames_with_ball_detection / processed_frames if processed_frames > 0 else 0.0
    
    # Calculate statistics
    total_hits = len(hit_detections)
    head_hits = sum(1 for h in hit_detections if h['type'] == 'Head')
    foot_hits = sum(1 for h in hit_detections if h['type'] == 'Foot')
    unknown_hits = sum(1 for h in hit_detections if h['type'] == 'Unknown')
    
    if verbose:
        print(f"\n✅ Done! Video saved to {output_path}")
        print(f"📊 Total hits detected: {total_hits}")
        print(f"⚽ Ball detection: {detection_percentage:.1%}")
        if debug_frames_dir:
            print(f"🐛 Debug frames saved to: {debug_frames_dir}")
    
    # Save JSON summary
    video_filename = os.path.basename(video_path)
    json_output_path = os.path.join(
        os.path.dirname(output_path) or ".",
        os.path.splitext(video_filename)[0] + "_results.json"
    )
    
    # Create simplified hit detections for JSON (remove numpy arrays)
    hit_detections_json = [
        {
            'frame': h['frame'],
            'type': h['type'],
            'player_id': h['player_id'],
            'timestamp_sec': h['frame'] / fps
        }
        for h in hit_detections
    ]
    
    summary_data = {
        "video_name": video_filename,
        "ball_detection_percentage": round(detection_percentage, 4),
        "total_hits": total_hits,
        "head_hits": head_hits,
        "foot_hits": foot_hits,
        "unknown_hits": unknown_hits,
        "hit_detections": hit_detections_json
    }
    
    with open(json_output_path, "w") as json_file:
        json.dump(summary_data, json_file, indent=2)
    
    if verbose:
        print(f"📄 Results saved to {json_output_path}")
    
    # Return results
    return {
        'video_name': video_filename,
        'output_path': output_path,
        'json_path': json_output_path,
        'ball_detection_percentage': detection_percentage,
        'total_hits': total_hits,
        'head_hits': head_hits,
        'foot_hits': foot_hits,
        'unknown_hits': unknown_hits,
        'hit_detections': hit_detections,
        'debug_frames_dir': debug_frames_dir,
    }

