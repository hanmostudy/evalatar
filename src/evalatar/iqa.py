import glob
import os  # This import was missing but needed for the os.path.basename call

import cv2
import numpy as np
import pyiqa
import torch
from tqdm import tqdm

def calculate_iqa(
    generated_videos_path: str,
    device: str = None,
    metric_name: str = 'brisque',
    frame_sample_rate: int = 1
) -> float:
    """
    Calculate the average no-reference image quality assessment (NR-IQA) score for all videos in a given folder.

    This function uses the pyiqa library to evaluate video frames through sampling, and calculates the
    average score across all frames.

    :param generated_videos_path: Path to the folder containing generated videos. Supports wildcards,
                                  such as "path/to/gen/*.mp4".
    :param device: Computing device, such as 'cuda', 'cpu'. If None, CUDA will be automatically detected.
    :param metric_name: Name of the IQA algorithm to use. Must be an algorithm supported by the pyiqa library.
                        - 'brisque': Classic, fast, CPU-friendly (lower scores are better).
                        - 'maniqa': Advanced, Transformer-based, requires GPU (higher scores are better).
                        Defaults to 'brisque'.
    :param frame_sample_rate: Frame sampling rate (Hz). For example, 1 means sampling one frame per second for evaluation.
                              Set to 0 to evaluate all frames (slower).
    :return: Average IQA score across all sampled frames from all videos.
    """
    # 1. Automatically detect and set computing device
    if device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(device)
    print(f"Using device: {compute_device}")

    # 2. Create IQA evaluator based on name
    # When using certain models (such as MANIQA) for the first time, pyiqa will automatically download pretrained weights
    try:
        iqa_metric = pyiqa.create_metric(metric_name, device=compute_device)
        print(f"Successfully created IQA evaluator: {metric_name.upper()}")
        # Check whether higher or lower scores are better for this metric
        lower_is_better = iqa_metric.lower_better
        print(f"Whether lower scores are better for this metric: {lower_is_better}")
    except Exception as e:
        raise ValueError(f"Unable to create IQA metric named '{metric_name}'. Please ensure it is an algorithm supported by pyiqa. Error: {e}")

    # 3. Find all video files
    video_files = glob.glob(generated_videos_path)
    if not video_files:
        raise FileNotFoundError(f"No video files found in path {generated_videos_path}.")

    all_frame_scores = []
    
    # 4. Iterate through each video file
    for video_path in tqdm(video_files, desc="Evaluating video quality"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Unable to open video {video_path}, skipped.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate number of frames to skip based on sampling rate
        frame_skip = int(fps / frame_sample_rate) if frame_sample_rate > 0 else 1
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
                frame_tensor = frame_tensor.to(compute_device)
            
                try:
                    with torch.no_grad():
                        score = iqa_metric(frame_tensor)
                    all_frame_scores.append(score.item())
                except AssertionError as e:
                    # If pyiqa internally fails due to zero variance or other issues, catch it
                    print(f"\nWarning: Skipping a frame in {os.path.basename(video_path)} due to calculation error: {e}")
                    continue # Continue processing the next frame
                # ----------------------------------------------------
        
            frame_idx += 1

    if not all_frame_scores:
        print("Warning: Failed to evaluate any frames from any videos.")
        return 0.0

    # 5. Calculate and return the average of all scores
    average_score = np.mean(all_frame_scores)
    
    print(f"\nEvaluation completed! Evaluated a total of {len(all_frame_scores)} frames.")
    print(f"Average {metric_name.upper()} score: {average_score:.4f}")
    
    return average_score