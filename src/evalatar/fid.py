import glob
import os
import shutil
import tempfile

import cv2 
from tqdm import tqdm

from pytorch_fid.fid_score import calculate_fid_given_paths

# --- Internal helper function for extracting frames from videos ---
def _extract_frames(video_files: list, output_dir: str, target_fps: int):
    """
    Iterate through a list of video files, extract frames and save them to the specified output directory.
    """
    print(f"Extracting frames from {len(video_files)} videos to {output_dir}...")
    frame_counter = 0
    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Cannot open video {video_path}, skipping.")
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_fps <= 0 or frame_count <= 0:
            print(f"Warning: Invalid metadata for video {video_path}, skipping.")
            cap.release()
            continue
        
        # Calculate sampling interval
        frame_skip = round(original_fps / target_fps) if target_fps > 0 else int(original_fps)
        if frame_skip == 0:
            frame_skip = 1

        for frame_idx in range(0, frame_count, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Save frame as image file with unique filename
                output_filename = f"{video_name}_frame_{frame_idx:06d}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, frame)
                frame_counter += 1
        
        cap.release()
    print(f"Total extracted frames: {frame_counter}")


def calculate_fid(
    real_videos_path: str, 
    generated_videos_path: str, 
    batch_size: int = 50, 
    dims: int = 2048,
    target_fps: int = 30, 
    device: str = None
) -> float:
    """
    Calculate the Frechet Inception Distance (FID) between two sets of videos.
    
    This function uses the pytorch-fid library by first extracting video frames as images,
    then calculating the FID between image folders.

    :param real_videos_path: Path to folder containing real videos. Supports wildcards, e.g., "path/to/real/*.mp4".
    :param generated_videos_path: Path to folder containing generated videos. Supports wildcards, e.g., "path/to/gen/*.mp4".
    :param batch_size: Batch size used for pytorch-fid calculation.
    :param dims: Output dimension used by the pytorch-fid model.
    :param target_fps: Frame rate to sample from each video. For example, 1 means one frame per second.
    :param device: Computing device such as 'cuda' or 'cpu'. If None, CUDA will be automatically detected.
    :return: FID score (lower is better).
    """
    # 1. Find video files
    real_video_files = glob.glob(real_videos_path)
    generated_video_files = glob.glob(generated_videos_path)

    if not real_video_files:
        raise ValueError(f"No video files found at path {real_videos_path}.")
    if not generated_video_files:
        raise ValueError(f"No video files found at path {generated_videos_path}.")

    # 2. Create temporary directories to store extracted frames
    real_frames_dir = tempfile.mkdtemp()
    gen_frames_dir = tempfile.mkdtemp()
    
    fid_score = 0.0
    
    # 3. Use try...finally structure to ensure temporary directories are always deleted
    try:
        # 4. Extract frames from real and generated videos
        _extract_frames(real_video_files, real_frames_dir, target_fps)
        _extract_frames(generated_video_files, gen_frames_dir, target_fps)

        # 5. Set computing device
        if device is None:
            import torch
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            device_str = device
        print(f"Using device '{device_str}' for FID calculation.")

        # 6. Call pytorch-fid core function to perform calculation
        print("Starting FID score calculation using pytorch-fid...")
        fid_score = calculate_fid_given_paths(
            paths=[real_frames_dir, gen_frames_dir],
            batch_size=batch_size,
            device=device_str,
            dims=dims  
        )

    finally:
        # 7. Clean up temporary directories
        print("Cleaning up temporary files...")
        shutil.rmtree(real_frames_dir)
        shutil.rmtree(gen_frames_dir)
        print("Cleanup completed.")
        
    return fid_score