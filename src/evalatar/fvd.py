import glob

from cdfvd.fvd import cdfvd as CDFVD
import numpy as np
import torch
from torchvision.io import read_video
from torchvision.transforms import Resize

# Add patch to resolve VideoMAE model loading issue
def patch_torch_load():
    """Add patch for VideoMAE model loading"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
        # Check if it's a VideoMAE model file
        if isinstance(f, str) and ('vit_g_hybrid_pt_1200e_ssv2_ft.pth' in f or 'videomae' in f.lower()):
            # For VideoMAE models, explicitly set weights_only=False
            weights_only = False
        elif weights_only is None:
            # Maintain default behavior for other cases
            weights_only = True
            
        try:
            return original_load(f, map_location=map_location, pickle_module=pickle_module,
                               weights_only=weights_only, **kwargs)
        except Exception:
            # If failed, fallback to weights_only=False
            return original_load(f, map_location=map_location, pickle_module=pickle_module,
                               weights_only=False, **kwargs)
    
    torch.load = patched_load

# Apply patch
patch_torch_load()

def load_videos_from_pattern(pattern, num_frames):
    """Load videos from file pattern and process to unified format"""
    video_paths = glob.glob(pattern)
    videos = []
    resize = Resize((224, 224))  # Ensure resolution matches model input

    for path in video_paths:
        # Read video (returns: (video_frames, audio, metadata))
        frames, _, _ = read_video(path, pts_unit='sec')  # Shape: (T, H, W, C), value range [0, 255], type uint8
        frames = frames[:num_frames]                     # Extract specified number of frames

        # Handle insufficient frames
        if len(frames) < num_frames:
            pad_length = num_frames - len(frames)
            frames = torch.cat([frames, frames[-1:].repeat(pad_length, 1, 1, 1)], dim=0)

        # Resize and convert to numpy array
        frames = resize(frames.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # (T, H, W, C)
        videos.append(frames.numpy().astype(np.uint8))

    return np.stack(videos, axis=0)                                      # Shape: (B, T, H, W, C)

def calculate_fvd(real_video_pattern, fake_video_pattern, num_frames=16, model='videomae', device='cuda'):
    """
    Interface implementation for FVD score calculation
    
    Parameters:
        real_video_pattern: File pattern for real videos (e.g., 'real/*.mp4')
        fake_video_pattern: File pattern for generated videos (e.g., 'generated/*.mp4')
        num_frames: Number of frames to use per video
        model: Feature extraction model, options: 'videomae' or 'i3d'
        device: Computation device, 'cuda' or 'cpu'
    
    Returns:
        FVD score
    """
    # Load and preprocess videos
    real_videos = load_videos_from_pattern(real_video_pattern, num_frames)
    fake_videos = load_videos_from_pattern(fake_video_pattern, num_frames)

    # Initialize FVD evaluator
    evaluator = CDFVD(
        model=model,
        n_real='full',  # Use all real videos
        n_fake='full',  # Use all generated videos
        device=device,
        compute_feats=False
    )

    # Calculate FVD
    with torch.no_grad():
        fvd_score = evaluator.compute_fvd(real_videos, fake_videos)
    
    return float(fvd_score)