import os
import tempfile
import warnings
import traceback

import cv2
import librosa
from moviepy.editor import VideoFileClip
import numpy as np

# Import necessary modules from SyncNet repository
try:
    from .syncnet_python.SyncNetInstance import SyncNetInstance
    SYNCNET_AVAILABLE = True
    print("[DEBUG] SyncNet module imported successfully.")
except ImportError:
    SYNCNET_AVAILABLE = False
    warnings.warn("SyncNet module not found, will use simplified version. Please ensure syncnet_python directory is in the correct location")
    print("[DEBUG] SyncNet module import FAILED.")

def _extract_audio_and_frames(video_path, target_fps=25):
    """Extract audio and video frames from video"""
    with VideoFileClip(video_path) as clip:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            clip.audio.write_audiofile(temp_audio_path, logger=None, fps=16000)
        
        audio_samples, samplerate = librosa.load(temp_audio_path, sr=16000)
        os.unlink(temp_audio_path)
        
        frames = []
        total_frames = int(clip.duration * target_fps)
        for i in range(total_frames):
            t = i / target_fps
            if t < clip.duration:
                frame = clip.get_frame(t)
                frames.append(frame.astype(np.uint8))
    
    return audio_samples, frames, samplerate

def _calculate_simple_sync(frames, audio_samples, samplerate, fps=25):
    """Simplified sync calculation (Fallback method)"""
    if len(frames) == 0 or len(audio_samples) == 0:
        return 0.5, 0.5
    
    samples_per_frame = samplerate / fps
    num_frames = min(len(frames), int(len(audio_samples) / samples_per_frame))
    
    if num_frames < 2:
        return 0.5, 0.5
    
    frame_brightness = []
    audio_energy = []
    
    for i in range(num_frames):
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        frame_brightness.append(brightness)
        
        start_sample = int(i * samples_per_frame)
        end_sample = int((i + 1) * samples_per_frame)
        if end_sample > len(audio_samples):
            end_sample = len(audio_samples)
        
        if start_sample < end_sample:
            frame_audio = audio_samples[start_sample:end_sample]
            energy = np.mean(np.abs(frame_audio))
        else:
            energy = 0
        audio_energy.append(energy)
    
    frame_brightness = np.array(frame_brightness)
    audio_energy = np.array(audio_energy)
    
    if len(frame_brightness) < 2 or len(audio_energy) < 2:
        return 0.5, 0.5
    
    # Avoid division by zero
    den_bright = (np.max(frame_brightness) - np.min(frame_brightness))
    if den_bright == 0: den_bright = 1e-8
    
    den_audio = (np.max(audio_energy) - np.min(audio_energy))
    if den_audio == 0: den_audio = 1e-8

    frame_brightness = (frame_brightness - np.min(frame_brightness)) / den_bright
    audio_energy = (audio_energy - np.min(audio_energy)) / den_audio
    
    try:
        correlation_matrix = np.corrcoef(frame_brightness, audio_energy)
        correlation = float(correlation_matrix[0, 1])
        
        if np.isnan(correlation) or np.isinf(correlation):
            correlation = 0.0
        
        sync_confidence = max(0.0, min(1.0, (correlation + 1) / 2))
        sync_distance = max(0.0, min(1.0, 1.0 - sync_confidence))
        
        return sync_confidence, sync_distance
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0.5, 0.5

def _create_syncnet_options(tmp_dir):
    """Create configuration parameters required by SyncNet"""
    class Options:
        def __init__(self):
            # --- FIX: Use absolute path to locate the model file ---
            # This ensures it works regardless of where the script is called from
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.initial_model = os.path.join(base_path, "syncnet_python", "data", "syncnet_v2.model")
            
            self.batch_size = 20
            self.vshift = 15
            self.tmp_dir = tmp_dir
            self.reference = "temp_sync"
    
    return Options()

def calculate_sync_c(video_path):
    """Calculate Sync-C (synchronization confidence)"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    
    if SYNCNET_AVAILABLE:
        print(f"\n[DEBUG] SyncNet module found. Attempting Sync-C for {os.path.basename(video_path)}...")
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Configure parameters
                opt = _create_syncnet_options(tmp_dir)
                
                print(f"[DEBUG] Loading SyncNet model from: {opt.initial_model}")
                if not os.path.exists(opt.initial_model):
                     print(f"[DEBUG] ERROR: Model file NOT found at {opt.initial_model}")
                else:
                     print(f"[DEBUG] Model file exists. Size: {os.path.getsize(opt.initial_model) / 1024 / 1024:.2f} MB")

                # Initialize SyncNet model
                s = SyncNetInstance()
                s.loadParameters(opt.initial_model)
                
                # Evaluate video
                _, conf, _ = s.evaluate(opt, video_path)
                
                print(f"[DEBUG] SyncNet raw confidence score: {conf}")
                
                # Normalize confidence to 0-1 range (SyncNet typically outputs > 0 for good sync)
                # Adjust normalization factor based on empirical data if needed
                sync_confidence = max(0.0, min(1.0, conf / 10.0))  
                return float(sync_confidence)
        except Exception as e:
            print(f"\n[DEBUG] SyncNet calculation CRASHED. Error: {e}")
            traceback.print_exc()
            print("[DEBUG] Falling back to simplified version...")
    else:
        print("\n[DEBUG] SyncNet module NOT available. Using simplified version directly.")
    
    # Use simplified version
    audio_samples, frames, samplerate = _extract_audio_and_frames(video_path)
    sync_confidence, _ = _calculate_simple_sync(frames, audio_samples, samplerate)
    print(f"[DEBUG] Simplified method Sync-C: {sync_confidence}")
    return float(sync_confidence)

def calculate_sync_d(video_path):
    """Calculate Sync-D (synchronization distance)"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    
    if SYNCNET_AVAILABLE:
        print(f"\n[DEBUG] SyncNet module found. Attempting Sync-D for {os.path.basename(video_path)}...")
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Configure parameters
                opt = _create_syncnet_options(tmp_dir)
                
                # Initialize SyncNet model
                s = SyncNetInstance()
                s.loadParameters(opt.initial_model)
                
                # Evaluate video
                _, _, dists = s.evaluate(opt, video_path)
                
                # Calculate average distance and normalize
                min_dist = np.mean(np.min(dists, axis=1))
                print(f"[DEBUG] SyncNet raw min distance: {min_dist}")

                sync_distance = max(0.0, min(1.0, min_dist / 20.0))  # Adjust according to typical value range
                return float(sync_distance)
        except Exception as e:
            print(f"\n[DEBUG] SyncNet calculation CRASHED. Error: {e}")
            print("[DEBUG] Falling back to simplified version...")
    
    # Use simplified version
    audio_samples, frames, samplerate = _extract_audio_and_frames(video_path)
    _, sync_distance = _calculate_simple_sync(frames, audio_samples, samplerate)
    print(f"[DEBUG] Simplified method Sync-D: {sync_distance}")
    return float(sync_distance)

# For compatibility, place functions in evalatar module
class evalatar:
    calculate_sync_c = staticmethod(calculate_sync_c)
    calculate_sync_d = staticmethod(calculate_sync_d)