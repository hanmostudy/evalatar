import glob
import os

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

class _ASEEvaluator:
    """
    An internal helper class that encapsulates face landmark detection and calculation using Google MediaPipe.
    """
    _instance = None

    def __init__(self, device):
        # MediaPipe is fast enough on CPU, so we don't force GPU usage here
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Define the indices of the landmarks we care about among MediaPipe's 478 landmarks
        # (These indices can be found in the MediaPipe official documentation)
        self.LIPS_INDICES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            191, 80, 81, 82, 13, 312, 311, 310, 415,
            37, 0, 267, 269, 270, 409, 291
        ]
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def _get_landmarks(self, frame_rgb):
        """Extract 478 landmarks from a single frame."""
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        
        # Convert MediaPipe's NormalizedLandmark to a (478, 2) NumPy array
        # and multiply by image dimensions to get pixel coordinates
        h, w, _ = frame_rgb.shape
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([(lm.x * w, lm.y * h) for lm in landmarks])

    def _get_eye_centers(self, landmarks):
        left_eye_pts = landmarks[self.LEFT_EYE_INDICES]
        right_eye_pts = landmarks[self.RIGHT_EYE_INDICES]
        left_eye_center = left_eye_pts.mean(axis=0)
        right_eye_center = right_eye_pts.mean(axis=0)
        return right_eye_center, left_eye_center
        
    def _get_mouth_landmarks(self, landmarks):
        return landmarks[self.LIPS_INDICES]

    def calculate_normalized_distance(self, frame1_bgr, frame2_bgr):
        """Calculate normalized average distance of lip landmarks between two frames."""
        # MediaPipe requires RGB format
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)

        landmarks1 = self._get_landmarks(frame1_rgb)
        landmarks2 = self._get_landmarks(frame2_rgb)

        if landmarks1 is None or landmarks2 is None:
            return None

        # 1. Calculate normalization factor: Inter-Pupillary Distance (IPD)
        r_center1, l_center1 = self._get_eye_centers(landmarks1)
        ipd = np.linalg.norm(r_center1 - l_center1)
        if ipd == 0:
            return None

        # 2. Extract lip landmarks
        mouth1 = self._get_mouth_landmarks(landmarks1)
        mouth2 = self._get_mouth_landmarks(landmarks2)

        # 3. Calculate distances and normalize
        distances = np.linalg.norm(mouth1 - mouth2, axis=1) / ipd
        return distances.tolist()

    @classmethod
    def get_instance(cls, device):
        if cls._instance is None:
            print("Initializing MediaPipe face mesh detector...")
            cls._instance = cls(device)
        return cls._instance


def calculate_ase(
    real_videos_path: str,
    generated_videos_path: str,
    device: str = None
) -> float:
    """
    Calculate the Average Semantic Error (ASE) between two sets of videos.
    This function uses Google MediaPipe as the backend for efficient and accurate landmark detection.
    """
    # 1. Set device and get evaluator instance
    if device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(device)
    print(f"Using device: {compute_device}")
    
    evaluator = _ASEEvaluator.get_instance(compute_device)

    # 2. Find and pair video files
    real_files = sorted(glob.glob(real_videos_path))
    gen_files = sorted(glob.glob(generated_videos_path))

    if not real_files or not gen_files:
        raise FileNotFoundError("No video files found in one or both of the specified paths.")
    
    if len(real_files) != len(gen_files):
        print(f"Warning: Number of real videos ({len(real_files)}) does not match number of generated videos ({len(gen_files)}). Only comparing the common portion.")
        min_len = min(len(real_files), len(gen_files))
        real_files, gen_files = real_files[:min_len], gen_files[:min_len]

    all_distances = []
    
    # 3. Iterate through video pairs
    for real_path, gen_path in tqdm(zip(real_files, gen_files), total=len(real_files), desc="Processing video pairs"):
        cap_real = cv2.VideoCapture(real_path)
        cap_gen = cv2.VideoCapture(gen_path)

        if not cap_real.isOpened() or not cap_gen.isOpened():
            print(f"Warning: Unable to open video pair {os.path.basename(real_path)} / {os.path.basename(gen_path)}, skipped.")
            continue
        
        # 4. Compare frame by frame
        while True:
            ret_real, frame_real = cap_real.read()
            ret_gen, frame_gen = cap_gen.read()
            
            if not ret_real or not ret_gen:
                break
            
            distances = evaluator.calculate_normalized_distance(frame_real, frame_gen)
            if distances is not None:
                all_distances.extend(distances)
        
        cap_real.release()
        cap_gen.release()

    if not all_distances:
        print("Warning: Failed to calculate landmark distances on any video frames.")
        return 0.0

    # 5. Calculate final average
    mean_ase = np.mean(all_distances)
    
    print(f"\nEvaluation completed! Calculated distances on {len(all_distances)} landmarks in total.")
    print(f"Average Semantic Error (ASE): {mean_ase:.6f}")
    
    return mean_ase