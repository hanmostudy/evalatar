import glob
import os

import cv2
from insightface.app import FaceAnalysis
import numpy as np
from tqdm import tqdm

def calculate_csim(
    generated_videos_path: str,
    reference_identity_path: str,
    device: str = 'cuda',
    frame_sample_rate: int = 1
) -> float:
    """
    Calculate CSIM (Cosine Similarity for Identity Matching) using InsightFace for modern and better compatibility
    
    This function evaluates identity consistency by comparing facial embeddings between reference image 
    and generated video frames using cosine similarity.
    
    :param generated_videos_path: Path to generated videos. Can be a single video file or a pattern like "path/to/videos/*.mp4"
    :param reference_identity_path: Path to reference identity image containing the target person's face
    :param device: Computing device, 'cpu' or 'cuda'
    :param frame_sample_rate: Frame sampling rate (Hz). For example, 1 means sampling one frame per second.
                              Set to 0 to evaluate all frames (slower).
    :return: Average cosine similarity score across all sampled frames
    """
    print("Loading InsightFace face recognition model...")
    
    # Initialize FaceAnalysis with appropriate execution provider
    app = FaceAnalysis(providers=['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Load reference image
    ref_img = cv2.imread(reference_identity_path)
    if ref_img is None:
        raise FileNotFoundError(f"Failed to load reference image: {reference_identity_path}")
    
    # Extract facial features from reference image
    ref_faces = app.get(ref_img)
    if len(ref_faces) == 0:
        raise ValueError("No face detected in reference image")
    
    ref_embedding = ref_faces[0].embedding
    
    # Get list of video files
    video_files = [generated_videos_path] if os.path.isfile(generated_videos_path) else glob.glob(generated_videos_path)
    if not video_files:
        raise FileNotFoundError(f"No video files found in path {generated_videos_path}.")

    all_scores = []
    
    # Process each video file
    for video_path in tqdm(video_files, desc="Evaluating video identity similarity"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Unable to open video {video_path}, skipped.")
            continue

        # Calculate frame skipping interval based on sampling rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(fps / frame_sample_rate)) if frame_sample_rate > 0 else 1
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame if it's a sampling frame
            if frame_idx % frame_skip == 0:
                try:
                    # Detect faces in current frame
                    faces = app.get(frame)
                    
                    if len(faces) > 0:
                        # Use the first detected face
                        face_embedding = faces[0].embedding
                        
                        # Calculate cosine similarity between reference and current face embeddings
                        similarity = np.dot(ref_embedding, face_embedding) / (
                            np.linalg.norm(ref_embedding) * np.linalg.norm(face_embedding)
                        )
                        all_scores.append(similarity)
                        
                except Exception as e:
                    # Ignore face detection failures but report other errors
                    if "Face could not be detected" not in str(e):
                        print(f"Error processing frame: {str(e)}")
            
            frame_idx += 1
            
        cap.release()

    # Handle case where no similarities were computed
    if not all_scores:
        print("Warning: Failed to compute similarity for any frames from any videos.")
        return 0.0

    # Calculate and return mean CSIM score
    mean_csim = np.mean(all_scores)
    print(f"\nEvaluation completed! Average identity cosine similarity (CSIM): {mean_csim:.6f}")
    return mean_csim