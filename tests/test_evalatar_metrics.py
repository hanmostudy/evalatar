import os
import shutil
import cv2
import pytest
import torch
from moviepy.editor import VideoFileClip, ImageClip
import logging

# Import our library's functions and necessary dependencies for frame extraction
from evalatar.fid import calculate_fid
from evalatar.fvd import calculate_fvd
from evalatar.sync import calculate_sync_c, calculate_sync_d
from evalatar.ase import calculate_ase
from evalatar.csim import calculate_csim
from evalatar.iqa import calculate_iqa
from insightface.app import FaceAnalysis # Import for smart frame extraction

# --- Constants for Test Configuration ---
TESTS_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(TESTS_DIR, "test_assets")
TEMP_DIR = os.path.join(TESTS_DIR, "temp_generated_assets")

# Use a real human face video, which is more reliable for face detection models.
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=CPCoMyuBRM0" 

# --- Helper Functions and Fixtures ---

def get_device():
    """Returns 'cuda' if available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_temp_dir():
    """Create and clean up the temporary directory for non-cached generated assets."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    yield
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

@pytest.fixture(scope="session", autouse=True)
def suppress_deep_logs():
    """Suppresses verbose logging from underlying libraries for cleaner test output."""
    libraries_to_suppress = ["onnxruntime", "pyiqa", "insightface", "moviepy", "yt_dlp"]
    original_levels = {}
    for lib_name in libraries_to_suppress:
        lib_logger = logging.getLogger(lib_name)
        original_levels[lib_name] = lib_logger.level
        lib_logger.setLevel(logging.WARNING)
    yield
    for lib_name, level in original_levels.items():
        logging.getLogger(lib_name).setLevel(level)

@pytest.fixture(scope="session")
def downloaded_video():
    """
    Downloads a real YouTube video if not already cached in `tests/test_assets`.
    """
    try:
        import yt_dlp
    except ImportError:
        pytest.fail("yt-dlp is not installed. Please run `pip install yt-dlp`.")
    
    video_filename = "Real_Video_Demo.mp4"
    video_path = os.path.join(ASSETS_DIR, video_filename)

    if not (os.path.exists(video_path) and os.path.getsize(video_path) > 0):
        print(f"\nDownloading test video to '{video_path}'...")
        ydl_opts = {
            'format': 'best[ext=mp4][height<=480]',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([TEST_VIDEO_URL])
            print("Download complete.")
        except Exception as e:
            pytest.fail(f"Failed to download test video: {e}")
    else:
        print(f"\nUsing cached test video: {os.path.basename(video_path)}")

    return video_path

@pytest.fixture(scope="session")
def test_assets():
    """Provides paths to local image assets."""
    assets_paths = {
        "face": os.path.join(ASSETS_DIR, "face.jpg"),
        "person_a": os.path.join(ASSETS_DIR, "person_a.jpg"),
        "person_b": os.path.join(ASSETS_DIR, "person_b.jpg"),
    }
    for name, path in assets_paths.items():
        if not os.path.exists(path):
            pytest.fail(f"Required test asset '{os.path.basename(path)}' not found in '{ASSETS_DIR}'.")
    return assets_paths

@pytest.fixture(scope="session")
def desynced_video(downloaded_video):
    """Dynamically creates a desynchronized version of the downloaded video."""
    desynced_video_path = os.path.join(TEMP_DIR, "desynced_video.mp4")
    with VideoFileClip(downloaded_video) as video_clip:
        if video_clip.audio is None:
            pytest.fail("Downloaded video has no audio track, cannot test sync.")
        audio = video_clip.audio
        shifted_audio = audio.set_start(0.5)
        final_clip = video_clip.set_audio(shifted_audio)
        final_clip.write_videofile(desynced_video_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
    return desynced_video_path

@pytest.fixture(scope="session")
def static_video(test_assets):
    """
    OPTIMIZATION: Creates a static video for reuse in multiple tests.
    """
    static_video_path = os.path.join(TEMP_DIR, "static_video.mp4")
    ImageClip(test_assets["face"]).set_duration(5).write_videofile(static_video_path, fps=24, logger=None)
    return static_video_path

# --- Test Functions (with markers for selective running) ---

@pytest.mark.fid
def test_fid_calculation(downloaded_video, static_video):
    print("\n[FID Test 1/2] Testing for near-zero FID (video vs. itself)...")
    fid_zero = calculate_fid(
        real_videos_path=downloaded_video,
        generated_videos_path=downloaded_video,
        device=get_device(), target_fps=5
    )
    print(f"  - Result: {fid_zero:.4f}")
    assert fid_zero == pytest.approx(0.0, abs=1e-4)

    print("[FID Test 2/2] Testing for high FID (dynamic video vs. static video)...")
    fid_high = calculate_fid(
        real_videos_path=downloaded_video,
        generated_videos_path=static_video,
        device=get_device(), target_fps=5
    )
    print(f"  - Result: {fid_high:.4f}")
    assert fid_high > 10.0

@pytest.mark.fvd
def test_fvd_calculation(downloaded_video, static_video):
    print("\n[FVD Test 1/2] Testing for near-zero FVD (video vs. itself)...")
    fvd_zero = calculate_fvd(
        real_video_pattern=downloaded_video,
        fake_video_pattern=downloaded_video,
        device=get_device(), model='videomae'
    )
    print(f"  - Result: {fvd_zero:.4f}")
    assert fvd_zero == pytest.approx(0.0, abs=1e-4)
    
    print("[FVD Test 2/2] Testing for high FVD (dynamic vs. static video)...")
    fvd_high = calculate_fvd(
        real_video_pattern=downloaded_video,
        fake_video_pattern=static_video,
        device=get_device(), model='i3d'
    )
    print(f"  - Result: {fvd_high:.4f}")
    assert fvd_high > 5.0

@pytest.mark.sync
def test_sync_calculation(downloaded_video, desynced_video):
    print("\n[Sync Test 1/2] Calculating scores for original (synced) video...")
    sync_c_good = calculate_sync_c(downloaded_video)
    sync_d_good = calculate_sync_d(downloaded_video)
    print(f"  - Synced Results: Sync-C={sync_c_good:.4f}, Sync-D={sync_d_good:.4f}")

    print("[Sync Test 2/2] Calculating scores for desynchronized video...")
    sync_c_bad = calculate_sync_c(desynced_video)
    sync_d_bad = calculate_sync_d(desynced_video)
    print(f"  - Desynced Results: Sync-C={sync_c_bad:.4f}, Sync-D={sync_d_bad:.4f}")

    assert sync_c_good > sync_c_bad
    assert sync_d_good < sync_d_bad

@pytest.mark.ase
def test_ase_calculation(downloaded_video):
    print("\n[ASE Test] Testing for near-zero ASE (video vs. itself)...")
    ase_score = calculate_ase(
        real_videos_path=downloaded_video,
        generated_videos_path=downloaded_video,
        device=get_device()
    )
    print(f"  - Result: {ase_score:.4f}")
    assert ase_score == pytest.approx(0.0, abs=1e-4)

@pytest.mark.csim
def test_csim_calculation(downloaded_video, test_assets):
    print("\n[CSIM Test] Extracting reference frame at the 1-minute mark...")
    cap = cv2.VideoCapture(downloaded_video)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps if fps > 0 else 0
    
    if duration_seconds < 60:
        pytest.fail(f"Test video is only {duration_seconds:.2f}s long, cannot extract frame from 1-min mark.")

    target_time_ms = 60 * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, target_time_ms)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        pytest.fail("Failed to read the frame at the 1-minute mark.")

    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(frame)
    if len(faces) == 0:
        pytest.fail("No face was detected in the frame at the 1-minute mark.")

    positive_ref_path = os.path.join(TEMP_DIR, "positive_ref_face.jpg")
    cv2.imwrite(positive_ref_path, frame)
    print("  - Successfully extracted and verified frame at 01:00.")

    print("\n[CSIM Test 1/2] Testing for high CSIM (video vs. its own face)...")
    csim_high = calculate_csim(
        generated_videos_path=downloaded_video,
        reference_identity_path=positive_ref_path,
        device=get_device(), frame_sample_rate=5
    )
    print(f"  - Result: {csim_high:.4f}")
    
    print("[CSIM Test 2/2] Testing for low CSIM (video vs. a different person's face)...")
    csim_low = calculate_csim(
        generated_videos_path=downloaded_video,
        reference_identity_path=test_assets["person_b"],
        device=get_device(), frame_sample_rate=5
    )
    print(f"  - Result: {csim_low:.4f}")

    assert csim_high > 0.7, "CSIM with a reference face from the same video should be high."
    assert csim_low < 0.5, "CSIM with a different person's face should be low."
    assert csim_high > csim_low, "Self-similarity score must be higher than with a negative sample."

@pytest.mark.iqa
def test_iqa_calculation(test_assets):
    clear_image_path = test_assets["face"]
    blurred_image_path = os.path.join(TEMP_DIR, "blurred_face.jpg")
    clear_img = cv2.imread(clear_image_path)
    blurred_img = cv2.GaussianBlur(clear_img, (51, 51), 0)
    cv2.imwrite(blurred_image_path, blurred_img)
    
    clear_video_path = os.path.join(TEMP_DIR, "clear_video.mp4")
    blurred_video_path = os.path.join(TEMP_DIR, "blurred_video.mp4")
    ImageClip(clear_image_path).set_duration(1).write_videofile(clear_video_path, fps=24, logger=None)
    ImageClip(blurred_image_path).set_duration(1).write_videofile(blurred_video_path, fps=24, logger=None)

    print("\n[IQA Test] Comparing IQA scores of a clear vs. a blurred image...")
    score_good = calculate_iqa(
        generated_videos_path=clear_video_path, 
        metric_name='brisque', device=get_device()
    )
    score_bad = calculate_iqa(
        generated_videos_path=blurred_video_path,
        metric_name='brisque', device=get_device()
    )
    print(f"  - Clear Image BRISQUE Score: {score_good:.4f}")
    print(f"  - Blurred Image BRISQUE Score: {score_bad:.4f}")

    assert score_good < score_bad, "The blurred image should have a worse (higher) BRISQUE score."

if __name__ == "__main__":
    print("Running tests directly...")
    pytest.main([__file__, "-v", "-s"])