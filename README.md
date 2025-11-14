# Evalatar

A Python library for evaluating the performance of digital human video generation models.

[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/hanmostudy/evalatar/blob/main/LICENSE)

## Introduction

Evalatar is a Python library specifically designed for evaluating the performance of digital human video generation models. It provides multiple evaluation metrics to help researchers and developers quantify the quality, identity consistency, synchronization, and other aspects of generated videos.

## Supported Evaluation Metrics

- **FID** (Fréchet Inception Distance): Measures the distance between generated videos and real videos in feature space
- **FVD** (Fréchet Video Distance): Video version of FID that considers the temporal dimension
- **CSIM** (Cosine Similarity for Identity Matching): Evaluates identity consistency using InsightFace
- **ASE** (Average Semantic Error): Semantic error based on MediaPipe facial landmark detection
- **SYNC** (Audio-Visual Synchronization): Audio-video synchronization evaluation (based on SyncNet)
- **IQA** (Image Quality Assessment): No-reference image quality assessment

## Installation

### Using pip

```bash
pip install evalatar
```

### From Source (Development)

To set up the project for development:

1. Clone the repository:
```bash
git clone https://github.com/hanmostudy/evalatar.git
cd evalatar
```

2. Install `uv` if you haven't already:
```bash
# On macOS and Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip:
pip install uv
```

3. Synchronize project dependencies using `uv`:
```bash
uv sync
```

4. Install PyTorch separately (can be GPU version):
```bash
# For CPU-only version
uv pip install torch==2.7.0 torchvision

# For CUDA 12.1 version
uv pip install torch==2.7.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
```

5. Activate the environment:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage Examples

```python
import evalatar

# Calculate FID score
fid_score = evalatar.calculate_fid(
    real_videos_path="path/to/real/videos/*.mp4",
    generated_videos_path="path/to/generated/videos/*.mp4",
    batch_size=50,
    dims=2048,
    target_fps=30,
    device=None  # Auto-detect CUDA or CPU
)

# Calculate FVD score
fvd_score = evalatar.calculate_fvd(
    real_video_pattern="path/to/real/videos/*.mp4",
    fake_video_pattern="path/to/generated/videos/*.mp4",
    num_frames=16,
    model='videomae',  # or 'i3d'
    device='cuda'      # or 'cpu'
)

# Calculate identity consistency
csim_score = evalatar.calculate_csim(
    generated_videos_path="path/to/generated/videos/*.mp4",
    reference_identity_path="path/to/reference/face.jpg",
    device='cuda',        # or 'cpu'
    frame_sample_rate=1   # Sample 1 frame per second
)

# Calculate audio-video synchronization
sync_c_score = evalatar.calculate_sync_c("path/to/video.mp4")
sync_d_score = evalatar.calculate_sync_d("path/to/video.mp4")

# Calculate semantic error
ase_score = evalatar.calculate_ase(
    real_videos_path="path/to/real/videos/*.mp4",
    generated_videos_path="path/to/generated/videos/*.mp4",
    device=None  # Auto-detect CUDA or CPU
)

# Calculate video quality
iqa_score = evalatar.calculate_iqa(
    generated_videos_path="path/to/generated/videos/*.mp4",
    device=None,          # Auto-detect CUDA or CPU
    metric_name='brisque', # or 'maniqa'
    frame_sample_rate=1   # Sample 1 frame per second
)
```

## Running Tests

To test the metric calculation functions, you can run the test suite:

```bash
# Run all tests
python -m pytest tests/test_evalatar_metrics.py -v -s

# Run specific metric tests
python -m pytest tests/test_evalatar_metrics.py -v -s -m fid  # FID tests only
python -m pytest tests/test_evalatar_metrics.py -v -s -m fvd  # FVD tests only
python -m pytest tests/test_evalatar_metrics.py -v -s -m sync # Sync tests only
python -m pytest tests/test_evalatar_metrics.py -v -s -m ase  # ASE tests only
python -m pytest tests/test_evalatar_metrics.py -v -s -m csim # CSIM tests only
python -m pytest tests/test_evalatar_metrics.py -v -s -m iqa  # IQA tests only
```

The tests will automatically download a sample YouTube video for evaluation if it's not already present in the test assets.

## Third-Party Components

This project incorporates components from several third-party open source projects:

1. **SyncNet** - Audio-visual synchronization neural network
   - Location: [src/evalatar/syncnet_python](src/evalatar/syncnet_python)
   - Original Author: Joon Son Chung
   - License: MIT License
   - Project URL: https://github.com/joonson/syncnet_python

2. **CDFVD** - For FVD (Fréchet Video Distance) calculations
   - PyPI Package: `cd-fvd`
   - License: MIT License

3. **PyTorch FID** - For FID (Fréchet Inception Distance) calculations
   - PyPI Package: `pytorch-fid`
   - License: Apache License 2.0

4. **PyIQA** - For IQA (Image Quality Assessment) calculations
   - PyPI Package: `pyiqa`
   - License: Apache License 2.0

## Troubleshooting

### FVD Model Download Issues

**Problem**: When using FVD with VideoMAE model, you may encounter model download failures or loading errors. This is due to the VideoMAE model URL being changed to Hugging Face.

**Solution**: 
1. Manually download the VideoMAE model from the new URL:
   ```
   https://huggingface.co/OpenGVLab/InternVideoMAE_models/resolve/main/mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth
   ```

2. Place the downloaded model file in the appropriate cache directory: `~/.venv/Lib/site-packages/cdfvd/third_party/VideoMAEv2/`

3. Alternatively, you can use the I3D model instead by specifying `model='i3d'` in the `calculate_fvd` function.

## Dependencies

- Python >= 3.11, < 3.12
- MediaPipe
- OpenCV
- PyTorch
- And other dependencies listed in [pyproject.toml](pyproject.toml)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues and pull requests to improve this project.