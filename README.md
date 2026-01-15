# Two-Stage Temporal Action Recognition for Industrial Assembly

Production-ready temporal action recognition system for industrial assembly process analysis.

## 🎯 Overview

This system implements a **two-stage hierarchical action recognition pipeline** that:
1. **Stage 1:** Identifies coarse action categories (FirstWord strategy)
2. **Stage 2:** Refines predictions with fine-grained action classification per family

The architecture strictly follows the reference notebook methodology for stable, production-aligned temporal modeling.

## 🏗️ Architecture

```
Video Input
    ↓
R3D-18 Feature Extractor (512-D per chunk)
    ↓
Stage-1: Coarse Action Classifier (14 classes)
    ↓
Temporal Smoothing + Segment Merging
    ↓
Stage-2: Fine-Grained Classifiers (per family)
    ↓
Action Timeline Output
```

### Key Components

- **Visual Backbone:** R3D-18 (ResNet3D-18) pretrained on Kinetics-400
- **Temporal Classifier:** MLP with LayerNorm, Dropout, and ReLU activations
- **Feature Dimension:** 512-D per temporal chunk
- **Temporal Sampling:** 8 FPS, 16 frames per chunk

## 📊 Dataset

- **Total Clips:** 48 video clips
- **Coarse Actions:** 14 (FirstWord strategy)
- **Fine-Grained Actions:** 24
- **Value Categories:** VA, RNVA, NVA

### Labeling Strategy

**FirstWord (Coarse):**
- `hand_tight_nipple_to_pipe_va` → `hand`
- `apply_greace_to_o_ring_va` → `apply`
- `mount_o_ring_to_pipe_va` → `mount`

**Full Label (Fine-Grained):**
- Complete folder name preserved

**Value Category:**
- `_va` → Value-Added
- `_rnva` → Required Non-Value-Added
- `_nva` → Non-Value-Added

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/PathanWasim/temporal_model_two_stage_pipeline.git
cd temporal_model_two_stage_pipeline

# Install dependencies
pip install -r requirements.txt

# Or run setup script for validation
python setup.py
```

**System Requirements:**
- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)
- OpenCV, NumPy, Pandas, tqdm

### Dataset Setup

**Create your dataset structure:**
```bash
# Training videos (filename-based labeling)
dataset/
├── apply_loctite_to_nipple_rnva_001.mp4
├── hand_tight_nipple_to_pipe_va_001.mp4
├── mount_o_ring_to_pipe_va_001.mp4
└── ...

# Test videos
test/
├── apply_loctite_to_nipple_rnva_002.mp4
├── fill_mes_rnva_002.mp4
└── ...
```

**Filename Format:** `<action_name>_<va|rnva|nva>_<id>.mp4`
- `va` = Value-Added
- `rnva` = Required Non-Value-Added  
- `nva` = Non-Value-Added

### Training (Research System - FIXED_T=64)

```bash
python train_two_stage_model.py
```

### Training (Deployment System - FIXED_T=16)

```bash
python train_deployment_model.py
```

**Training Configuration:**
- Epochs: 40 (Stage-1), 50 (Stage-2)
- Batch Size: 4
- Learning Rate: 1e-3
- Mixed Precision: FP16 (enabled on CUDA)
- Hardware: NVIDIA RTX 4050 (or any CUDA GPU)

**Outputs:**
- Research: `outputs_two_stage/` (FIXED_T=64)
- Deployment: `outputs_deployment/` (FIXED_T=16)

### Inference (Research System)

```bash
python inference_two_stage.py
```

**Output:** JSON and TXT timelines in `inference_results/`

### Inference (Deployment System - Industry Ready)

```bash
python inference_deployment.py
```

**Input:** Videos in `test/` directory  
**Output:** 
- JSON timelines in `inference_results/json/`
- Annotated videos in `inference_results/videos/`

## 📁 Project Structure

```
.
├── train_two_stage_model.py          # Research training (FIXED_T=64)
├── inference_two_stage.py            # Research inference
├── train_deployment_model.py         # Deployment training (FIXED_T=16)
├── inference_deployment.py           # Deployment inference with video overlays
├── TRAINING_SUMMARY.md               # Detailed training report
├── README.md                         # This file
├── dataset/                          # Training videos (flat structure)
│   ├── apply_loctite_to_nipple_rnva_001.mp4
│   ├── hand_tight_nipple_to_pipe_va_001.mp4
│   └── ...
├── test/                             # Test videos
│   ├── apply_loctite_to_nipple_rnva_002.mp4
│   ├── fill_mes_rnva_002.mp4
│   └── ...
├── outputs_two_stage/                # Research models (FIXED_T=64)
│   ├── stage1/
│   ├── stage2/
│   └── feat_cache/
├── outputs_deployment/               # Deployment models (FIXED_T=16)
│   ├── stage1/
│   ├── stage2/
│   └── feat_cache/
└── inference_results/                # Deployment outputs
    ├── json/                         # JSON timelines
    │   └── video_name_timeline.json
    └── videos/                       # Annotated videos
        └── video_name_annotated.mp4
```

## 📈 Training Results

### Stage-1 (Coarse Actions)
- **Classes:** 14
- **Training Accuracy:** 100%
- **Final Loss:** ~0.0001

### Stage-2 (Fine-Grained Actions)
- **Families Trained:** 8 out of 14
- **Training Accuracy:** 100% (all families)
- **Final Loss:** 0.0001 - 0.0008

**Skipped Families:** 6 (single fine-grained class, no disambiguation needed)

## 🎯 Deployment Features (Industry Ready)

### Dual Output Format
1. **JSON Timelines**: Machine-readable format for system integration
2. **Annotated Videos**: Human-readable format with color-coded overlays

### Video Overlay System
- **Location**: Top-left corner of video frames
- **Content**: 
  - Coarse action classification
  - Fine-grained action prediction
  - Value category (VA/RNVA/NVA)
- **Color Coding**:
  - **VA (Value-Added)**: Green text
  - **RNVA (Required Non-Value-Added)**: Orange text
  - **NVA (Non-Value-Added)**: Red text

### Optimized Configuration
- **FIXED_T**: 16 frames (deployment) vs 64 frames (research)
- **Better Interpretability**: Shorter temporal sequences
- **Single Prediction**: One result per video (aggregated via majority voting)
- **Industry Format**: Filename-based labeling (`action_va_001.mp4`)

## 🎯 Inference Output Format

### JSON Format (Deployment)
```json
[
  {
    "start": 0.0,
    "end": 8.97,
    "duration": 8.97,
    "coarse_action": "apply",
    "fine_action": "apply_loctite_to_nipple",
    "value_category": "RNVA",
    "num_frames": 5
  }
]
```

### Research Format (Legacy)
```json
[
  {
    "start": 0.0,
    "end": 32.01,
    "duration": 32.01,
    "coarse_action": "mount",
    "fine_action": "mount_side_panel_assembly_to_base_frame_va",
    "num_frames": 16
  },
  {
    "start": 34.14,
    "end": 38.41,
    "duration": 4.27,
    "coarse_action": "tight",
    "fine_action": "tight_bolts_with_air_gun_va",
    "num_frames": 3
  }
]
```

### Text Format
```
================================================================================
ACTION TIMELINE
================================================================================

Segment 1:
  Time: 0.00s - 32.01s (32.01s)
  Coarse Action: mount
  Fine Action: mount_side_panel_assembly_to_base_frame_va
  Frames: 16

Segment 2:
  Time: 34.14s - 38.41s (4.27s)
  Coarse Action: tight
  Fine Action: tight_bolts_with_air_gun_va
  Frames: 3
```

## ⚙️ Configuration

### Training Parameters

**Research System (`train_two_stage_model.py`):**
```python
FIXED_T = 64            # Sequence length (research)
TARGET_FPS = 8          # Temporal sampling rate
CLIP_LEN = 16           # Frames per chunk
BATCH_SIZE = 4          # Batch size
EPOCHS_1 = 40           # Stage-1 epochs
EPOCHS_2 = 50           # Stage-2 epochs
LR = 1e-3               # Learning rate
WEIGHT_DECAY = 1e-4     # Weight decay
```

**Deployment System (`train_deployment_model.py`):**
```python
FIXED_T = 16            # Sequence length (deployment)
TARGET_FPS = 8          # Temporal sampling rate
CLIP_LEN = 16           # Frames per chunk
BATCH_SIZE = 4          # Batch size
EPOCHS_1 = 40           # Stage-1 epochs
EPOCHS_2 = 50           # Stage-2 epochs
LR = 1e-3               # Learning rate
WEIGHT_DECAY = 1e-4     # Weight decay
```

### Inference Parameters

**Research System (`inference_two_stage.py`):**
```python
SMOOTH_K_STAGE1 = 9     # Smoothing window (odd)
MIN_SEG_DUR_S = 1.0     # Minimum segment duration (seconds)
```

**Deployment System (`inference_deployment.py`):**
```python
SMOOTH_K_STAGE1 = 9     # Smoothing window (odd)
MIN_SEG_DUR_S = 1.0     # Minimum segment duration (seconds)
# Video overlay colors:
# VA = Green, RNVA = Orange, NVA = Red
```

## 🔧 Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 4050)
- **VRAM:** 4-6 GB
- **RAM:** 8 GB+
- **Storage:** ~2 GB for models and cache

## 📊 Performance

- **Training Time:** ~12 minutes (full pipeline)
- **Inference Speed:** ~1-2 seconds per video (depends on length)
- **Feature Caching:** Enabled (speeds up repeated runs)
- **Mixed Precision:** FP16 (reduces memory, increases speed)

## 🎓 Reference

This implementation strictly follows:
**`Temporal_Model_Generic_TwoStage_FirstWord_Timeline_Overlay.ipynb`**

All architectural decisions, hyperparameters, and labeling strategies are preserved from the reference notebook.

## 📝 Key Features

✅ **Two-Stage Architecture:** Hierarchical coarse-to-fine prediction  
✅ **FirstWord Strategy:** Stable coarse action labeling  
✅ **Temporal Smoothing:** Reduces jitter in predictions  
✅ **Feature Caching:** Efficient repeated training  
✅ **Mixed Precision:** FP16 for faster training  
✅ **Value Category Preservation:** VA/RNVA/NVA semantics maintained  
✅ **Production-Ready:** Inference pipeline for unseen videos  

## 🚨 Important Notes

### Training Mode
- **No validation split:** Full dataset used for training
- **No evaluation:** Representation learning focus
- **No accuracy reporting:** Training metrics only

### Inference Mode
- **Temporal smoothing:** Applied to Stage-1 predictions
- **Segment merging:** Short segments merged with neighbors
- **Majority voting:** Per-segment predictions aggregated

## 🔍 Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` in training script
- Reduce `FIXED_T` (sequence length)
- Enable mixed precision (already enabled by default)

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`
- Feature caching should speed up repeated runs

### Inference Errors
- Ensure models are trained: `outputs_two_stage/stage1/best.pt` exists
- Check test video format: MP4, AVI, MOV supported
- Verify video codec compatibility with OpenCV

## 📚 Citation

If you use this code, please reference the original methodology from the reference notebook.

## 📧 Support

For issues or questions, please check:
1. `TRAINING_SUMMARY.md` for detailed training information
2. Code comments in `train_two_stage_model.py` and `inference_two_stage.py`
3. Reference notebook for conceptual understanding

---

**Status:** ✅ Production-Ready (Both Research & Deployment Systems)  
**Last Updated:** January 16, 2026  
**Hardware Tested:** NVIDIA RTX 4050 (Laptop)  
**Industry Demo:** Ready with dual output format (JSON + Video overlays)
