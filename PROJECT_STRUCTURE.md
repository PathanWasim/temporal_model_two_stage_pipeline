# Project Structure

This document describes the clean repository structure for the Two-Stage Temporal Action Recognition system.

## 📁 Repository Files

### Core Implementation
```
├── train_two_stage_model.py          # Research training (FIXED_T=64)
├── train_deployment_model.py         # Deployment training (FIXED_T=16) ⭐
├── inference_two_stage.py            # Research inference
├── inference_deployment.py           # Deployment inference with overlays ⭐
├── test_model.py                     # Testing utilities
└── Temporal_Model_Generic_TwoStage_FirstWord_Timeline_Overlay.ipynb  # Reference notebook
```

### Documentation
```
├── README.md                         # Main documentation
├── TRAINING_SUMMARY.md               # Training results and metrics
├── PROJECT_STRUCTURE.md              # This file
└── .gitignore                        # Git ignore rules
```

## 🚫 Excluded from Repository (.gitignore)

### Data Files (Large)
- `dataset/` - Training video files
- `test/` - Test video files  
- `Dataset_class/` - Additional datasets
- `Onedrive/` - External data files

### Generated Outputs
- `outputs_*/` - Trained model weights and caches
- `inference_results/` - Generated JSON and video outputs
- `feat_cache/` - Cached R3D-18 features

### System Files
- `__pycache__/` - Python cache
- `*.pt`, `*.pth` - PyTorch model files
- `*.mp4`, `*.avi`, `*.mov` - Video files
- `*.xlsx`, `*.xls` - Excel files

## 🎯 Key Files for Users

### For Training
1. **`train_deployment_model.py`** - Main training script (recommended)
2. **`train_two_stage_model.py`** - Research version

### For Inference  
1. **`inference_deployment.py`** - Industry-ready inference with video overlays
2. **`inference_two_stage.py`** - Research version

### For Understanding
1. **`README.md`** - Complete usage guide
2. **`Temporal_Model_Generic_TwoStage_FirstWord_Timeline_Overlay.ipynb`** - Reference methodology
3. **`TRAINING_SUMMARY.md`** - Training results and performance metrics

## 📋 Setup Instructions

1. **Clone the repository**
2. **Install dependencies**: `pip install torch torchvision opencv-python numpy pandas tqdm`
3. **Prepare your dataset** in the expected format (see README.md)
4. **Run training**: `python train_deployment_model.py`
5. **Run inference**: `python inference_deployment.py`

## 🎯 Deployment Focus

The repository is optimized for the **deployment system** (FIXED_T=16) which provides:
- Industry-ready video overlays
- JSON timeline outputs
- Optimized temporal modeling
- Value category classification (VA/RNVA/NVA)

The research system (FIXED_T=64) is included for completeness and comparison.