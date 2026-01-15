"""
Two-Stage Temporal Action Recognition - Inference Script
Processes test videos and generates action timelines
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from train_two_stage_model import (
    build_r3d18_feature_extractor,
    extract_features_steps_r3d18,
    TemporalClassifier,
    DEVICE, TARGET_FPS, CLIP_LEN, FEATURE_DIM
)

# Paths
STAGE1_DIR = "outputs_two_stage/stage1"
STAGE2_DIR = "outputs_two_stage/stage2"
TEST_DIR = "test"

# Smoothing parameters (from reference notebook)
SMOOTH_K_STAGE1 = 9  # Odd window for stage-1 smoothing
MIN_SEG_DUR_S = 1.0  # Merge segments shorter than this

def load_label_map(path: str) -> Dict:
    """Load label map from JSON"""
    with open(path, "r") as f:
        return json.load(f)

def load_stage2_registry() -> Dict:
    """Load Stage-2 model registry"""
    registry_path = os.path.join(STAGE2_DIR, "stage2_registry.json")
    with open(registry_path, "r") as f:
        return json.load(f)

def smooth_predictions(preds: np.ndarray, k: int = 9) -> np.ndarray:
    """
    Smooth predictions using majority voting in sliding window
    k: window size (should be odd)
    """
    if len(preds) == 0:
        return preds
    if k <= 1:
        return preds
    
    k = k if k % 2 == 1 else k + 1  # Ensure odd
    half = k // 2
    smoothed = np.copy(preds)
    
    for i in range(len(preds)):
        start = max(0, i - half)
        end = min(len(preds), i + half + 1)
        window = preds[start:end]
        vals, counts = np.unique(window, return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    
    return smoothed

def merge_short_segments(preds: np.ndarray, times: np.ndarray, min_dur: float) -> np.ndarray:
    """
    Merge segments shorter than min_dur seconds
    """
    if len(preds) == 0:
        return preds
    
    merged = np.copy(preds)
    i = 0
    while i < len(merged):
        # Find segment boundaries
        current_label = merged[i]
        j = i
        while j < len(merged) and merged[j] == current_label:
            j += 1
        
        # Check segment duration
        seg_start = times[i]
        seg_end = times[j-1] if j < len(times) else times[-1]
        duration = seg_end - seg_start
        
        if duration < min_dur and i > 0:
            # Merge with previous segment
            merged[i:j] = merged[i-1]
        
        i = j
    
    return merged

@torch.no_grad()
def predict_stage1(feats: torch.Tensor, model: nn.Module, label_map: Dict) -> np.ndarray:
    """
    Stage-1 prediction: coarse action classification
    feats: [T, 512]
    Returns: [T] array of predicted class indices
    """
    model.eval()
    feats = feats.unsqueeze(0).to(DEVICE)  # [1, T, 512]
    logits = model(feats)  # [1, T, num_classes]
    preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # [T]
    return preds

@torch.no_grad()
def predict_stage2(feats: torch.Tensor, model: nn.Module, label_map: Dict) -> int:
    """
    Stage-2 prediction: fine-grained action for a segment
    feats: [T, 512]
    Returns: single predicted class index (majority vote)
    """
    model.eval()
    feats = feats.unsqueeze(0).to(DEVICE)  # [1, T, 512]
    logits = model(feats)  # [1, T, num_classes]
    preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # [T]
    
    # Majority vote
    vals, counts = np.unique(preds, return_counts=True)
    return int(vals[np.argmax(counts)])

def inference_two_stage(video_path: str, backbone, r3d_mean, r3d_std,
                        stage1_model, stage1_l2i, stage1_i2l,
                        stage2_registry, stage2_models, stage2_label_maps) -> List[Dict]:
    """
    Two-stage inference on a video
    Returns: list of segments with predictions
    """
    print(f"\n🎬 Processing: {Path(video_path).name}")
    
    # Extract features
    print("   Extracting R3D-18 features...")
    feats, times, src_fps = extract_features_steps_r3d18(
        video_path, TARGET_FPS, CLIP_LEN, backbone, r3d_mean, r3d_std
    )
    print(f"   Features: {feats.shape} | Times: {times.shape} | FPS: {src_fps:.1f}")
    
    # Stage-1: Coarse prediction
    print("   Stage-1: Predicting coarse actions...")
    stage1_preds = predict_stage1(feats, stage1_model, stage1_l2i)
    
    # Smooth Stage-1 predictions
    stage1_preds_smooth = smooth_predictions(stage1_preds, k=SMOOTH_K_STAGE1)
    stage1_preds_smooth = merge_short_segments(stage1_preds_smooth, times.numpy(), MIN_SEG_DUR_S)
    
    # Identify segments
    segments = []
    i = 0
    while i < len(stage1_preds_smooth):
        coarse_idx = stage1_preds_smooth[i]
        coarse_label = stage1_i2l[coarse_idx]
        
        # Find segment end
        j = i
        while j < len(stage1_preds_smooth) and stage1_preds_smooth[j] == coarse_idx:
            j += 1
        
        seg_feats = feats[i:j]
        seg_start = float(times[i])
        seg_end = float(times[j-1]) if j < len(times) else float(times[-1])
        
        # Stage-2: Fine-grained prediction
        fine_label = coarse_label  # Default
        if coarse_label in stage2_registry and stage2_registry[coarse_label].get("trained"):
            if coarse_label in stage2_models:
                stage2_model = stage2_models[coarse_label]
                stage2_l2i = stage2_label_maps[coarse_label]
                stage2_i2l = {v: k for k, v in stage2_l2i.items()}
                
                fine_idx = predict_stage2(seg_feats, stage2_model, stage2_l2i)
                fine_label = stage2_i2l[fine_idx]
        
        segments.append({
            "start": seg_start,
            "end": seg_end,
            "duration": seg_end - seg_start,
            "coarse_action": coarse_label,
            "fine_action": fine_label,
            "num_frames": j - i
        })
        
        i = j
    
    print(f"   ✅ Detected {len(segments)} action segments")
    return segments

def visualize_timeline(segments: List[Dict], output_path: str):
    """Create a simple text-based timeline visualization"""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ACTION TIMELINE\n")
        f.write("=" * 80 + "\n\n")
        
        for i, seg in enumerate(segments, 1):
            f.write(f"Segment {i}:\n")
            f.write(f"  Time: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)\n")
            f.write(f"  Coarse Action: {seg['coarse_action']}\n")
            f.write(f"  Fine Action: {seg['fine_action']}\n")
            f.write(f"  Frames: {seg['num_frames']}\n")
            f.write("\n")

def main():
    print("=" * 80)
    print("TWO-STAGE TEMPORAL ACTION RECOGNITION - INFERENCE")
    print("=" * 80)
    
    # Load R3D-18 backbone
    print("\n🧠 Loading R3D-18 feature extractor...")
    backbone_r3d, r3d_mean, r3d_std = build_r3d18_feature_extractor(DEVICE)
    print("   ✅ R3D-18 ready")
    
    # Load Stage-1 model
    print("\n📦 Loading Stage-1 model...")
    stage1_l2i = load_label_map(os.path.join(STAGE1_DIR, "label_map.json"))
    stage1_i2l = {v: k for k, v in stage1_l2i.items()}
    stage1_model = TemporalClassifier(in_dim=FEATURE_DIM, num_classes=len(stage1_l2i), hidden=256).to(DEVICE)
    stage1_model.load_state_dict(torch.load(os.path.join(STAGE1_DIR, "best.pt"), map_location=DEVICE))
    stage1_model.eval()
    print(f"   ✅ Stage-1 loaded | Classes: {len(stage1_l2i)}")
    
    # Load Stage-2 models
    print("\n📦 Loading Stage-2 models...")
    stage2_registry = load_stage2_registry()
    stage2_models = {}
    stage2_label_maps = {}
    
    for family, info in stage2_registry.items():
        if info.get("trained"):
            family_dir = os.path.join(STAGE2_DIR, family)
            label_map = load_label_map(os.path.join(family_dir, "label_map.json"))
            model = TemporalClassifier(in_dim=FEATURE_DIM, num_classes=len(label_map), hidden=256).to(DEVICE)
            model.load_state_dict(torch.load(os.path.join(family_dir, "best.pt"), map_location=DEVICE))
            model.eval()
            stage2_models[family] = model
            stage2_label_maps[family] = label_map
            print(f"   ✅ {family}: {len(label_map)} classes")
    
    print(f"\n   Total Stage-2 models loaded: {len(stage2_models)}")
    
    # Find test videos
    print(f"\n🔍 Scanning test directory: {TEST_DIR}")
    test_videos = []
    if os.path.exists(TEST_DIR):
        for ext in ["*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI", "*.MOV"]:
            import glob
            test_videos.extend(glob.glob(os.path.join(TEST_DIR, ext)))
    
    if len(test_videos) == 0:
        print("   ⚠️  No test videos found")
        return
    
    print(f"   Found {len(test_videos)} test videos")
    
    # Process each test video
    os.makedirs("inference_results", exist_ok=True)
    
    for video_path in test_videos:
        try:
            segments = inference_two_stage(
                video_path, backbone_r3d, r3d_mean, r3d_std,
                stage1_model, stage1_l2i, stage1_i2l,
                stage2_registry, stage2_models, stage2_label_maps
            )
            
            # Save results
            video_name = Path(video_path).stem
            json_path = f"inference_results/{video_name}_timeline.json"
            txt_path = f"inference_results/{video_name}_timeline.txt"
            
            with open(json_path, "w") as f:
                json.dump(segments, f, indent=2)
            
            visualize_timeline(segments, txt_path)
            
            print(f"   💾 Results saved:")
            print(f"      - {json_path}")
            print(f"      - {txt_path}")
            
        except Exception as e:
            print(f"   ❌ Error processing {Path(video_path).name}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ INFERENCE COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: inference_results/")

if __name__ == "__main__":
    main()
