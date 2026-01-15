#!/usr/bin/env python3
"""
Two-Stage Temporal Action Recognition - Example Usage
Demonstrates how to use the system for training and inference.
"""

import os
import sys
from pathlib import Path

def check_setup():
    """Check if the system is properly set up"""
    print("🔍 Checking system setup...")
    
    # Check if required files exist
    required_files = [
        'train_deployment_model.py',
        'inference_deployment.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Check directories
    if not os.path.exists('dataset'):
        print("⚠️  dataset/ directory not found - create it and add your training videos")
    
    if not os.path.exists('test'):
        print("⚠️  test/ directory not found - create it and add your test videos")
    
    print("✅ System setup looks good!")
    return True

def example_training():
    """Example training workflow"""
    print("\n" + "="*50)
    print("📚 TRAINING EXAMPLE")
    print("="*50)
    
    print("1. Prepare your dataset in dataset/ folder")
    print("   Format: action_name_va_001.mp4")
    print("   Example: hand_tight_nipple_to_pipe_va_001.mp4")
    
    print("\n2. Run training:")
    print("   python train_deployment_model.py")
    
    print("\n3. Training will create:")
    print("   - outputs_deployment/stage1/best.pt")
    print("   - outputs_deployment/stage2/{family}/best.pt")
    print("   - Feature cache for faster re-training")

def example_inference():
    """Example inference workflow"""
    print("\n" + "="*50)
    print("🎯 INFERENCE EXAMPLE") 
    print("="*50)
    
    print("1. Add test videos to test/ folder")
    print("   Format: action_name_va_002.mp4")
    
    print("\n2. Run inference:")
    print("   python inference_deployment.py")
    
    print("\n3. Outputs will be generated:")
    print("   - inference_results/json/video_name_timeline.json")
    print("   - inference_results/videos/video_name_annotated.mp4")
    
    print("\n4. JSON format:")
    print('   [{"start": 0.0, "end": 10.5, "coarse_action": "apply",')
    print('     "fine_action": "apply_loctite_to_nipple", "value_category": "RNVA"}]')
    
    print("\n5. Video overlay shows:")
    print("   COARSE ACTION: apply")
    print("   FINE ACTION  : apply_loctite_to_nipple") 
    print("   VALUE TYPE   : RNVA (in orange color)")

def example_dataset_structure():
    """Show example dataset structure"""
    print("\n" + "="*50)
    print("📁 DATASET STRUCTURE EXAMPLE")
    print("="*50)
    
    example_files = [
        "apply_loctite_to_nipple_rnva_001.mp4",
        "hand_tight_nipple_to_pipe_va_001.mp4", 
        "mount_o_ring_to_pipe_va_001.mp4",
        "get_side_panel_rnva_001.mp4",
        "tight_bolts_with_air_gun_va_001.mp4"
    ]
    
    print("dataset/")
    for file in example_files:
        print(f"├── {file}")
    print("└── ...")
    
    print("\ntest/")
    for file in example_files:
        test_file = file.replace("_001.mp4", "_002.mp4")
        print(f"├── {test_file}")
    print("└── ...")

def main():
    print("🎬 Two-Stage Temporal Action Recognition - Example Usage")
    print("="*60)
    
    if not check_setup():
        print("\n💡 Run 'python setup.py' to check your installation")
        return
    
    example_dataset_structure()
    example_training()
    example_inference()
    
    print("\n" + "="*60)
    print("📖 For detailed information, see:")
    print("   - README.md - Complete documentation")
    print("   - PROJECT_STRUCTURE.md - File organization")
    print("   - TRAINING_SUMMARY.md - Training results")
    print("="*60)

if __name__ == "__main__":
    main()