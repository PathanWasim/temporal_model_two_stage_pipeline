#!/usr/bin/env python3
"""
Two-Stage Temporal Action Recognition - Setup Script
Quick setup and validation for the temporal action recognition system.
"""

import os
import sys
import subprocess
import torch

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'cv2', 'numpy', 'pandas', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} missing")
    
    return len(missing) == 0

def create_directories():
    """Create necessary directories"""
    dirs = [
        'dataset',
        'test', 
        'outputs_deployment',
        'outputs_two_stage',
        'inference_results'
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Directory created: {dir_name}/")

def main():
    print("=" * 60)
    print("TWO-STAGE TEMPORAL ACTION RECOGNITION - SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check PyTorch
    if not check_pytorch():
        print("\n💡 Install PyTorch:")
        print("   pip install torch torchvision")
        print("   # For CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    
    # Check other dependencies
    if not check_dependencies():
        print("\n💡 Install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\n🚀 Next steps:")
    print("1. Add your video dataset to dataset/ folder")
    print("2. Add test videos to test/ folder") 
    print("3. Run training: python train_deployment_model.py")
    print("4. Run inference: python inference_deployment.py")
    print("\n📖 See README.md for detailed instructions")

if __name__ == "__main__":
    main()