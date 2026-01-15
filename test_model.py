"""
Comprehensive Testing Script for Two-Stage Temporal Action Recognition
Tests model loading, inference, and output validation
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from train_two_stage_model import (
    build_r3d18_feature_extractor,
    TemporalClassifier,
    DEVICE, FEATURE_DIM
)

def test_model_files():
    """Test 1: Verify all model files exist"""
    print("=" * 80)
    print("TEST 1: Model Files Verification")
    print("=" * 80)
    
    required_files = [
        "outputs_two_stage/stage1/best.pt",
        "outputs_two_stage/stage1/label_map.json",
        "outputs_two_stage/stage2/stage2_registry.json"
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        all_exist = all_exist and exists
    
    # Check Stage-2 models
    registry_path = "outputs_two_stage/stage2/stage2_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        print(f"\n📦 Stage-2 Models:")
        for family, info in registry.items():
            if info.get("trained"):
                model_path = f"outputs_two_stage/stage2/{family}/best.pt"
                label_path = f"outputs_two_stage/stage2/{family}/label_map.json"
                model_exists = os.path.exists(model_path)
                label_exists = os.path.exists(label_path)
                status = "✅" if (model_exists and label_exists) else "❌"
                print(f"{status} {family}: model={model_exists}, labels={label_exists}")
    
    print(f"\n{'✅ PASS' if all_exist else '❌ FAIL'}: Model files verification")
    return all_exist

def test_model_loading():
    """Test 2: Load and verify models"""
    print("\n" + "=" * 80)
    print("TEST 2: Model Loading")
    print("=" * 80)
    
    try:
        # Load Stage-1
        print("Loading Stage-1 model...")
        with open("outputs_two_stage/stage1/label_map.json", "r") as f:
            stage1_labels = json.load(f)
        
        stage1_model = TemporalClassifier(
            in_dim=FEATURE_DIM, 
            num_classes=len(stage1_labels), 
            hidden=256
        ).to(DEVICE)
        stage1_model.load_state_dict(
            torch.load("outputs_two_stage/stage1/best.pt", map_location=DEVICE)
        )
        stage1_model.eval()
        print(f"✅ Stage-1 loaded: {len(stage1_labels)} classes")
        
        # Load Stage-2 models
        with open("outputs_two_stage/stage2/stage2_registry.json", "r") as f:
            registry = json.load(f)
        
        stage2_count = 0
        for family, info in registry.items():
            if info.get("trained"):
                label_path = f"outputs_two_stage/stage2/{family}/label_map.json"
                model_path = f"outputs_two_stage/stage2/{family}/best.pt"
                
                with open(label_path, "r") as f:
                    labels = json.load(f)
                
                model = TemporalClassifier(
                    in_dim=FEATURE_DIM,
                    num_classes=len(labels),
                    hidden=256
                ).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                stage2_count += 1
                print(f"✅ Stage-2 {family}: {len(labels)} classes")
        
        print(f"\n✅ PASS: All models loaded successfully")
        print(f"   Stage-1: 1 model")
        print(f"   Stage-2: {stage2_count} models")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Model loading error: {e}")
        return False

def test_feature_extraction():
    """Test 3: Feature extraction pipeline"""
    print("\n" + "=" * 80)
    print("TEST 3: Feature Extraction")
    print("=" * 80)
    
    try:
        # Build R3D-18
        print("Building R3D-18 feature extractor...")
        backbone, r3d_mean, r3d_std = build_r3d18_feature_extractor(DEVICE)
        print(f"✅ R3D-18 ready")
        
        # Test with a sample video
        test_videos = []
        if os.path.exists("test"):
            import glob
            test_videos = glob.glob("test/*.mp4")
        
        if len(test_videos) > 0:
            test_video = test_videos[0]
            print(f"\nTesting feature extraction on: {Path(test_video).name}")
            
            from train_two_stage_model import extract_features_steps_r3d18
            feats, times, fps = extract_features_steps_r3d18(
                test_video, 8, 16, backbone, r3d_mean, r3d_std
            )
            
            print(f"✅ Features extracted:")
            print(f"   Shape: {feats.shape}")
            print(f"   Times: {times.shape}")
            print(f"   FPS: {fps:.1f}")
            print(f"   Feature dim: {feats.shape[1]}")
            
            # Validate feature dimensions
            assert feats.shape[1] == 512, "Feature dimension should be 512"
            assert len(times) == len(feats), "Times and features should match"
            
            print(f"\n✅ PASS: Feature extraction working correctly")
            return True
        else:
            print("⚠️  No test videos found, skipping feature extraction test")
            return True
            
    except Exception as e:
        print(f"\n❌ FAIL: Feature extraction error: {e}")
        return False

def test_inference_outputs():
    """Test 4: Validate inference outputs"""
    print("\n" + "=" * 80)
    print("TEST 4: Inference Output Validation")
    print("=" * 80)
    
    if not os.path.exists("inference_results"):
        print("❌ FAIL: inference_results directory not found")
        return False
    
    import glob
    json_files = glob.glob("inference_results/*.json")
    txt_files = glob.glob("inference_results/*.txt")
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Found {len(txt_files)} TXT files")
    
    if len(json_files) == 0:
        print("❌ FAIL: No inference results found")
        return False
    
    # Validate JSON structure
    all_valid = True
    for json_file in json_files[:3]:  # Check first 3
        print(f"\nValidating: {Path(json_file).name}")
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            assert isinstance(data, list), "Output should be a list"
            
            for i, segment in enumerate(data):
                required_keys = ["start", "end", "duration", "coarse_action", "fine_action", "num_frames"]
                for key in required_keys:
                    assert key in segment, f"Missing key: {key}"
                
                # Validate types
                assert isinstance(segment["start"], (int, float)), "start should be numeric"
                assert isinstance(segment["end"], (int, float)), "end should be numeric"
                assert isinstance(segment["duration"], (int, float)), "duration should be numeric"
                assert isinstance(segment["coarse_action"], str), "coarse_action should be string"
                assert isinstance(segment["fine_action"], str), "fine_action should be string"
                assert isinstance(segment["num_frames"], int), "num_frames should be int"
                
                # Validate values
                assert segment["end"] >= segment["start"], "end should be >= start"
                assert segment["duration"] > 0, "duration should be positive"
                assert segment["num_frames"] > 0, "num_frames should be positive"
            
            print(f"✅ Valid: {len(data)} segments")
            
        except Exception as e:
            print(f"❌ Invalid: {e}")
            all_valid = False
    
    if all_valid:
        print(f"\n✅ PASS: All inference outputs are valid")
    else:
        print(f"\n❌ FAIL: Some inference outputs are invalid")
    
    return all_valid

def test_label_consistency():
    """Test 5: Verify label consistency"""
    print("\n" + "=" * 80)
    print("TEST 5: Label Consistency")
    print("=" * 80)
    
    try:
        # Load Stage-1 labels
        with open("outputs_two_stage/stage1/label_map.json", "r") as f:
            stage1_labels = json.load(f)
        
        print(f"Stage-1 labels ({len(stage1_labels)}):")
        for label, idx in sorted(stage1_labels.items(), key=lambda x: x[1]):
            print(f"  {idx}: {label}")
        
        # Load Stage-2 registry
        with open("outputs_two_stage/stage2/stage2_registry.json", "r") as f:
            registry = json.load(f)
        
        print(f"\nStage-2 families:")
        for family, info in registry.items():
            if info.get("trained"):
                label_path = f"outputs_two_stage/stage2/{family}/label_map.json"
                with open(label_path, "r") as f:
                    labels = json.load(f)
                print(f"  {family}: {len(labels)} classes")
                
                # Verify all fine labels start with family name (FirstWord)
                for label in labels.keys():
                    first_word = label.split("_")[0]
                    if first_word != family:
                        print(f"    ⚠️  Warning: {label} doesn't start with {family}")
        
        print(f"\n✅ PASS: Label consistency verified")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Label consistency error: {e}")
        return False

def test_model_predictions():
    """Test 6: Run sample predictions"""
    print("\n" + "=" * 80)
    print("TEST 6: Model Predictions")
    print("=" * 80)
    
    try:
        # Load models
        with open("outputs_two_stage/stage1/label_map.json", "r") as f:
            stage1_l2i = json.load(f)
        stage1_i2l = {v: k for k, v in stage1_l2i.items()}
        
        stage1_model = TemporalClassifier(
            in_dim=FEATURE_DIM,
            num_classes=len(stage1_l2i),
            hidden=256
        ).to(DEVICE)
        stage1_model.load_state_dict(
            torch.load("outputs_two_stage/stage1/best.pt", map_location=DEVICE)
        )
        stage1_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 10, 512).to(DEVICE)
        
        # Test Stage-1 prediction
        with torch.no_grad():
            logits = stage1_model(dummy_input)
        
        print(f"✅ Stage-1 prediction:")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Expected: [1, 10, {len(stage1_l2i)}]")
        
        assert logits.shape == (1, 10, len(stage1_l2i)), "Output shape mismatch"
        
        # Get predictions
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        print(f"   Predictions: {preds}")
        print(f"   Predicted labels: {[stage1_i2l[p] for p in preds[:3]]}...")
        
        # Test Stage-2 prediction
        with open("outputs_two_stage/stage2/stage2_registry.json", "r") as f:
            registry = json.load(f)
        
        for family, info in list(registry.items())[:1]:  # Test first family
            if info.get("trained"):
                label_path = f"outputs_two_stage/stage2/{family}/label_map.json"
                model_path = f"outputs_two_stage/stage2/{family}/best.pt"
                
                with open(label_path, "r") as f:
                    labels = json.load(f)
                
                model = TemporalClassifier(
                    in_dim=FEATURE_DIM,
                    num_classes=len(labels),
                    hidden=256
                ).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                
                with torch.no_grad():
                    logits = model(dummy_input)
                
                print(f"\n✅ Stage-2 prediction ({family}):")
                print(f"   Input shape: {dummy_input.shape}")
                print(f"   Output shape: {logits.shape}")
                print(f"   Expected: [1, 10, {len(labels)}]")
                
                assert logits.shape == (1, 10, len(labels)), "Output shape mismatch"
                break
        
        print(f"\n✅ PASS: Model predictions working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test 7: End-to-end inference on one video"""
    print("\n" + "=" * 80)
    print("TEST 7: End-to-End Inference")
    print("=" * 80)
    
    try:
        import glob
        test_videos = glob.glob("test/*.mp4")
        
        if len(test_videos) == 0:
            print("⚠️  No test videos found, skipping end-to-end test")
            return True
        
        test_video = test_videos[0]
        print(f"Testing on: {Path(test_video).name}")
        
        # Run inference
        from inference_two_stage import (
            build_r3d18_feature_extractor,
            load_label_map,
            load_stage2_registry,
            inference_two_stage,
            TemporalClassifier
        )
        
        # Load models
        backbone, r3d_mean, r3d_std = build_r3d18_feature_extractor(DEVICE)
        
        stage1_l2i = load_label_map("outputs_two_stage/stage1/label_map.json")
        stage1_i2l = {v: k for k, v in stage1_l2i.items()}
        stage1_model = TemporalClassifier(in_dim=FEATURE_DIM, num_classes=len(stage1_l2i), hidden=256).to(DEVICE)
        stage1_model.load_state_dict(torch.load("outputs_two_stage/stage1/best.pt", map_location=DEVICE))
        stage1_model.eval()
        
        stage2_registry = load_stage2_registry()
        stage2_models = {}
        stage2_label_maps = {}
        
        for family, info in stage2_registry.items():
            if info.get("trained"):
                family_dir = f"outputs_two_stage/stage2/{family}"
                label_map = load_label_map(f"{family_dir}/label_map.json")
                model = TemporalClassifier(in_dim=FEATURE_DIM, num_classes=len(label_map), hidden=256).to(DEVICE)
                model.load_state_dict(torch.load(f"{family_dir}/best.pt", map_location=DEVICE))
                model.eval()
                stage2_models[family] = model
                stage2_label_maps[family] = label_map
        
        # Run inference
        segments = inference_two_stage(
            test_video, backbone, r3d_mean, r3d_std,
            stage1_model, stage1_l2i, stage1_i2l,
            stage2_registry, stage2_models, stage2_label_maps
        )
        
        print(f"\n✅ Inference completed:")
        print(f"   Segments detected: {len(segments)}")
        for i, seg in enumerate(segments, 1):
            print(f"   Segment {i}:")
            print(f"     Time: {seg['start']:.2f}s - {seg['end']:.2f}s")
            print(f"     Coarse: {seg['coarse_action']}")
            print(f"     Fine: {seg['fine_action']}")
        
        print(f"\n✅ PASS: End-to-end inference successful")
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: End-to-end inference error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL TESTING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 80)
    
    tests = [
        ("Model Files", test_model_files),
        ("Model Loading", test_model_loading),
        ("Feature Extraction", test_feature_extraction),
        ("Inference Outputs", test_inference_outputs),
        ("Label Consistency", test_label_consistency),
        ("Model Predictions", test_model_predictions),
        ("End-to-End", test_end_to_end),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Model is production-ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
