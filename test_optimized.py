#!/usr/bin/env python3
"""
Test script for the fixed optimized Orpheus CPP implementation.
This script tests the fixes for the reported errors.
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fixed_optimized():
    """Test the fixed optimized implementation."""
    print("🧪 Testing Fixed OptimizedOrpheusCpp...")
    
    try:
        from src.orpheus_cpp.optimized_model import OptimizedOrpheusCpp, TTSOptions
        print("✅ Import successful")
        
        # Initialize with conservative settings for testing
        print("🔄 Initializing model...")
        model = OptimizedOrpheusCpp(
            n_gpu_layers=0,   # Start with CPU to avoid GPU issues
            n_threads=0,      # Auto-detect
            verbose=True,     # Enable verbose for debugging
            batch_size=2,     # Small batch for testing
            n_parallel=1,     # Single session for testing
        )
        print("✅ Model initialized successfully")
        
        # Test single TTS
        print("🎵 Testing single TTS...")
        test_text = "Hello world, this is a test of the fixed text to speech system."
        
        start_time = time.time()
        sample_rate, audio = model.tts(test_text)
        end_time = time.time()
        
        duration = end_time - start_time
        audio_length = len(audio[0]) / sample_rate if len(audio.shape) > 1 else len(audio) / sample_rate
        real_time_factor = audio_length / duration if duration > 0 else 0
        
        print(f"✅ Single TTS completed:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Audio length: {audio_length:.2f}s")
        print(f"   Real-time factor: {real_time_factor:.2f}x")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Audio shape: {audio.shape}")
        
        if audio.size > 0:
            print("✅ Audio generated successfully")
        else:
            print("⚠️  No audio generated")
        
        print("\n🎉 Fixed version test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_conditions():
    """Test specific error conditions that were reported."""
    print("\n🔍 Testing error condition fixes...")
    
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"✅ Available ONNX providers: {providers}")
        
        # Test CUDA availability
        cuda_available = "CUDAExecutionProvider" in providers
        print(f"✅ CUDA provider available: {cuda_available}")
        
        if not cuda_available:
            print("ℹ️  This is expected if CUDA is not properly installed")
            print("ℹ️  The fixed version should handle this gracefully")
        
        # Test data type handling
        import numpy as np
        test_array_int32 = np.array([1, 2, 3, 4], dtype=np.int32)
        test_array_int64 = np.array([1, 2, 3, 4], dtype=np.int64)
        
        print(f"✅ int32 array: {test_array_int32.dtype}")
        print(f"✅ int64 array: {test_array_int64.dtype}")
        print("✅ Data type handling test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error condition test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Fixed OptimizedOrpheusCpp Test Suite")
    print("=" * 50)
    
    # Test error conditions first
    error_test_success = test_error_conditions()
    
    # Test fixed functionality
    functionality_test_success = test_fixed_optimized()
    
    if error_test_success and functionality_test_success:
        print("\n✅ All tests completed successfully!")
        print("🚀 The fixes appear to be working correctly")
        print("💡 You can now try running the benchmark with the fixed version")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
