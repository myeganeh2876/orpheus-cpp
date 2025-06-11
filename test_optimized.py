#!/usr/bin/env python3
"""
Simple test script for the optimized Orpheus CPP implementation.
This script tests basic functionality before running full benchmarks.
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_optimized_basic():
    """Test basic functionality of the optimized implementation."""
    print("🧪 Testing OptimizedOrpheusCpp basic functionality...")
    
    try:
        from src.orpheus_cpp.optimized_model import OptimizedOrpheusCpp, TTSOptions
        print("✅ Import successful")
        
        # Initialize with conservative settings for testing
        print("🔄 Initializing model...")
        model = OptimizedOrpheusCpp(
            n_gpu_layers=-1,  # All layers on GPU
            n_threads=0,      # Auto-detect
            verbose=False,
            batch_size=4,     # Smaller batch for testing
            n_parallel=2,     # Fewer parallel sessions for testing
        )
        print("✅ Model initialized successfully")
        
        # Test single TTS
        print("🎵 Testing single TTS...")
        test_text = "This is a test of the optimized text to speech system."
        
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
        
        # Test batch TTS
        print("🎵 Testing batch TTS...")
        test_texts = [
            "First test sentence for batch processing.",
            "Second test sentence with different content.",
            "Third and final test sentence."
        ]
        
        start_time = time.time()
        batch_results = model.batch_tts(test_texts)
        end_time = time.time()
        
        batch_duration = end_time - start_time
        total_audio_length = sum(len(audio[0]) / sample_rate if len(audio.shape) > 1 else len(audio) / sample_rate 
                                for sample_rate, audio in batch_results)
        batch_real_time_factor = total_audio_length / batch_duration if batch_duration > 0 else 0
        
        print(f"✅ Batch TTS completed:")
        print(f"   Batch duration: {batch_duration:.2f}s")
        print(f"   Total audio length: {total_audio_length:.2f}s")
        print(f"   Batch real-time factor: {batch_real_time_factor:.2f}x")
        print(f"   Texts processed: {len(batch_results)}")
        
        # Test streaming (just a few chunks)
        print("🎵 Testing streaming TTS...")
        chunk_count = 0
        start_time = time.time()
        
        for sample_rate, audio_chunk in model.stream_tts_sync(test_text):
            chunk_count += 1
            if chunk_count >= 3:  # Just test first few chunks
                break
        
        end_time = time.time()
        streaming_duration = end_time - start_time
        
        print(f"✅ Streaming TTS test completed:")
        print(f"   Chunks received: {chunk_count}")
        print(f"   Time to first chunks: {streaming_duration:.2f}s")
        
        print("\n🎉 All tests passed! OptimizedOrpheusCpp is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_availability():
    """Test GPU availability and CUDA setup."""
    print("\n🔍 Testing GPU and CUDA availability...")
    
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            print("⚠️  CUDA not available - will fall back to CPU")
            
    except ImportError:
        print("⚠️  PyTorch not available")
    
    try:
        import cupy as cp
        print(f"✅ CuPy available: {cp.__version__}")
        print(f"✅ CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    except ImportError:
        print("⚠️  CuPy not available - will use CPU arrays")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        cuda_available = "CUDAExecutionProvider" in providers
        print(f"✅ ONNX CUDA provider: {cuda_available}")
    except ImportError:
        print("❌ ONNX Runtime not available")

def main():
    """Main test function."""
    print("🚀 OptimizedOrpheusCpp Test Suite")
    print("=" * 50)
    
    # Test GPU availability first
    test_gpu_availability()
    
    # Test basic functionality
    success = test_optimized_basic()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🚀 Ready to run full benchmarks with: python benchmark_tts.py")
    else:
        print("\n❌ Tests failed. Please check your setup.")
        print("💡 Try running: ./setup_optimized.sh")
        sys.exit(1)

if __name__ == "__main__":
    main()
