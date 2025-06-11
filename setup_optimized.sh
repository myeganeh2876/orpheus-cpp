#!/bin/bash

# Setup script for optimized Orpheus CPP on dual RTX 6000 Ada
# This script installs all dependencies for maximum GPU utilization

set -e

echo "🚀 Setting up Optimized Orpheus CPP for Dual RTX 6000 Ada"
echo "=========================================================="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✅ NVIDIA drivers detected"
nvidia-smi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.10+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_optimized" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_optimized
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_optimized/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-optimized llama-cpp-python
echo "🦙 Installing CUDA-optimized llama-cpp-python..."
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Install other requirements
echo "📋 Installing other requirements..."
pip install -r requirements_optimized.txt

# Install the package in development mode
echo "📦 Installing orpheus-cpp in development mode..."
pip install -e .

# Verify installations
echo "🔍 Verifying installations..."

# Check CUDA availability
python3 -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
"

# Check CuPy
python3 -c "
try:
    import cupy as cp
    print(f'CuPy available: True')
    print(f'CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')
except ImportError:
    print('CuPy available: False')
"

# Check ONNX Runtime GPU
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'ONNX Runtime providers: {providers}')
print(f'CUDA provider available: {\"CUDAExecutionProvider\" in providers}')
"

# Check llama-cpp-python CUDA
python3 -c "
try:
    from llama_cpp import Llama
    print('llama-cpp-python imported successfully')
except ImportError as e:
    print(f'llama-cpp-python import error: {e}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source venv_optimized/bin/activate"
echo "2. Run the benchmark: python benchmark_tts.py --texts 10 --optimized-only"
echo "3. For full comparison: python benchmark_tts.py --texts 100"
echo ""
echo "💡 Tips for maximum performance:"
echo "- Ensure both RTX 6000 Ada GPUs are detected"
echo "- Monitor GPU utilization with: watch -n 1 nvidia-smi"
echo "- Use batch processing for better throughput"
echo ""
echo "🔧 Configuration recommendations:"
echo "- Set CUDA_VISIBLE_DEVICES=0,1 to use both GPUs"
echo "- Increase batch size if you have sufficient VRAM"
echo "- Use n_parallel=8 or higher for maximum throughput"
