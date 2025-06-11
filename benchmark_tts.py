#!/usr/bin/env python3
"""
Fixed TTS Benchmarking Script

This script benchmarks the fixed optimized version of Orpheus CPP
with proper error handling and fallbacks.
"""

import asyncio
import json
import time
import statistics
import psutil
import threading
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import os

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import GPUtil
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False
    print("GPUtil not available. Install with: pip install GPUtil")

try:
    from src.orpheus_cpp.model import OrpheusCpp
    from src.orpheus_cpp.optimized_model import OptimizedOrpheusCpp, TTSOptions
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
    print("pip install cupy-cuda12x")
    sys.exit(1)


# Sample texts for benchmarking (reduced set for testing)
BENCHMARK_TEXTS = [
    "The rapid advancement of artificial intelligence has transformed numerous industries, from healthcare to finance, creating unprecedented opportunities for innovation and efficiency.",
    
    "Climate change represents one of the most pressing challenges of our time, requiring immediate and coordinated global action to reduce greenhouse gas emissions.",
    
    "The digital revolution has democratized access to information and education, enabling people from all backgrounds to learn new skills and connect with others across the globe.",
    
    "Space exploration continues to capture human imagination and drive technological innovation, with private companies now joining government agencies in the quest to explore Mars.",
    
    "The human brain remains one of the most complex and fascinating structures in the known universe, containing approximately 86 billion neurons that form intricate networks.",
    
    "Renewable energy technologies have reached a tipping point where they are now cost-competitive with fossil fuels in many markets, accelerating the global transition to clean energy.",
    
    "The rise of remote work has fundamentally altered traditional employment patterns, offering greater flexibility for workers while challenging organizations to maintain productivity.",
    
    "Biotechnology advances are revolutionizing medicine through personalized treatments, gene therapy, and precision diagnostics that target diseases at the molecular level.",
    
    "The Internet of Things is creating a interconnected world where everyday objects can communicate and share data, enabling smart cities and automated homes.",
    
    "Quantum computing represents a paradigm shift in computational power, with the potential to solve complex problems that are intractable for classical computers."
]


class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
        def monitor():
            while self.monitoring:
                # CPU and Memory
                self.cpu_usage.append(psutil.cpu_percent(interval=1))
                self.memory_usage.append(psutil.virtual_memory().percent)
                
                # GPU monitoring if available
                if GPU_MONITORING:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_load = [gpu.load * 100 for gpu in gpus]
                            gpu_mem = [gpu.memoryUtil * 100 for gpu in gpus]
                            self.gpu_usage.append(gpu_load)
                            self.gpu_memory.append(gpu_mem)
                    except Exception as e:
                        print(f"GPU monitoring error: {e}")
                
                time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        
        stats = {
            'cpu': {
                'avg': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max': max(self.cpu_usage) if self.cpu_usage else 0,
                'min': min(self.cpu_usage) if self.cpu_usage else 0,
            },
            'memory': {
                'avg': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'max': max(self.memory_usage) if self.memory_usage else 0,
                'min': min(self.memory_usage) if self.memory_usage else 0,
            }
        }
        
        if self.gpu_usage and GPU_MONITORING:
            # Average across all GPUs and time
            all_gpu_loads = [load for loads in self.gpu_usage for load in loads]
            all_gpu_memory = [mem for mems in self.gpu_memory for mem in mems]
            
            stats['gpu'] = {
                'avg_load': statistics.mean(all_gpu_loads) if all_gpu_loads else 0,
                'max_load': max(all_gpu_loads) if all_gpu_loads else 0,
                'avg_memory': statistics.mean(all_gpu_memory) if all_gpu_memory else 0,
                'max_memory': max(all_gpu_memory) if all_gpu_memory else 0,
            }
        
        return stats


class TTSBenchmark:
    """Fixed TTS benchmarking suite."""
    
    def __init__(self):
        self.results = {
            'original': {},
            'optimized_fixed': {},
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
        }
        
        if GPU_MONITORING:
            try:
                gpus = GPUtil.getGPUs()
                info['gpus'] = [
                    {
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'driver': gpu.driver
                    }
                    for gpu in gpus
                ]
            except Exception as e:
                info['gpu_error'] = str(e)
        
        # Check ONNX providers
        try:
            import onnxruntime
            info['onnx_providers'] = onnxruntime.get_available_providers()
        except Exception as e:
            info['onnx_error'] = str(e)
        
        return info
    
    def benchmark_original(self, texts: List[str], num_runs: int = 1) -> Dict[str, Any]:
        """Benchmark the original OrpheusCpp implementation."""
        print(f"\n🔥 Benchmarking Original OrpheusCpp ({len(texts)} texts, {num_runs} runs)")
        
        try:
            # Initialize with conservative settings
            model = OrpheusCpp(n_gpu_layers=0, verbose=False)  # Start with CPU
            
            monitor = SystemMonitor()
            monitor.start_monitoring()
            
            times = []
            audio_lengths = []
            errors = 0
            
            start_time = time.time()
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")
                
                for i, text in enumerate(texts):
                    try:
                        text_start = time.time()
                        sample_rate, audio = model.tts(text)
                        text_end = time.time()
                        
                        duration = text_end - text_start
                        audio_length = len(audio[0]) / sample_rate if len(audio.shape) > 1 else len(audio) / sample_rate
                        
                        times.append(duration)
                        audio_lengths.append(audio_length)
                        
                        if (i + 1) % 5 == 0:
                            print(f"    Completed {i + 1}/{len(texts)} texts")
                            
                    except Exception as e:
                        print(f"    Error processing text {i}: {e}")
                        errors += 1
            
            total_time = time.time() - start_time
            system_stats = monitor.stop_monitoring()
            
            results = {
                'total_time': total_time,
                'avg_time_per_text': statistics.mean(times) if times else 0,
                'median_time_per_text': statistics.median(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'total_audio_length': sum(audio_lengths),
                'avg_audio_length': statistics.mean(audio_lengths) if audio_lengths else 0,
                'texts_per_second': len(texts) * num_runs / total_time if total_time > 0 else 0,
                'real_time_factor': sum(audio_lengths) / sum(times) if sum(times) > 0 else 0,
                'errors': errors,
                'success_rate': (len(texts) * num_runs - errors) / (len(texts) * num_runs) * 100,
                'system_stats': system_stats
            }
            
            print(f"  ✅ Original benchmark completed in {total_time:.2f}s")
            print(f"  📊 Average time per text: {results['avg_time_per_text']:.2f}s")
            print(f"  🎵 Real-time factor: {results['real_time_factor']:.2f}x")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Original benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_optimized_fixed(self, texts: List[str], num_runs: int = 1) -> Dict[str, Any]:
        """Benchmark the fixed optimized OrpheusCpp implementation."""
        print(f"\n🚀 Benchmarking Fixed OptimizedOrpheusCpp ({len(texts)} texts, {num_runs} runs)")
        
        try:
            # Initialize with adaptive settings based on available hardware
            import onnxruntime
            available_providers = onnxruntime.get_available_providers()
            has_cuda = "CUDAExecutionProvider" in available_providers
            
            if has_cuda:
                print("  🎮 CUDA detected - using GPU acceleration")
                n_gpu_layers = -1  # All layers on GPU
                n_parallel = 2    # Multiple sessions
            else:
                print("  💻 No CUDA - using CPU optimization")
                n_gpu_layers = 0   # CPU only
                n_parallel = 1    # Single session
            
            model = OptimizedOrpheusCpp(
                n_gpu_layers=n_gpu_layers,
                n_threads=0,      # Auto-detect
                verbose=False,
                batch_size=4,     # Moderate batch size
                n_parallel=n_parallel,
            )
            
            monitor = SystemMonitor()
            monitor.start_monitoring()
            
            times = []
            audio_lengths = []
            errors = 0
            
            start_time = time.time()
            
            # Test both individual and batch processing
            batch_times = []
            batch_size = 3
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")
                
                # Process in batches for better efficiency
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    try:
                        batch_start = time.time()
                        results = model.batch_tts(batch_texts)
                        batch_end = time.time()
                        
                        batch_duration = batch_end - batch_start
                        batch_times.append(batch_duration)
                        
                        for j, (sample_rate, audio) in enumerate(results):
                            if audio.size > 0:
                                audio_length = len(audio[0]) / sample_rate if len(audio.shape) > 1 else len(audio) / sample_rate
                                audio_lengths.append(audio_length)
                                times.append(batch_duration / len(batch_texts))  # Approximate per-text time
                            else:
                                errors += 1
                        
                        print(f"    Completed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                        
                    except Exception as e:
                        print(f"    Error processing batch {i//batch_size}: {e}")
                        errors += len(batch_texts)
            
            total_time = time.time() - start_time
            system_stats = monitor.stop_monitoring()
            
            results = {
                'total_time': total_time,
                'avg_time_per_text': statistics.mean(times) if times else 0,
                'median_time_per_text': statistics.median(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'total_audio_length': sum(audio_lengths),
                'avg_audio_length': statistics.mean(audio_lengths) if audio_lengths else 0,
                'texts_per_second': len(texts) * num_runs / total_time if total_time > 0 else 0,
                'real_time_factor': sum(audio_lengths) / sum(times) if sum(times) > 0 else 0,
                'batch_avg_time': statistics.mean(batch_times) if batch_times else 0,
                'errors': errors,
                'success_rate': (len(texts) * num_runs - errors) / (len(texts) * num_runs) * 100,
                'system_stats': system_stats,
                'used_cuda': has_cuda
            }
            
            print(f"  ✅ Fixed optimized benchmark completed in {total_time:.2f}s")
            print(f"  📊 Average time per text: {results['avg_time_per_text']:.2f}s")
            print(f"  🎵 Real-time factor: {results['real_time_factor']:.2f}x")
            print(f"  🚀 Batch processing avg: {results['batch_avg_time']:.2f}s per batch")
            print(f"  🎮 Used CUDA: {has_cuda}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Fixed optimized benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def run_comparison(self, num_texts: int = 10, num_runs: int = 1) -> Dict[str, Any]:
        """Run comparison between original and fixed optimized versions."""
        print(f"\n🏁 Starting Fixed TTS Benchmark Comparison")
        print(f"📝 Testing {num_texts} texts with {num_runs} runs each")
        print(f"💻 System: {self.results['system_info']['cpu_count']} CPU cores, "
              f"{self.results['system_info']['memory_total_gb']:.1f}GB RAM")
        
        if 'gpus' in self.results['system_info']:
            for i, gpu in enumerate(self.results['system_info']['gpus']):
                print(f"🎮 GPU {i}: {gpu['name']} ({gpu['memory_total_mb']}MB)")
        
        if 'onnx_providers' in self.results['system_info']:
            print(f"🔧 ONNX providers: {self.results['system_info']['onnx_providers']}")
        
        # Select subset of texts for testing
        test_texts = BENCHMARK_TEXTS[:num_texts]
        
        # Benchmark original version
        self.results['original'] = self.benchmark_original(test_texts, num_runs)
        
        # Benchmark fixed optimized version
        self.results['optimized_fixed'] = self.benchmark_optimized_fixed(test_texts, num_runs)
        
        # Calculate improvements
        if ('error' not in self.results['original'] and 
            'error' not in self.results['optimized_fixed']):
            
            orig = self.results['original']
            opt = self.results['optimized_fixed']
            
            improvements = {
                'speed_improvement': orig['avg_time_per_text'] / opt['avg_time_per_text'] if opt['avg_time_per_text'] > 0 else 0,
                'throughput_improvement': opt['texts_per_second'] / orig['texts_per_second'] if orig['texts_per_second'] > 0 else 0,
                'total_time_reduction': (orig['total_time'] - opt['total_time']) / orig['total_time'] * 100 if orig['total_time'] > 0 else 0,
                'real_time_factor_improvement': opt['real_time_factor'] / orig['real_time_factor'] if orig['real_time_factor'] > 0 else 0,
            }
            
            self.results['improvements'] = improvements
            
            print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
            print(f"  🚀 Speed improvement: {improvements['speed_improvement']:.2f}x faster")
            print(f"  📊 Throughput improvement: {improvements['throughput_improvement']:.2f}x")
            print(f"  ⏱️  Total time reduction: {improvements['total_time_reduction']:.1f}%")
            print(f"  🎵 Real-time factor improvement: {improvements['real_time_factor_improvement']:.2f}x")
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tts_benchmark_fixed_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print a comprehensive summary of results."""
        print(f"\n" + "="*80)
        print(f"🎯 FIXED TTS BENCHMARK SUMMARY")
        print(f"="*80)
        
        # Handle original version results
        if self.results['original'] and 'error' not in self.results['original']:
            orig = self.results['original']
            print(f"\n📊 ORIGINAL VERSION:")
            print(f"  Total time: {orig['total_time']:.2f}s")
            print(f"  Avg time per text: {orig['avg_time_per_text']:.2f}s")
            print(f"  Texts per second: {orig['texts_per_second']:.2f}")
            print(f"  Real-time factor: {orig['real_time_factor']:.2f}x")
            print(f"  Success rate: {orig['success_rate']:.1f}%")
        elif self.results['original'] and 'error' in self.results['original']:
            print(f"\n📊 ORIGINAL VERSION:")
            print(f"  ❌ Benchmark failed: {self.results['original']['error']}")
        
        # Handle fixed optimized version results
        if self.results['optimized_fixed'] and 'error' not in self.results['optimized_fixed']:
            opt = self.results['optimized_fixed']
            print(f"\n🚀 FIXED OPTIMIZED VERSION:")
            print(f"  Total time: {opt['total_time']:.2f}s")
            print(f"  Avg time per text: {opt['avg_time_per_text']:.2f}s")
            print(f"  Texts per second: {opt['texts_per_second']:.2f}")
            print(f"  Real-time factor: {opt['real_time_factor']:.2f}x")
            print(f"  Success rate: {opt['success_rate']:.1f}%")
            print(f"  Used CUDA: {opt.get('used_cuda', False)}")
        elif self.results['optimized_fixed'] and 'error' in self.results['optimized_fixed']:
            print(f"\n🚀 FIXED OPTIMIZED VERSION:")
            print(f"  ❌ Benchmark failed: {self.results['optimized_fixed']['error']}")
        
        # Handle improvements calculation
        if 'improvements' in self.results:
            imp = self.results['improvements']
            print(f"\n📈 PERFORMANCE GAINS:")
            print(f"  Speed improvement: {imp['speed_improvement']:.2f}x")
            print(f"  Throughput improvement: {imp['throughput_improvement']:.2f}x")
            print(f"  Time reduction: {imp['total_time_reduction']:.1f}%")
            print(f"  Real-time factor improvement: {imp['real_time_factor_improvement']:.2f}x")
        
        print(f"\n" + "="*80)


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Fixed TTS Performance Benchmark")
    parser.add_argument("--texts", type=int, default=10, 
                       help="Number of texts to benchmark (default: 10)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per benchmark (default: 1)")
    parser.add_argument("--original-only", action="store_true",
                       help="Only benchmark original version")
    parser.add_argument("--optimized-only", action="store_true",
                       help="Only benchmark fixed optimized version")
    parser.add_argument("--output", type=str,
                       help="Output filename for results JSON")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.texts > len(BENCHMARK_TEXTS):
        print(f"Warning: Only {len(BENCHMARK_TEXTS)} texts available, using all of them")
        args.texts = len(BENCHMARK_TEXTS)
    
    benchmark = TTSBenchmark()
    
    try:
        if args.original_only:
            test_texts = BENCHMARK_TEXTS[:args.texts]
            benchmark.results['original'] = benchmark.benchmark_original(test_texts, args.runs)
        elif args.optimized_only:
            test_texts = BENCHMARK_TEXTS[:args.texts]
            benchmark.results['optimized_fixed'] = benchmark.benchmark_optimized_fixed(test_texts, args.runs)
        else:
            benchmark.run_comparison(args.texts, args.runs)
        
        benchmark.print_summary()
        filename = benchmark.save_results(args.output)
        
        print(f"\n🎉 Fixed benchmark completed successfully!")
        print(f"📄 Detailed results saved to: {filename}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
