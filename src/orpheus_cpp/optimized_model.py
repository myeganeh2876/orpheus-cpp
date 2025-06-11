import asyncio
import platform
import sys
import threading
import time
import os
from typing import (
    AsyncGenerator,
    Generator,
    Iterator,
    Literal,
    cast,
)
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available. Install with: pip install cupy-cuda12x")


class TTSOptions(TypedDict):
    max_tokens: NotRequired[int]
    """Maximum number of tokens to generate. Default: 2048"""
    temperature: NotRequired[float]
    """Temperature for top-p sampling. Default: 0.8"""
    top_p: NotRequired[float]
    """Top-p sampling. Default: 0.95"""
    top_k: NotRequired[int]
    """Top-k sampling. Default: 40"""
    min_p: NotRequired[float]
    """Minimum probability for top-p sampling. Default: 0.05"""
    pre_buffer_size: NotRequired[float]
    """Seconds of audio to generate before yielding the first chunk. Smoother audio streaming at the cost of higher time to wait for the first chunk."""
    voice_id: NotRequired[
        Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    ]
    """The voice to use for the TTS. Default: "tara"."""


CUSTOM_TOKEN_PREFIX = "<custom_token_"


class OptimizedOrpheusCpp:
    """Optimized version for dual RTX 6000 Ada setup with maximum GPU utilization."""
    
    lang_to_model = {
        "en": "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
        "es": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF",
        "fr": "freddyaboulton/3b-fr-ft-research_release-Q4_K_M-GGUF",
        "de": "freddyaboulton/3b-de-ft-research_release-Q4_K_M-GGUF",
        "it": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF",
        "hi": "freddyaboulton/3b-hi-ft-research_release-Q4_K_M-GGUF",
        "zh": "freddyaboulton/3b-zh-ft-research_release-Q4_K_M-GGUF",
        "ko": "freddyaboulton/3b-ko-ft-research_release-Q4_K_M-GGUF",
    }

    def __init__(
        self,
        n_gpu_layers: int = -1,  # Use all GPU layers by default
        n_threads: int = 0,  # Auto-detect optimal thread count
        verbose: bool = True,
        lang: Literal["en", "es", "ko", "fr"] = "en",
        gpu_split: list[float] | None = None,  # For multi-GPU setup
        use_mmap: bool = True,  # Memory mapping for faster loading
        use_mlock: bool = True,  # Lock model in memory
        batch_size: int = 8,  # Increased batch size for better GPU utilization
        n_parallel: int = 4,  # Number of parallel inference sessions
    ):
        import importlib.util

        if importlib.util.find_spec("llama_cpp") is None:
            raise ImportError(
                "llama_cpp is not installed. Please install with CUDA support: "
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        # Auto-detect optimal thread count if not specified
        if n_threads == 0:
            n_threads = min(mp.cpu_count(), 32)  # Cap at 32 for optimal performance
            
        # Set up GPU split for dual RTX 6000 Ada (48GB each)
        if gpu_split is None:
            gpu_split = [0.5, 0.5]  # Equal split between two GPUs
            
        repo_id = self.lang_to_model[lang]
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=repo_id.split("/")[-1].lower().replace("-gguf", ".gguf"),
        )
        
        from llama_cpp import Llama

        print(f"Initializing with {n_gpu_layers} GPU layers, {n_threads} threads")
        print(f"GPU split: {gpu_split}")
        print(f"Batch size: {batch_size}")

        # Optimized Llama configuration for dual RTX 6000 Ada
        self._llm = Llama(
            model_path=model_file,
            n_ctx=4096,  # Increased context window
            verbose=verbose,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads,  # Batch processing threads
            batch_size=batch_size,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            tensor_split=gpu_split,  # Split model across GPUs
            main_gpu=0,  # Primary GPU
            flash_attn=True,  # Enable flash attention if available
        )

        # Initialize multiple SNAC sessions for parallel processing
        repo_id = "onnx-community/snac_24khz-ONNX"
        snac_model_file = "decoder_model.onnx"
        snac_model_path = hf_hub_download(
            repo_id, subfolder="onnx", filename=snac_model_file
        )

        # Create multiple ONNX sessions for parallel audio generation
        self._snac_sessions = []
        self._session_lock = threading.Lock()
        
        # CUDA provider options for maximum performance
        cuda_provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,  # 24GB per session
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
        
        for i in range(n_parallel):
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            session_options.inter_op_num_threads = 4
            session_options.intra_op_num_threads = 8
            
            # Alternate between GPUs for load balancing
            gpu_id = i % 2
            cuda_options = cuda_provider_options.copy()
            cuda_options['device_id'] = gpu_id
            
            session = onnxruntime.InferenceSession(
                snac_model_path,
                providers=[
                    ('CUDAExecutionProvider', cuda_options),
                    'CPUExecutionProvider'
                ],
                sess_options=session_options
            )
            self._snac_sessions.append(session)
            
        self._current_session_idx = 0
        self._executor = ThreadPoolExecutor(max_workers=n_parallel * 2)
        
        print(f"Initialized {len(self._snac_sessions)} SNAC sessions across {2} GPUs")
        
        # Pre-warm the models
        self._warmup()

    def _warmup(self):
        """Pre-warm the models for optimal performance."""
        print("Warming up models...")
        warmup_text = "This is a warmup text to initialize the models."
        try:
            # Warmup language model
            warmup_tokens = list(self._token_gen(warmup_text, TTSOptions(max_tokens=50)))
            
            # Warmup SNAC sessions
            dummy_codes = [np.ones((1, 4), dtype=np.int32) for _ in range(3)]
            for session in self._snac_sessions:
                input_names = [x.name for x in session.get_inputs()]
                input_dict = dict(zip(input_names, dummy_codes))
                session.run(None, input_dict)
                
            print("Warmup completed successfully")
        except Exception as e:
            print(f"Warmup warning: {e}")

    def _get_next_session(self):
        """Get the next available SNAC session in round-robin fashion."""
        with self._session_lock:
            session = self._snac_sessions[self._current_session_idx]
            self._current_session_idx = (self._current_session_idx + 1) % len(self._snac_sessions)
            return session

    def _token_to_id(self, token_text: str, index: int) -> int | None:
        token_string = token_text.strip()
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

        if last_token_start == -1:
            return None

        last_token = token_string[last_token_start:]

        if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError:
                return None
        else:
            return None

    def _decode(
        self, token_gen: Generator[str, None, None]
    ) -> Generator[np.ndarray, None, None]:
        """Optimized token decoder with parallel audio generation."""
        buffer = []
        count = 0
        futures = []
        
        for token_text in token_gen:
            token = self._token_to_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Convert to audio when we have enough tokens
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:].copy()
                    
                    # Submit audio conversion to thread pool for parallel processing
                    future = self._executor.submit(self._convert_to_audio, buffer_to_proc)
                    futures.append(future)
                    
                    # Yield completed audio samples
                    completed_futures = [f for f in futures if f.done()]
                    for future in completed_futures:
                        try:
                            audio_samples = future.result()
                            if audio_samples is not None:
                                yield audio_samples
                        except Exception as e:
                            print(f"Audio conversion error: {e}")
                        futures.remove(future)
        
        # Process remaining futures
        for future in futures:
            try:
                audio_samples = future.result(timeout=5.0)
                if audio_samples is not None:
                    yield audio_samples
            except Exception as e:
                print(f"Final audio conversion error: {e}")

    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        """Optimized audio conversion with GPU acceleration."""
        if len(multiframe) < 28:
            return None

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        # Use CuPy for GPU-accelerated array operations if available
        if CUPY_AVAILABLE:
            try:
                return self._convert_to_audio_gpu(frame, num_frames)
            except Exception as e:
                print(f"GPU conversion failed, falling back to CPU: {e}")
        
        return self._convert_to_audio_cpu(frame, num_frames)

    def _convert_to_audio_gpu(self, frame: list[int], num_frames: int) -> np.ndarray | None:
        """GPU-accelerated audio conversion using CuPy."""
        # Initialize GPU arrays
        codes_0 = cp.array([], dtype=cp.int32)
        codes_1 = cp.array([], dtype=cp.int32)
        codes_2 = cp.array([], dtype=cp.int32)

        for j in range(num_frames):
            i = 7 * j
            codes_0 = cp.append(codes_0, frame[i])
            codes_1 = cp.append(codes_1, [frame[i + 1], frame[i + 4]])
            codes_2 = cp.append(codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])

        # Reshape arrays
        codes_0 = cp.expand_dims(codes_0, axis=0)
        codes_1 = cp.expand_dims(codes_1, axis=0)
        codes_2 = cp.expand_dims(codes_2, axis=0)

        # Validate ranges on GPU
        if (cp.any(codes_0 < 0) or cp.any(codes_0 > 4096) or
            cp.any(codes_1 < 0) or cp.any(codes_1 > 4096) or
            cp.any(codes_2 < 0) or cp.any(codes_2 > 4096)):
            return None

        # Convert to CPU arrays for ONNX (ONNX Runtime handles GPU internally)
        codes_0_cpu = cp.asnumpy(codes_0)
        codes_1_cpu = cp.asnumpy(codes_1)
        codes_2_cpu = cp.asnumpy(codes_2)

        return self._run_snac_inference(codes_0_cpu, codes_1_cpu, codes_2_cpu)

    def _convert_to_audio_cpu(self, frame: list[int], num_frames: int) -> np.ndarray | None:
        """CPU-based audio conversion (fallback)."""
        codes_0 = np.array([], dtype=np.int32)
        codes_1 = np.array([], dtype=np.int32)
        codes_2 = np.array([], dtype=np.int32)

        for j in range(num_frames):
            i = 7 * j
            codes_0 = np.append(codes_0, frame[i])
            codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])
            codes_2 = np.append(codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])

        codes_0 = np.expand_dims(codes_0, axis=0)
        codes_1 = np.expand_dims(codes_1, axis=0)
        codes_2 = np.expand_dims(codes_2, axis=0)

        if (np.any(codes_0 < 0) or np.any(codes_0 > 4096) or
            np.any(codes_1 < 0) or np.any(codes_1 > 4096) or
            np.any(codes_2 < 0) or np.any(codes_2 > 4096)):
            return None

        return self._run_snac_inference(codes_0, codes_1, codes_2)

    def _run_snac_inference(self, codes_0, codes_1, codes_2) -> np.ndarray | None:
        """Run SNAC inference with load balancing across sessions."""
        session = self._get_next_session()
        
        try:
            input_names = [x.name for x in session.get_inputs()]
            input_dict = dict(zip(input_names, [codes_0, codes_1, codes_2]))
            
            # Run inference
            audio_hat = session.run(None, input_dict)[0]
            
            # Process output
            audio_np = audio_hat[:, :, 2048:4096]
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return audio_int16.tobytes()
            
        except Exception as e:
            print(f"SNAC inference error: {e}")
            return None

    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        buffer = []
        for _, array in self.stream_tts_sync(text, options):
            buffer.append(array)
        return (24_000, np.concatenate(buffer, axis=1))

    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        queue = asyncio.Queue(maxsize=10)  # Limit queue size to prevent memory issues
        finished = asyncio.Event()

        def stream_to_queue(text, options, queue, finished):
            try:
                for chunk in self.stream_tts_sync(text, options):
                    queue.put_nowait(chunk)
            except Exception as e:
                print(f"Streaming error: {e}")
            finally:
                finished.set()

        thread = threading.Thread(
            target=stream_to_queue, args=(text, options, queue, finished)
        )
        thread.start()
        
        while not finished.is_set():
            try:
                yield await asyncio.wait_for(queue.get(), 0.1)
            except (asyncio.TimeoutError, TimeoutError):
                pass
                
        while not queue.empty():
            chunk = queue.get_nowait()
            yield chunk

    def _token_gen(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[str, None, None]:
        from llama_cpp import CreateCompletionStreamResponse

        options = options or TTSOptions()
        voice_id = options.get("voice_id", "tara")
        text = f"<|audio|>{voice_id}: {text}<|eot_id|><custom_token_4>"
        
        token_gen = self._llm(
            text,
            max_tokens=options.get("max_tokens", 2_048),
            stream=True,
            temperature=options.get("temperature", 0.8),
            top_p=options.get("top_p", 0.95),
            top_k=options.get("top_k", 40),
            min_p=options.get("min_p", 0.05),
            repeat_penalty=1.1,  # Prevent repetition
        )
        
        for token in cast(Iterator[CreateCompletionStreamResponse], token_gen):
            yield token["choices"][0]["text"]

    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        options = options or TTSOptions()
        token_gen = self._token_gen(text, options)
        pre_buffer = np.array([], dtype=np.int16).reshape(1, 0)
        pre_buffer_size = int(24_000 * options.get("pre_buffer_size", 0.5))  # Reduced for faster response
        started_playback = False
        
        for audio_bytes in self._decode(token_gen):
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
            if not started_playback:
                pre_buffer = np.concatenate([pre_buffer, audio_array], axis=1)
                if pre_buffer.shape[1] >= pre_buffer_size:
                    started_playback = True
                    yield (24_000, pre_buffer)
            else:
                yield (24_000, audio_array)
                
        if not started_playback and pre_buffer.shape[1] > 0:
            yield (24_000, pre_buffer)

    def batch_tts(
        self, texts: list[str], options: TTSOptions | None = None
    ) -> list[tuple[int, NDArray[np.int16]]]:
        """Batch processing for multiple texts with parallel execution."""
        with ThreadPoolExecutor(max_workers=len(self._snac_sessions)) as executor:
            futures = [executor.submit(self.tts, text, options) for text in texts]
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=60))
                except Exception as e:
                    print(f"Batch TTS error: {e}")
                    results.append((24_000, np.array([], dtype=np.int16).reshape(1, 0)))
            return results

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
