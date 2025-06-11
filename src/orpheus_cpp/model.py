import asyncio
import platform
import sys
import threading
from typing import (
    AsyncGenerator,
    Generator,
    Iterator,
    Literal,
    cast,
)

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from typing_extensions import NotRequired, TypedDict


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


class OrpheusCpp:
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
        n_gpu_layers: int = 0,
        n_threads: int = 0,
        verbose: bool = True,
        lang: Literal["en", "es", "ko", "fr"] = "es",
    ):
        import importlib.util

        if importlib.util.find_spec("llama_cpp") is None:
            if sys.platform == "darwin":
                # Check if macOS 11.0+ on arm64 (Apple Silicon)
                is_arm64 = platform.machine() == "arm64"
                version = platform.mac_ver()[0].split(".")
                is_macos_11_plus = len(version) >= 2 and int(version[0]) >= 11
                is_macos_10_less = len(version) >= 2 and int(version[0]) < 11

                if is_arm64 and is_macos_11_plus:
                    extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal"
                elif is_macos_10_less:
                    raise ImportError(
                        "llama_cpp does not have pre-built wheels for macOS 10.x "
                        "Follow install instructions at https://github.com/abetlen/llama-cpp-python"
                    )
                else:
                    extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            else:
                extra_index_url = "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"

            raise ImportError(
                f"llama_cpp is not installed. Please install it using `pip install llama-cpp-python {extra_index_url}`."
            )
        repo_id = self.lang_to_model[lang]
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=repo_id.split("/")[-1].lower().replace("-gguf", ".gguf"),
        )
        from llama_cpp import Llama

        if n_gpu_layers == 0:
            print(
                "Running model without GPU Acceleration. Please set n_gpu_layers parameters to control the number of layers to offload to GPU."
            )

        self._llm = Llama(
            model_path=model_file,
            n_ctx=0,
            verbose=verbose,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            batch_size=1,
        )

        repo_id = "onnx-community/snac_24khz-ONNX"
        snac_model_file = "decoder_model.onnx"
        snac_model_path = hf_hub_download(
            repo_id, subfolder="onnx", filename=snac_model_file
        )

        # Load SNAC model with optimizations
        self._snac_session = onnxruntime.InferenceSession(
            snac_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def _token_to_id(self, token_text: str, index: int) -> int | None:
        token_string = token_text.strip()

        # Find the last token in the string
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

        if last_token_start == -1:
            return None

        # Extract the last token
        last_token = token_string[last_token_start:]

        # Process the last token
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
        """Asynchronous token decoder that converts token stream to audio stream."""
        buffer = []
        count = 0
        for token_text in token_gen:
            token = self._token_to_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Convert to audio when we have enough tokens
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = self._convert_to_audio(buffer_to_proc)
                    if audio_samples is not None:
                        yield audio_samples

    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        if len(multiframe) < 28:  # Ensure we have enough tokens
            return None

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        # Initialize empty numpy arrays instead of torch tensors
        codes_0 = np.array([], dtype=np.int32)
        codes_1 = np.array([], dtype=np.int32)
        codes_2 = np.array([], dtype=np.int32)

        for j in range(num_frames):
            i = 7 * j
            # Append values to numpy arrays
            codes_0 = np.append(codes_0, frame[i])

            codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])

            codes_2 = np.append(
                codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]]
            )

        # Reshape arrays to match the expected input format (add batch dimension)
        codes_0 = np.expand_dims(codes_0, axis=0)
        codes_1 = np.expand_dims(codes_1, axis=0)
        codes_2 = np.expand_dims(codes_2, axis=0)

        # Check that all tokens are between 0 and 4096
        if (
            np.any(codes_0 < 0)
            or np.any(codes_0 > 4096)
            or np.any(codes_1 < 0)
            or np.any(codes_1 > 4096)
            or np.any(codes_2 < 0)
            or np.any(codes_2 > 4096)
        ):
            return None

        # Create input dictionary for ONNX session

        snac_input_names = [x.name for x in self._snac_session.get_inputs()]

        input_dict = dict(zip(snac_input_names, [codes_0, codes_1, codes_2]))

        # Run inference
        audio_hat = self._snac_session.run(None, input_dict)[0]

        # Process output
        audio_np = audio_hat[:, :, 2048:4096]
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

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
        queue = asyncio.Queue()
        finished = asyncio.Event()

        def strem_to_queue(text, options, queue, finished):
            for chunk in self.stream_tts_sync(text, options):
                queue.put_nowait(chunk)
            finished.set()

        thread = threading.Thread(
            target=strem_to_queue, args=(text, options, queue, finished)
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
        )
        for token in cast(Iterator[CreateCompletionStreamResponse], token_gen):
            yield token["choices"][0]["text"]

    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        options = options or TTSOptions()
        token_gen = self._token_gen(text, options)
        pre_buffer = np.array([], dtype=np.int16).reshape(1, 0)
        pre_buffer_size = 24_000 * options.get("pre_buffer_size", 1.5)
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
        if not started_playback:
            yield (24_000, pre_buffer)
