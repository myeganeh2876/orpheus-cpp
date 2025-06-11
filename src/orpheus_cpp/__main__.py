import importlib.util

if importlib.util.find_spec("fastrtc") is None:
    raise RuntimeError(
        "fastrtc is not installed. Please install it using 'pip install fastrtc>=0.0.17'."
    )

import asyncio
import json
import random

import gradio as gr
import httpx
import numpy as np
import numpy.typing as npt
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    AudioEmitType,
    Stream,
    wait_for_item,
)
from fastrtc.utils import create_message
from huggingface_hub import InferenceClient

from orpheus_cpp.model import OrpheusCpp

async_client = httpx.AsyncClient()

client = InferenceClient(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct", provider="sambanova"
)


language_code_to_language = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "ko": "Korean",
    "hi": "Hindi",
}


def generate_message(lang: str):
    system_prompt = """You are a creative text generator that generates short sentences from everyday life across multiple languages.
Only response with the sentence in the target language. No other text!
Example: "Hello!  I'm so excited to talk to you! This is going to be fun!"
Example: I'm nervous about the interview tomorrow
Example: Hola, ¿cómo estás?
Example: 안녕하세요, 오늘 기분이 좋아요!
Example: こんにちは、今日は元気ですか？
Example: नमस्ते, आप कैसे हैं?
Example: "Ciao, come stai?"
"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Give me a short sentence please for this language code: {language_code_to_language[lang]}.",
            },
        ],
        max_tokens=100,
        seed=random.randint(0, 1000000),
    )
    msg = response.choices[0].message.content
    if msg:
        msg = msg.replace('"', "")
    return msg


class OrpheusStream(AsyncStreamHandler):
    def __init__(self):
        super().__init__(output_sample_rate=24_000, output_frame_size=480)
        self.latest_msg = ""
        self.latest_voice_id = "tara"
        self.audio_queue: asyncio.Queue[AudioEmitType] = asyncio.Queue()
        self.trigger_event = asyncio.Event()

    async def start_up(self):
        await self.wait_for_args()

    async def receive(self, frame: tuple[int, npt.NDArray[np.int16]]) -> None:
        msg, cb, lang, voice_id, _ = self.latest_args[1:]
        if (
            msg != self.latest_msg
            or voice_id != self.latest_voice_id
            or self.trigger_event.is_set()
        ):
            await self.send_message(create_message("log", "pause_detected"))
            if language_to_model[lang] is None:
                await self.send_message(
                    json.dumps({"type": "warning", "message": "loading_model"})
                )
                language_to_model[lang] = OrpheusCpp(lang=lang)
            model = language_to_model[lang]
            # Initialize variables
            all_audio = np.array([], dtype=np.int16)
            started_playback = False

            async for sample_rate, chunk in model.stream_tts(
                msg, options={"voice_id": voice_id}
            ):
                all_audio = np.concatenate([all_audio, chunk.squeeze()])
                if not started_playback:
                    started_playback = True
                    await self.send_message(create_message("log", "response_starting"))
                await self.audio_queue.put((sample_rate, chunk))

            cb.append({"role": "user", "content": msg})
            cb.append(
                {
                    "role": "assistant",
                    "content": gr.Audio(value=(sample_rate, all_audio)),
                }
            )
            await self.audio_queue.put(AdditionalOutputs(cb))
            self.latest_msg = msg
            self.latest_voice_id = voice_id
            self.trigger_event.clear()

    async def emit(self) -> AudioEmitType:
        return await wait_for_item(self.audio_queue)

    def copy(self):
        return OrpheusStream()


chat = gr.Chatbot(
    label="Conversation",
    type="messages",
    allow_tags=[
        "giggle",
        "laugh",
        "chuckle",
        "sigh",
        "cough",
        "sniffle",
        "groan",
        "yawn",
        "gasp",
    ],
)
generate = gr.Button(
    value="Generate Prompt",
)

language_to_voice = {
    "en": ["tara", "jess", "leah", "leo", "dan", "mia", "zac", "zoe"],
    "es": ["javi", "sergio", "maria"],
    "fr": ["pierre", "amelie", "marie"],
    "de": ["jana", "thomas", "max"],
    "it": ["pietro", "giulia", "carlo"],
    "zh": ["长乐", "白芷"],
    "ko": ["유나", "준서"],
    "hi": ["ऋतिका"],
}

language_to_model = {}

prompt = gr.Textbox(label="Prompt", value="Hello, how are you?")
available_languages = ["en", "fr", "es", "de", "it", "zh", "ko", "hi"]
for lang in available_languages:
    language_to_model[lang] = None
language_to_model["en"] = OrpheusCpp(lang="en")

language_dropdown = gr.Dropdown(
    choices=available_languages, value="en", label="Language"
)
voice_dropdown = gr.Dropdown(
    choices=language_to_voice["en"], value="tara", label="Voice"
)

stream = Stream(
    OrpheusStream(),
    modality="audio",
    mode="send-receive",
    additional_inputs=[
        prompt,
        chat,
        language_dropdown,
        voice_dropdown,
        generate,
    ],
    additional_outputs=[chat],
    additional_outputs_handler=lambda old, new: new,
    ui_args={
        "title": "Orpheus.cpp - Fast Streaming TTS over WebRTC",
        "subtitle": "Powered by FastRTC ⚡️ + llama.cpp 🦙",
        "send_input_on": "submit",
    },
)


def trigger_event(webrtc_id: str):
    stream.webrtc_component.handlers[webrtc_id].trigger_event.set()  # type: ignore


with stream.ui:
    prompt.submit(
        trigger_event,  # type: ignore
        inputs=[stream.webrtc_component],
        outputs=None,
    )
    generate.click(generate_message, inputs=[language_dropdown], outputs=[prompt])
    language_dropdown.change(
        lambda lang: gr.Dropdown(
            choices=language_to_voice[lang], value=language_to_voice[lang][0]
        ),
        inputs=[language_dropdown],
        outputs=[voice_dropdown],
    )


stream.ui.launch()
