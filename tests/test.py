import numpy as np
from scipy.io.wavfile import write

from orpheus_cpp import OrpheusCpp

orpheus = OrpheusCpp(verbose=False)

text = "I really hope the project deadline doesn't get moved up again."
buffer = []
for i, (sr, chunk) in enumerate(
    orpheus.stream_tts_sync(text, options={"voice_id": "tara"})
):
    buffer.append(chunk)
    print(f"Generated chunk {i}")
buffer = np.concatenate(buffer, axis=1)
write("output.wav", 24_000, np.concatenate(buffer))
