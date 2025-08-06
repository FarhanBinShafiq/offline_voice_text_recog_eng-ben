import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile

import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

duration = 5  # seconds

print("কিছু বলুন...")  # Speak something...
fs = 16000
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    scipy.io.wavfile.write(f.name, fs, recording)
    audio_path = f.name

model = whisper.load_model("small")  # Try "small", "medium", or "large" for better accuracy
result = model.transcribe(audio_path, language="bn", task="transcribe")
print("আপনি বললেন:", result["text"])