from vosk import Model, KaldiRecognizer
import pyaudio
import json
import time

model = Model("./vosk-model-small-en-in-0.4")
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=8192)
stream.start_stream()

try:
    print("Speak something... (say 'exit' or 'quit' to stop)")
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print(f"[{time.strftime('%H:%M:%S')}] You said:", text)
                if text.lower() in ["exit", "quit"]:
                    print("Exiting by voice command.")
                    break
        else:
            partial = json.loads(recognizer.PartialResult()).get("partial", "")
            if partial:
                print(f"Listening: {partial}", end="\r")
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    mic.terminate()