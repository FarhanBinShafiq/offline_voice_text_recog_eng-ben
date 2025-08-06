from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import wave
import os

app = Flask(__name__)

model = Model("vosk-model-small-en-in-0.4")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files["audio"]
    audio.save("temp_upload")
    # Convert to WAV (PCM 16kHz mono)
    sound = AudioSegment.from_file("temp_upload")
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export("temp.wav", format="wav")
    os.remove("temp_upload")

    wf = wave.open("temp.wav", "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(rec.Result())
    results.append(rec.FinalResult())
    wf.close()
    os.remove("temp.wav")

    text = ""
    for res in results:
        part = eval(res).get("text", "")
        if part:
            text += part + " "
    return jsonify({"text": text.strip()})

if __name__ == "__main__":
    app.run(debug=True)