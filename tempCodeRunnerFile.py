# analyzer.py
import whisper
import librosa
import numpy as np
import os

model = whisper.load_model("base")

def analyze_audio(file_path):
    # Load and process audio
    y, sr = librosa.load(file_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = np.sqrt(np.mean(y**2))
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    avg_pitch = np.mean(pitch)

    # Transcription
    result = model.transcribe(file_path)
    transcript = result["text"]

    return {
        "filename": os.path.basename(file_path),
        "duration_sec": round(duration, 2),
        "rms": round(rms, 4),
        "avg_pitch_hz": round(avg_pitch, 2),
        "transcript": transcript.strip()
    }
