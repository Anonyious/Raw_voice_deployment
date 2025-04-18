import whisper
import librosa
import numpy as np
import os
import subprocess

# Load Whisper model once
model = whisper.load_model("base")

def convert_to_wav(input_path, output_path):
    try:
        command = [
            os.path.join(os.path.dirname(__file__), "ffmpeg.exe"),  # Use downloaded ffmpeg
            "-y",  # Overwrite output file if exists
            "-i", input_path,
            output_path
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert to WAV: {str(e)}")

def analyze_audio(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        # Load with librosa
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y ** 2))
        pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
        valid_pitches = pitch[(pitch > 0) & ~np.isnan(pitch)]
        avg_pitch = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0.0

        # Whisper wants proper WAV, convert if needed
        base, ext = os.path.splitext(file_path)
        converted_path = base + "_converted.wav"
        convert_to_wav(file_path, converted_path)

        # Transcribe
        result = model.transcribe(converted_path)
        transcript = result["text"].strip()

        # Clean up
        if os.path.exists(converted_path):
            os.remove(converted_path)

        return {
            "filename": os.path.basename(file_path),
            "duration_sec": round(duration, 2),
            "rms": round(rms, 4),
            "avg_pitch_hz": round(avg_pitch, 2),
            "transcript": transcript
        }

    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}")
