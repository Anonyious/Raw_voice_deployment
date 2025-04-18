from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze_audio
import os
import uuid
import traceback

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Temp directory setup
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded file
        file_ext = os.path.splitext(file.filename)[1]
        temp_filename = f"audio_{uuid.uuid4().hex}{file_ext}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process file
        result = analyze_audio(temp_path)
        return result

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("🔥 Exception Traceback:\n", traceback_str)
        return {"error": f"Internal Server Error: {str(e)}"}

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
