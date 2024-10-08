from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import os

from fam.llm.fast_inference import TTS

app = FastAPI()

tts = TTS()

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    if version not in ["metavoice"]:
        return {"error": "Invalid version"}

    top_p = 0.95
    guidance_scale = 3.0
    temperature = 1.0

    # Clear previous results and create results directory
    shutil.rmtree("/results_metavoice", ignore_errors=True)
    Path("/results_metavoice").mkdir()

    # Save the uploaded speaker_wav to a temporary file
    speaker_wav_path = process_speaker_reference(speaker_wav.file.read())

    try:
        wav_file_path = tts.synthesise(
            text=text,
            spk_ref_path=speaker_wav_path,
            top_p=top_p,
            guidance_scale=guidance_scale,
            temperature=temperature,
        )

        output_path = "/results_metavoice/output.wav"
        # Copy the generated wav file to the output path
        shutil.copy(wav_file_path, output_path)
        return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        return {"error": str(e)}

def process_speaker_reference(speaker_wav_bytes: bytes) -> str:
    speaker_wav_path = "/app/speaker.wav"
    with open(speaker_wav_path, "wb") as f:
        f.write(speaker_wav_bytes)
    return speaker_wav_path

@app.get("/info")
def info():
    return {
        "versions": ["metavoice"],
        "requires_text": [False],
    }

@app.get("/ready")
def ready():
    return "ready"