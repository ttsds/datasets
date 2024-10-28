import os
import sys
import numpy as np
import torch
import torchaudio
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5ForSpeechToSpeech,
)
import soundfile as sf
from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

app = FastAPI()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and models at startup
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
speaker_model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc").to(device)

def process_speaker_reference(speaker_wav: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/tmp") as tmp:
        tmp.write(speaker_wav)
        return tmp.name

def synthesize_speecht5(text, reference_audio_path, output_path):
    # Load and prepare reference audio
    speech_array, sampling_rate = torchaudio.load(reference_audio_path)
    speech_array = speech_array.squeeze()
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)
        sampling_rate = 16000

    speech_array = speech_array.to(device)

    with torch.no_grad():
        # Extract speaker embeddings
        speaker_embeddings = speaker_model.encoder(speech_array.unsqueeze(0))

    # Prepare input text
    inputs = processor(text=text, return_tensors="pt").to(device)

    # Generate speech
    with torch.no_grad():
        speech = tts_model.generate_speech(
            inputs["input_ids"], speaker_embeddings, vocoder=vocoder
        )

    # Save output speech
    speech = speech.cpu()
    sf.write(output_path, speech.numpy(), samplerate=16000)
    print(f"Speech has been saved to {output_path}")

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    speaker_wav: UploadFile = File(...),
):
    # Process the speaker reference
    audio_prompt = process_speaker_reference(speaker_wav.file.read())
    # Create temporary output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/tmp") as tmp_output:
        output_path = tmp_output.name
    # Run the TTS
    synthesize_speecht5(text, audio_prompt, output_path)
    # Return the output file
    return FileResponse(output_path, media_type="audio/wav")

@app.get("/info")
def info():
    return {
        "versions": ["SpeechT5"],
        "requires_text": [True],
    }

@app.get("/ready")
def ready():
    return "ready"
