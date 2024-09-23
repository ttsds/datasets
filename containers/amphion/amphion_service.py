import tempfile
import os
import sys
from pathlib import Path
import subprocess
import shutil
import argparse

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import librosa
import numpy as np
import torch
import soundfile as sf
from pydantic import BaseModel


os.chdir("/app/Amphion")
os.environ.update({"WORK_DIR": "/app/Amphion"})

sys.path.append(".")

from models.tts.valle_v2.valle_inference import ValleInference
from models.tts.valle_v2.g2p_processor import G2pProcessor
from models.tts.valle.valle_inference import VALLEInference
from models.tts.naturalspeech2.ns2_inference import NS2Inference
from utils.util import load_config

def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)

# all the defaults from build_parser (including none)
default_ns2_args = {
    "config": "egs/tts/NaturalSpeech2/exp_config.json",
    "dataset": None,
    "testing_set": "test",
    "test_list_file": None,
    "speaker_name": None,
    "text": "",
    "vocoder_dir": None,
    "acoustics_dir": None,
    "checkpoint_path": None,
    "mode": "single",
    "log_level": "warning",
    "pitch_control": 1.0,
    "energy_control": 1.0,
    "duration_control": 1.0,
    "output_dir": None,
}

default_valle_args = {
    "config": "ckpts/tts/valle1/args.json",
    "output_dir": "/results_valle1",
    "vocoder_dir": "ckpts/tts/valle1",
    "acoustics_dir": "ckpts/tts/valle1",
    "infer_mode": "single",
    "text_prompt": "",
    "audio_prompt": "",
    "top_k": -100,
    "temperature": 1.0,
    "continual": False,
    "copysyn": False,
    "mode": "single",
    "log_level": "debug",
} 


app = FastAPI()

def synthesize_ns2(
    text: str,
    audio_prompt: str,
):
    # without exec
    os.environ.update({"WORK_DIR": "."})
    os.chdir("/app/Amphion")
    shutil.rmtree("/results_naturalspeech2", ignore_errors=True)
    os.mkdir("/results_naturalspeech2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = {
        "mode": "single",
        "config": "egs/tts/NaturalSpeech2/exp_config.json",
        "text": text,
        "checkpoint_path": "ckpts/tts/naturalspeech2_libritts/checkpoint/epoch-0089_step-0512912_loss-6.367693",
        "ref_audio": audio_prompt,
        "output_dir": "/results_naturalspeech2",
        "device": device,
        "inference_step": 200,
    }
    # overwrite default args with the provided args
    for k, v in args.items():
        default_ns2_args[k] = v
    # into namespace
    args = argparse.Namespace(**default_ns2_args)
    cfg = load_config(args.config)
    if torch.cuda.is_available():
        cuda_relevant()
    # don't use build_inference, use NS2Inference directly
    inferencer = NS2Inference(args, cfg)
    inferencer.inference()
    return next(Path("/results_naturalspeech2").rglob("*.wav"))

def synthesize_valle(
    text: str,
    audio_prompt: str,
    text_prompt: str,
):
    os.environ.update({"WORK_DIR": "."})
    os.chdir("/app/Amphion")
    shutil.rmtree("/results_valle1", ignore_errors=True)
    os.mkdir("/results_valle1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = {
        "mode": "single",
        "config": "ckpts/tts/valle1/args.json",
        "output_dir": "/results_valle1",
        "infer_mode": "single",
        "text_prompt": text_prompt,
        "audio_prompt": audio_prompt,
        "top_k": -100,
        "temperature": 1.0,
        "continual": False,
        "copysyn": False,
        "text": text,
    }
    # overwrite default args with the provided args
    for k, v in args.items():
        default_valle_args[k] = v
    # into namespace
    args = argparse.Namespace(**default_valle_args)
    cfg = load_config(args.config)
    if torch.cuda.is_available():
        cuda_relevant()
    # don't use build_inference, use VALLEInference directly
    inferencer = VALLEInference(args, cfg)
    inferencer.inference()
    return next(Path("/results_valle1").rglob("*.wav"))

def synthesize_valle2(
    text: str,
    audio_prompt: str,
    text_prompt: str,
):
    os.environ.update({"WORK_DIR": "."})
    os.chdir("/app/Amphion")
    shutil.rmtree("/results_valle2", ignore_errors=True)
    os.mkdir("/results_valle2")
    ar_model_path = 'ckpts/tts/valle2/valle_ar_mls_196000.bin'
    nar_model_path = 'ckpts/tts/valle2/valle_nar_mls_164000.bin'
    speechtokenizer_path = 'ckpts/tts/valle2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValleInference(ar_path=ar_model_path, nar_path=nar_model_path, device=device, speechtokenizer_path=speechtokenizer_path)
    wav, _ = librosa.load(audio_prompt, sr=16000)
    # remove an additional 10ms from both ends
    wav = wav[int(0.01*16000):int(-0.01*16000)]
    print(text_prompt)
    g2p = G2pProcessor()
    prompt_transcript = g2p(text_prompt, 'en')[1]
    target_transcript = g2p(text, 'en')[1]
    prompt_transcript = torch.tensor(prompt_transcript).long()
    target_transcript = torch.tensor(target_transcript).long()
    transcript = torch.cat([prompt_transcript, target_transcript], dim=-1)
    wav = torch.tensor(wav).float()
    batch = {
            'speech': wav.unsqueeze(0),
            'phone_ids': transcript.unsqueeze(0),
    }
    configs = [dict(
        max_length=2000,
        temperature=1.0,
        top_k=100,
        top_p=0.9,
        repeat_penalty=1.0,
        num_beams=1,
    )] # model inference hyperparameters
    output_wav = model(batch, configs)[0][0].cpu().numpy()
    output_path = '/results_valle2/output.wav'
    sf.write(output_path, output_wav, 16000)
    return output_path

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    if version not in ["NaturalSpeech 2", "VALL-E v1", "VALL-E v2"]:
        return {"error": "Invalid version"}
    if version == "NaturalSpeech 2":
        audio_prompt = process_speaker_reference(speaker_wav.file.read())
        output = synthesize_ns2(text, audio_prompt)
        return FileResponse(output)
    elif version == "VALL-E v1":
        audio_prompt = process_speaker_reference(speaker_wav.file.read())
        text_prompt = process_text_reference(speaker_txt)
        output = synthesize_valle(text, audio_prompt, text_prompt)
        return FileResponse(output)
    elif version == "VALL-E v2":
        audio_prompt = process_speaker_reference(speaker_wav.file.read())
        text_prompt = process_text_reference(speaker_txt)
        output = synthesize_valle2(text, audio_prompt, text_prompt)
        return FileResponse(output)

        
def process_speaker_reference(speaker_wav: bytes):
    with open("/app/Amphion/speaker.wav", "wb") as f:
        f.write(speaker_wav)
    return "/app/Amphion/speaker.wav"
    
def process_text_reference(speaker_txt: str):
    with open("/app/Amphion/speaker.txt", "w") as f:
        f.write(speaker_txt)
    return "/app/Amphion/speaker.txt"

@app.get("/info")
def info():
    return {
         "versions": ["NaturalSpeech 2", "VALL-E v1", "VALL-E v2"],
         "requires_text": [False, True, True],
    }

@app.get("/ready")
def ready():
    return "ready"