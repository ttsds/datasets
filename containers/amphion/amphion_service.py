import tempfile
import os
import sys
from pathlib import Path
import subprocess
import shutil

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import librosa
import numpy as np
import torch
import soundfile as sf

os.chdir("/app/Amphion")
sys.path.append(".")

from models.tts.valle_v2.valle_inference import ValleInference
from models.tts.valle_v2.g2p_processor import G2pProcessor

app = FastAPI()

@app.post("/synthesize", response_class=FileResponse)
def synthesize(text: str, version: str, speaker_wav: UploadFile, speaker_txt: str):
    if version not in ["NaturalSpeech 2", "VALL-E v1", "VALL-E v2"]:
        return {"error": "Invalid version"}
    if version == "NaturalSpeech 2":
        os.environ.update({"WORK_DIR": "."})
        os.chdir("/app/Amphion")
        audio_prompt = process_speaker_reference(speaker_wav.file.read())
        try:
            shutil.rmtree("/results_naturalspeech2")
            os.mkdir("/results_naturalspeech2")
            sys.argv = [
                "bins/tts/inference.py",
                "--mode=single",
                "--config=egs/tts/NaturalSpeech2/exp_config.json",
                "--text=" + text,
                "--checkpoint_path=ckpts/tts/naturalspeech2_libritts/checkpoint/epoch-0089_step-0512912_loss-6.367693",
                "--ref_audio=" + audio_prompt,
                "--output_dir=/results_naturalspeech2",
            ]
            script = open("bins/tts/inference.py").read()
            script = "import sys\nsys.path.append('.')\n" + script
            exec(script)
            out_path = next(Path("/results_naturalspeech2").glob("*.wav"))
            return FileResponse(out_path)
        except Exception as e:
            return {"error": str(e)}
        finally:
            os.remove(audio_prompt)
    elif version == "VALL-E v1":
        os.chdir("/app/Amphion")
        shutil.rmtree("/results_valle1")
        os.mkdir("/results_valle1")
        output = "/results_valle1/output.wav"
        audio_prompt = process_speaker_reference(speaker_wav.file.read())
        text_prompt = process_text_reference(speaker_txt)
        subprocess.run(
            [
                "sh",
                "egs/tts/VALLE/run.sh",
                "--stage",
                "3",
                "--gpu",
                "0",
                "--config",
                "ckpts/tts/valle1/args.json",
                "--infer_expt_dir",
                "ckpts/tts/valle1",
                "--infer_output_dir",
                "/results_valle1",
                "--infer_mode",
                "single",
                "--infer_text",
                text,
                "--infer_text_prompt",
                text_prompt,
                "--infer_audio_prompt",
                audio_prompt,
            ],
            check=True,
            cwd="/app/Amphion",
        )
        return FileResponse(output)
    elif version == "VALL-E v2":
        ar_model_path = "ckpts/tts/valle2/valle_ar_mls_196000.bin"
        nar_model_path = "ckpts/tts/valle2/valle_nar_mls_164000.bin"
        speechtokenizer_path = "ckpts/tts/valle2"
        model = ValleInference(
            ar_path=ar_model_path,
            nar_path=nar_model_path,
            speechtokenizer_path=speechtokenizer_path,
        )
        wav, _ = librosa.load(speaker_wav.file, sr=16000)
        wav = wav / np.max(np.abs(wav))
        wav = torch.tensor(wav).float()
        g2p = G2pProcessor
        prompt_transcript = g2p(text, "en")[1]
        target_transcript = g2p(text, "en")[1]
        prompt_transcript = torch.tensor(prompt_transcript).long()
        target_transcript = torch.tensor(target_transcript).long()
        transcript = torch.cat([prompt_transcript, target_transcript], dim=-1)
        batch = {
            "speech": wav.unsqueeze(0),
            "transcript": transcript.unsqueeze(0),
        }
        inference_configs = [
            dict(
                top_p=0.9,
                top_k=5,
                temperature=0.95,
                repeat_penalty=1.0,
                max_length=2000,
                num_beams=1,
            )
        ]
        output_wav = model(batch, inference_configs, return_prompt=False).squeeze(0).cpu().numpy()
        output_path = "/results_valle2/output.wav"
        sf.write(output_path, output_wav, 16000)
        return FileResponse(output_path)

        
def process_speaker_reference(self, speaker_wav: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(speaker_wav)
        temp.flush()
        return temp.name
    
def process_text_reference(self, speaker_txt: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp:
        temp.write(speaker_txt.encode())
        temp.flush()
        return temp.name

@app.get("/info")
def info():
    return {
         "versions": ["NaturalSpeech 2", "VALL-E v1", "VALL-E v2"],
         "needs_reference_transcript": False,
    }