from pathlib import Path
from argparse import ArgumentParser
import sys
import os

os.chdir("Amphion/Amphion")
sys.path.append(".")

import torch
import numpy as np
import torchaudio
import librosa
from models.tts.valle_v2.valle_inference import ValleInference
from models.tts.valle_v2.g2p_processor import G2pProcessor

model = None

ar_model_path = "ckpts/tts/valle2/valle_ar_mls_196000.bin"
nar_model_path = "ckpts/tts/valle2/valle_nar_mls_164000.bin"
speechtokenizer_path = "ckpts/tts/valle2"

def inference(
    prompt_audio_path,
    prompt_text,
    target_text,
    ar_model_path,
    nar_model_path,
    speechtokenizer_path,
    device=None,
    inference_configs=None,
):
    # Initialize the device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = ValleInference(
        ar_path=ar_model_path,
        nar_path=nar_model_path,
        speechtokenizer_path=speechtokenizer_path,
        device=device,
    )

    # Load the prompt audio
    wav, _ = librosa.load(prompt_audio_path, sr=16000)
    wav = wav / np.max(np.abs(wav))
    wav = torch.tensor(wav).float()
    prompt_text = str(prompt_text)
    target_text = str(target_text)
    # remove punctuation from prompt text and lowercase
    prompt_text = prompt_text.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "")

    # Process transcripts
    g2p = G2pProcessor()
    
    prompt_transcript = g2p(prompt_text, "en")[1]
    target_transcript = g2p(target_text, "en")[1]

    prompt_transcript = torch.tensor(prompt_transcript).long()
    target_transcript = torch.tensor(target_transcript).long()
    transcript = torch.cat([prompt_transcript, target_transcript], dim=-1)

    # Prepare the batch
    batch = {
        "speech": wav.unsqueeze(0),
        "phone_ids": transcript.unsqueeze(0),
    }

    # Default inference configurations if not provided
    if inference_configs is None:
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

    # Perform inference
    output_wav = model(batch, inference_configs, return_prompt=False).squeeze(0).cpu().numpy()
    return output_wav

if __name__ == "__main__":
    command_mode = sys.argv[1]
    if command_mode == "single":
        parser = ArgumentParser()
        parser.add_argument("mode", type=str)
        parser.add_argument("text", type=str)
        parser.add_argument("reference_audio", type=Path)
        parser.add_argument("reference_text", type=Path)
        parser.add_argument("output", type=Path)
    elif command_mode == "batch":
        parser = ArgumentParser()
        parser.add_argument("mode", type=str)
        parser.add_argument("input_file", type=Path)
        parser.add_argument("output_directory", type=Path)

    args = parser.parse_args()

    if command_mode == "single":
        output = inference(
            args.reference_audio,
            args.reference_text,
            args.text,
            ar_model_path,
            nar_model_path,
            speechtokenizer_path,
        )
        torchaudio.save(args.output, torch.tensor(output), 16000)
    elif command_mode == "batch":
        with open(args.input_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            text, reference_audio, reference_text = line.strip().split(",")
            output = inference(
                Path(reference_audio),
                Path(reference_text),
                text,
                ar_model_path,
                nar_model_path,
                speechtokenizer_path,
            )
            torchaudio.save(args.output_directory / f"{i:10}.wav", torch.tensor(output), 16000)