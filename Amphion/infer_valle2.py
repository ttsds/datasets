import os
import sys
from pathlib import Path
from tqdm import tqdm

if os.path.isdir("Amphion"):
    os.chdir("Amphion")
sys.path.append(".")

import torch
import librosa
import torchaudio
from models.tts.valle_v2.valle_inference import ValleInference
from models.tts.valle_v2.g2p_processor import G2pProcessor

model = None


def inference(
    prompt_audio_path,
    prompt_text_path,
    target_text_path,
    ar_model_path,
    nar_model_path,
    speechtokenizer_path,
    device=None,
    use_vocos=True,
    inference_configs=None,
):
    global model
    # Initialize the device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    if model is None:
        model = ValleInference(
            ar_path=ar_model_path,
            nar_path=nar_model_path,
            speechtokenizer_path=speechtokenizer_path,
            device=device,
        )

    # Load the prompt audio
    wav, _ = librosa.load(prompt_audio_path, sr=16000)
    wav = torch.tensor(wav, dtype=torch.float32)

    # Read the prompt and target transcripts
    with open(prompt_text_path, "r") as f:
        prompt_transcript_text = f.read().strip()

    with open(target_text_path, "r") as f:
        target_transcript_text = f.read().strip()

    # Process transcripts
    g2p = G2pProcessor()
    prompt_transcript = g2p(prompt_transcript_text, "en")[1]
    target_transcript = g2p(target_transcript_text, "en")[1]

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
                top_k=100,
                temperature=1.0,
                repeat_penalty=1.0,
                max_length=100000,
                num_beams=1,
            )
        ]

    # Perform inference
    output_wav = model(batch, inference_configs, return_prompt=False).squeeze(0)
    return output_wav


Path("../results_valle2").mkdir(exist_ok=True)
for i in tqdm(range(1, 101)):
    print(
        Path(f"../results_valle2/{i:03}.wav").exists(),
        Path(f"../results_valle2/{i:03}.wav"),
    )
    if Path(f"../results_valle2/{i:03}.wav").exists():
        continue
    prompt_audio_path = Path(f"../../test/{i:03}.wav")
    prompt_text_path = Path(f"../../test/{i:03}.orig.txt")
    target_text_path = Path(f"../../test/{i:03}.txt")
    ar_model_path = "ckpts/tts/valle2/valle_ar_mls_196000.bin"
    nar_model_path = "ckpts/tts/valle2/valle_nar_mls_164000.bin"
    speechtokenizer_path = "ckpts/tts/valle2"
    try:
        output_wav = inference(
            prompt_audio_path,
            prompt_text_path,
            target_text_path,
            ar_model_path,
            nar_model_path,
            speechtokenizer_path,
            device="cpu",
        )
        if output_wav.shape[1] < 16000 // 2:
            print("Skipping short audio")
            continue
        torchaudio.save(f"../results_valle2/{i:03}.wav", output_wav, 16000)
        with open(f"../results_valle2/{i:03}.txt", "w") as f:
            f.write(target_text_path.read_text())
    except Exception as e:
        print("Skipping", e)
