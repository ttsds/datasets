import sys
from pathlib import Path

sys.path.append("bark-vc")

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from encodec import EncodecModel
from encodec.utils import convert_audio
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav

model = ("quantifier_V1_hubert_base_ls960_23.pth", "tokenizer_large.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading HuBERT...")
hubert_model = CustomHubert(HuBERTManager.make_sure_hubert_installed(), device=device)
print("Loading Quantizer...")
quant_model = CustomTokenizer.load_from_checkpoint(
    HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]),
    device,
)
print("Loading Encodec...")
encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model.to(device)

print("Downloaded and loaded models!")


def create_speaker_npz(audio_path, output_path):
    wav_file = audio_path
    out_file = output_path

    wav, sr = torchaudio.load(wav_file)

    wav_hubert = wav.to(device)

    if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
        wav_hubert = wav_hubert.mean(0, keepdim=True)

    print("Extracting semantics...")
    semantic_vectors = hubert_model.forward(wav_hubert, input_sample_hz=sr)
    print("Tokenizing semantics...")
    semantic_tokens = quant_model.get_token(semantic_vectors)
    print("Creating coarse and fine prompts...")
    wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)

    wav = wav.to(device)

    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    codes = codes.cpu()
    semantic_tokens = semantic_tokens.cpu()

    np.savez(
        out_file,
        semantic_prompt=semantic_tokens,
        fine_prompt=codes,
        coarse_prompt=codes[:2, :],
    )


Path("results").mkdir(exist_ok=True)
for i in tqdm(range(1, 101)):
    if not Path(f"../test/bark_{i:03}.npz").exists():
        create_speaker_npz(f"../test/{i:03}.wav", f"../test/bark_{i:03}.npz")
    if Path(f"results/{i:03}.wav").exists():
        continue
    text_prompt = open(f"../test/{i:03}.txt").read()
    try:
        audio_array = generate_audio(
            text_prompt, history_prompt=f"../test/bark_{i:03}.npz"
        )
        # save audio to disk
        write_wav(f"results/{i:03}.wav", SAMPLE_RATE, audio_array)
        # save text to disk
        with open(f"results/{i:03}.txt", "w") as f:
            f.write(text_prompt)
    except Exception as e:
        print(f"Failed to generate audio for {i:03}.txt: {e}, skipping")
