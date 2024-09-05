from pathlib import Path

from fam.llm.fast_inference import TTS

tts = TTS()

import sys
import os
from pathlib import Path
from tqdm import tqdm

Path("results").mkdir(exist_ok=True)

for i in tqdm(range(1, 101)):
    print(
        Path(f"results/{i:03}.wav").exists(),
        Path(f"results/{i:03}.wav"),
    )
    if Path(f"results/{i:03}.wav").exists():
        continue
    item = Path(f"../test/{i:03}.wav")
    audio_prompt = item
    infer_text = item.with_suffix(".txt").read_text()
    try:
        wav_file = Path(
            tts.synthesise(
                text=infer_text,
                spk_ref_path=str(audio_prompt),
                top_p=0.95,
                guidance_scale=3.0,
                temperature=1.0,
            )
        )
        Path(f"results/{i:03}.wav").write_bytes(wav_file.read_bytes())
        with open(f"results/{i:03}.txt", "w") as f:
            f.write(infer_text)
    except Exception as e:
        print("skip", e)
