# sh Amphion/egs/tts/VALLE/run.sh --stage 3 --gpu "0" \
#     --infer_expt_dir ckpts/tts/valle1 \
#     --infer_output_dir ../results \
#     --infer_mode "single" \
#     --infer_text "This is a clip of generated speech with the given text from a TTS model." \
#     --infer_text_prompt "Out of this incident, the officious sheriff managed most ingeniously to create an embroilment with the town of Lawrence, Buckley, who was alleged to have been accessory to the crime, obtained a peace warrant against Branson, a neighbor of the victim." \
#     --infer_audio_prompt ../../test/003.wav

from pathlib import Path
from tqdm import tqdm
import tarfile
import gzip
import subprocess
import shutil

Path("results").mkdir(exist_ok=True)

i = 0
for item in tqdm(sorted(list(Path("../test").rglob("*.wav"))), total=100):
    i += 1
    print(Path(f"results/{i:03}.wav").exists(), Path(f"results/{i:03}.wav"))
    if Path(f"results/{i:03}.wav").exists():
        continue
    audio_prompt = item
    text_prompt = item.with_suffix(".orig.txt").read_text()
    infer_text = item.with_suffix(".txt").read_text()
    try:
        subprocess.run(
            [
                "sh",
                "Amphion/egs/tts/VALLE/run.sh",
                "--stage",
                "3",
                "--gpu",
                "0",
                "--config",
                "ckpts/tts/valle1/args.json",
                "--infer_expt_dir",
                "ckpts/tts/valle1",
                "--infer_output_dir",
                "../results",
                "--infer_mode",
                "single",
                "--infer_text",
                infer_text,
                "--infer_text_prompt",
                text_prompt,
                "--infer_audio_prompt",
                "../" + str(audio_prompt),
            ],
            check=True,
        )
        shutil.move("results/single/test_pred.wav", f"results/{i:03}.wav")
        with open(f"results/{i:03}.txt", "w") as f:
            f.write(infer_text)
    except Exception as e:
        # without gpu
        print("Skipping", e)

shutil.rmtree("results/single")
