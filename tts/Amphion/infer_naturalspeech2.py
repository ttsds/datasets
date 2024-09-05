import sys
import os
from pathlib import Path
from tqdm import tqdm
import shutil

os.chdir("Amphion")

Path("../results_naturalspeech2").mkdir(exist_ok=True)

envs = {
    "WORK_DIR": ".",
    "CUDA_VISIBLE_DEVICES": "0",
}

os.environ.update(envs)

for i in tqdm(range(1, 101)):
    print(
        Path(f"../results_naturalspeech2/{i:03}.wav").exists(),
        Path(f"../results_naturalspeech2/{i:03}.wav"),
    )
    if Path(f"../results_naturalspeech2/{i:03}.wav").exists():
        continue
    item = Path(f"../../test/{i:03}.wav")
    audio_prompt = item
    infer_text = item.with_suffix(".txt").read_text()
    try:
        sys.argv = [
            "bins/tts/inference.py",
            "--mode=single",
            "--config=egs/tts/NaturalSpeech2/exp_config.json",
            "--text=" + str(infer_text),
            "--checkpoint_path=ckpts/tts/naturalspeech2_libritts/checkpoint/epoch-0089_step-0512912_loss-6.367693",
            "--ref_audio=" + str(audio_prompt),
            "--output_dir=../results_naturalspeech2/single",
        ]
        script = Path("bins/tts/inference.py").read_text()
        script = "import sys\nsys.path.append('.')\n" + script
        exec(script)
        # find the output file
        out_path = next(Path("../results_naturalspeech2/single").glob("*.wav"))
        shutil.move(
            out_path,
            f"../results_naturalspeech2/{i:03}.wav",
        )
        with open(f"../results_naturalspeech2/{i:03}.txt", "w") as f:
            f.write(infer_text)
    except Exception as e:
        # without gpu
        print("Skipping", e)

shutil.rmtree("../results_naturalspeech2/single")
