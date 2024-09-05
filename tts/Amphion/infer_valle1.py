from pathlib import Path
import subprocess
import shutil
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("reference_audio", type=Path)
    parser.add_argument("reference_text", type=Path)
    parser.add_argument("output", type=Path)

    args = parser.parse_args()

    Path.mkdir(Path("results"), exist_ok=True)

    text = args.text
    reference_audio = args.reference_audio
    reference_text = args.reference_text
    output = args.output
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
            text,
            "--infer_text_prompt",
            reference_text,
            "--infer_audio_prompt",
            str(reference_audio),
        ],
        check=True,
        cwd="Amphion",
    )
    # print current directory
    shutil.move("Amphion/results/single/test_pred.wav", output)
    shutil.rmtree("Amphion/results/single")