# Break-down CLI Inference
# 1. Encode reference audio: / 从语音生成 prompt:

# You should get a fake.npy file.

# 你应该能得到一个 fake.npy 文件.

# ## Enter the path to the audio file here
# src_audio = r"D:\PythonProject\vo_hutao_draw_appear.wav"

# !python tools/vqgan/inference.py \
#     -i {src_audio} \
#     --checkpoint-path "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"

# from IPython.display import Audio, display
# audio = Audio(filename="fake.wav")
# display(audio)

# 2. Generate semantic tokens from text: / 从文本生成语义 token:

#     This command will create a codes_N file in the working directory, where N is an integer starting from 0.

#     You may want to use --compile to fuse CUDA kernels for faster inference (~30 tokens/second -> ~300 tokens/second).

#     该命令会在工作目录下创建 codes_N 文件, 其中 N 是从 0 开始的整数.

#     您可以使用 --compile 来融合 cuda 内核以实现更快的推理 (~30 tokens/秒 -> ~300 tokens/秒)

# !python tools/llama/generate.py \
#     --text "hello world" \
#     --prompt-text "The text corresponding to reference audio" \
#     --prompt-tokens "fake.npy" \
#     --checkpoint-path "checkpoints/fish-speech-1.2-sft" \
#     --num-samples 2
#     # --compile

# 3. Generate speech from semantic tokens: / 从语义 token 生成人声:

# !python tools/vqgan/inference.py \
#     -i "codes_0.npy" \
#     --checkpoint-path "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"

# from IPython.display import Audio, display
# audio = Audio(filename="fake.wav")
# display(audio)

import sys
from pathlib import Path
import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

sys.path.append("fish-speech")

from tqdm import tqdm

from tools.vqgan.inference import main as encode_audio
from tools.llama.generate import main as generate_semantic_tokens


Path("results").mkdir(exist_ok=True)
for i in tqdm(range(1, 101)):
    # encode reference audio
    if Path(f"results/{i:03}.wav").exists():
        continue
    try:
        if not Path(f"../test/fish_{i:03}.wav").exists():
            encode_audio(
                input_path=Path(f"../test/{i:03}.wav"),
                output_path=Path(f"../test/fish_{i:03}.wav"),
                config_name="firefly_gan_vq",
                checkpoint_path="fish-speech/checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
                device="cuda",
            )
        # generate semantic tokens from text
        if not Path(f"../test/fish_{i:03}.npz").exists():
            ref_text = open(f"../test/{i:03}.txt").read()
            prompt_text = open(f"../test/{i:03}.orig.txt").read()
            print(ref_text, prompt_text)
            generate_semantic_tokens(
                text=ref_text,
                prompt_text=[prompt_text],
                prompt_tokens=[Path(f"../test/fish_{i:03}.npy")],
                num_samples=1,
                max_new_tokens=0,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                checkpoint_path="fish-speech/checkpoints/fish-speech-1.2-sft",
                device="cuda",
                compile=False,
                seed=42,
                half=False,
                iterative_prompt=True,
                chunk_length=100,
            )
        # generate speech from semantic tokens
        encode_audio(
            input_path=Path(f"codes_0.npy"),
            output_path=Path(f"results/{i:03}.wav"),
            config_name="firefly_gan_vq",
            checkpoint_path="fish-speech/checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
            device="cuda",
        )
        # write text to disk
        with open(f"results/{i:03}.txt", "w") as f:
            f.write(ref_text)
    except Exception as e:
        print(f"Failed to generate audio for {i:03}.txt: {e}, skipping")
