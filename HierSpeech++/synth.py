import subprocess
from pathlib import Path
from tqdm import tqdm

def synthesize(input_prompt, input_txt, output_dir, ckpt, ckpt_text2w2v, noise_scale_vc, noise_scale_ttv, denoise_ratio):
    subprocess.run([
        'python3',
        'inference.py',
        '--input_prompt', input_prompt,
        '--input_txt', input_txt,
        '--ckpt', ckpt,
        '--ckpt_text2w2v', ckpt_text2w2v,
        '--output_dir', output_dir,
        '--noise_scale_vc', noise_scale_vc,
        '--noise_scale_ttv', noise_scale_ttv,
        '--denoise_ratio', denoise_ratio
    ], cwd=Path('HierSpeechpp'))

for i in tqdm(range(1, 101)):
    synthesize(
        f'../../test/{i:03d}.wav',
        f'../../test/{i:03d}.txt',
        f'../results',
        '../models/main/hierspeechpp_v1.1_ckpt.pth',
        '../models/ttv/ttv_lt960_ckpt.pth',
        '0.333',
        '0.333',
        '0'
    )
    # write transcript
    with open(f'results/{i:03d}.txt', 'w') as f:
        f.write(f'{i:03d}\n')