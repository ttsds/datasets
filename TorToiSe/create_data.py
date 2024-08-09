from tortoise import api
from tortoise import utils
import torchaudio
from pathlib import Path
from tqdm import tqdm
import tarfile
import gzip

Path("results").mkdir(exist_ok=True)
tts = api.TextToSpeech(kv_cache=True)

for item in tqdm(Path("../test").rglob("*.wav"), total=100):
    if Path(f"results/{item.name}").exists():
        continue
    reference_clips = [utils.audio.load_audio(item, 22050)]
    transcript = open(item.with_suffix(".txt")).read()
    pcm_audio = tts.tts_with_preset(transcript, voice_samples=reference_clips, preset='fast')
    torchaudio.save(f"results/{item.name}", pcm_audio[0], 24000)
    with open(f"results/{item.with_suffix('.txt').name}", "w") as f:
        f.write(transcript)

# create .tar.gz
with tarfile.open("results.tar.gz", "w:gz") as tar:
    for item in Path("results").rglob("*"):
        tar.add(item)