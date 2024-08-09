git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
huggingface-cli download fishaudio/fish-speech-1.2-sft --local-dir checkpoints/fish-speech-1.2-sft
# remove lines 586-614 in tools/llama/generate.py and 48-68 tools/vqgan/inference.py
# with the following:
sed -i '586,614d' tools/llama/generate.py
sed -i '48,68d' tools/vqgan/inference.py
# replace "fake_audio = fake_audios[0, 0].float().cpu().numpy()" in tools/vqgan/inference.py
# with "fake_audio = fake_audios[0, 0].detach().float().cpu().numpy()"
sed -i 's/    fake_audio = fake_audios[0, 0].float().cpu().numpy()/    fake_audio = fake_audios[0, 0].detach().float().cpu().numpy()/g' tools/vqgan/inference.py
pip install .