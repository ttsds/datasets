from tortoise import api
from tortoise import utils
import torchaudio

clips_paths = ["../test/001.wav"]
reference_clips = [utils.audio.load_audio(p, 22050) for p in clips_paths]
tts = api.TextToSpeech()
pcm_audio = tts.tts_with_preset("This is a test", voice_samples=reference_clips, preset='fast')
torchaudio.save("test.wav", pcm_audio, 22050)