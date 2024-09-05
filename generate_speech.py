from subprocess import run
from pathlib import Path
import tempfile
import os
import logging

import whisper
import librosa
import yaml
import Levenshtein

WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small.en')

logger = logging.getLogger(__name__)

class TTS:
    def __init__(self, config_file, sampling_rate=22050, trim_silence=True):
        self.config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        self.sampling_rate = sampling_rate
        self.name = self.config['name']
        self.whisper = None
        self.root = str((Path(__file__).resolve().parent / Path('tts')).resolve())
        self.python = f'.venv/{self.config["venv"]}/bin/python'
        self.trim_silence = trim_silence

    def _get_reference_text(self, reference_audio, reference_text):
        if reference_text is None:
            logger.info(f'Recognizing text from reference audio using whisper model {WHISPER_MODEL}')
            if self.whisper is None:
                self.whisper = whisper.load_model(WHISPER_MODEL)
            reference_text = self.whisper.transcribe(reference_audio)["text"]
        return reference_text
        
    def generate_speech(self, text, reference_audio, reference_text=None):
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            reference_audio = Path(reference_audio).resolve()
            reference_text = self._get_reference_text(reference_audio, reference_text)
            temp_file = Path(temp_file.name).resolve()
            print(self.config['command'])
            command = self.config['command'].format(text=text, reference_audio=reference_audio, reference_text=reference_text, output=temp_file)
            command = command.replace('python ', self.python + ' ')
            print(command)
            run(command, shell=True, check=True, cwd=self.root)
            audio, sr = librosa.load(temp_file)
            if sr != self.sampling_rate:
                audio = librosa.resample(audio, sr, self.sampling_rate)
            # trim silence
            if self.trim_silence:
                audio, _ = librosa.effects.trim(audio)
            return audio
        
    def generate_speech_batch(self, texts, reference_audios, reference_texts=None):
        if 'batch_command' in self.config:
            # generate .csv file with text, reference_audio, reference_text
            with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
                for text, reference_audio, reference_text in zip(texts, reference_audios, reference_texts):
                    reference_text = self._get_reference_text(reference_audio, reference_text)
                    reference_audio = Path(reference_audio).resolve()
                    temp_file.write(f'{text},{reference_audio},{reference_text}\n')
                temp_file.flush()
                # create temporary directory to store generated audio files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir).resolve()
                    command = self.config['batch_command'].format(input_file=temp_file, output_directory=temp_dir)
                    command = command.replace('python ', self.python + ' ')
                    run(command, shell=True, check=True, cwd=self.root)
                    audio_files = sorted([f for f in Path(temp_dir).iterdir() if f.suffix == '.wav'])
                    audios = [librosa.load(f)[0] for f in audio_files]
                    return audios
        else:
            logger.info(f'Batch generation not supported for {self.name}. Generating audio files one by one.')
            results = [self.generate_speech(text, Path(reference_audio).resolve(), reference_text) for text, reference_audio, reference_text in zip(texts, reference_audios, reference_texts)]
            return results


# populate list of systems
TTS_SYSTEMS = {}

# get path of this python file
for system in (Path(__file__).resolve().parent / Path('tts')).iterdir():
    if system.is_dir():
        for config_file in system.glob('*.yaml'):
            tts = TTS(config_file)
            TTS_SYSTEMS[tts.name] = tts

# generate speech
def generate_speech(system, text, reference_audio, reference_text=None):
    if system not in TTS_SYSTEMS:
        systems = TTS_SYSTEMS.keys()
        dists = {s: Levenshtein.distance(system, s) for s in systems}
        closest = min(dists, key=dists.get)
        if dists[closest] < 5:
            raise ValueError(f'System {system} not found. Did you mean {closest}?')
        else:
            raise ValueError(f"System {system} not found. Available systems: {','.join(systems)}")
    return TTS_SYSTEMS[system].generate_speech(text, reference_audio, reference_text)

def generate_speech_batch(system, texts, reference_audios, reference_texts=None):
    return TTS_SYSTEMS[system].generate_speech_batch(texts, reference_audios, reference_texts)