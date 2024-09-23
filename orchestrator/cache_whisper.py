import os

import whisper

whisper.load_model(os.getenv("WHISPER_MODEL", "small.en"))