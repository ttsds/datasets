from time import sleep
import os

from docker_manager import DockerManager
import requests

from interface import start_gradio
from tts_api import TTSApi

if __name__ == "__main__":
    print("Starting orchestrator")
    
    try:
        manager = DockerManager()
        use_docker = True
    except Exception as e:
        print(e)
        use_docker = False

    systems = {
        "amphion": 8001,
        "bark": 8002,
        "fish": 8003,
        "styletts2": 8004,
        "parlertts": 8005,
    }

    print("Starting TTS API")
    api = TTSApi(systems, use_docker, whisper_model=os.getenv("WHISPER_MODEL", "base.en"))

    # get versions for each system
    for system, port in systems.items():
        if use_docker:
            container = manager.start_container(system, port=port)

    # Start Gradio interface
    start_gradio(api)