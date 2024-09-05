from docker_manager import DockerManager
from gradio_interface import start_gradio
from time import sleep
import requests

if __name__ == "__main__":
    print("Starting orchestrator")
    
    try:
        manager = DockerManager()
        no_docker = False
    except Exception as e:
        no_docker = True

    systems = {
        k: {
            "versions": [],
            "port": p
        } for k, p in [
            ("amphion", 8001),
        ]
    }
    # get versions for each system
    for system in list(systems.keys()):
        if not no_docker:
            container = manager.start_container(system, port=systems[system]["port"])
            # check if the container is ready
        retries = 3
        port = systems[system]["port"]
        if no_docker:
            request_url = f"http://{system}:{port}/info"
        else:
            request_url = f"http://localhost:{port}/info"
        sleep(1)
        response = requests.get(request_url)
        systems[system] = response.json()
        systems[system]["port"] = port

    # Start Gradio interface
    start_gradio(systems)