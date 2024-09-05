import docker
from random import randint

class DockerManager:
    def __init__(self):
        self.client = docker.from_env()
        # stop all containers
        for container in self.client.containers.list():
            container.stop()

    def start_container(self, image_name: str, port: int = None):
        # check if container already exists
        container = self.client.containers.run(f"{image_name}:latest", ports={8000: port}, detach=True)
        self.image2container[image_name] = container.name
        self.container2port[image_name] = port
        return container

    def get_container_port(self, container_name: str):
        return self.container2port[container_name]