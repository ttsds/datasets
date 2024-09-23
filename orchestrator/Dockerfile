FROM python:3.9

ARG GRADIO_SERVER_PORT=7860
ARG WHISPER_MODEL=base.en
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}
ENV WHISPER_MODEL=${WHISPER_MODEL}

WORKDIR /app

ADD requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

ADD cache_whisper.py /app/cache_whisper.py

RUN --mount=type=cache,target=/root/.cache/huggingface python cache_whisper.py

COPY . /app

CMD ["python", "main.py"]