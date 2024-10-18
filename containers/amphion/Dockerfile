ARG gpus=all
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y git git-lfs

WORKDIR /app
RUN git clone https://github.com/open-mmlab/Amphion.git
WORKDIR Amphion
RUN git checkout 211e1d4
WORKDIR /app
RUN git clone https://huggingface.co/amphion/valle_librilight_6k
WORKDIR valle_librilight_6k
RUN git lfs pull
WORKDIR /app
RUN git clone https://huggingface.co/amphion/naturalspeech2_libritts
WORKDIR naturalspeech2_libritts
RUN git lfs pull
WORKDIR /app
RUN git clone https://huggingface.co/amphion/valle
WORKDIR valle
RUN git lfs pull
WORKDIR /app

RUN python --version >> python_version.txt

WORKDIR Amphion
RUN sed -i '18s/.*/. ..\/..\/.venv\/Amphion\/bin\/activate/' egs/tts/VALLE/run.sh
RUN mkdir -p ckpts/tts
RUN ln -s  ../../../valle_librilight_6k  ckpts/tts/valle1
RUN ln -s  ../../../valle_libritts  ckpts/tts/valle1_libritts
RUN ln -s  ../../../naturalspeech2_libritts  ckpts/tts/naturalspeech2_libritts
RUN ln -s  ../../../valle  ckpts/tts/valle2

# copy requirements.txt from current local directory to /app
RUN apt-get update && apt-get install gcc build-essential g++-12 -y
RUN --mount=type=cache,target=/root/.cache/pip pip install fairseq

RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi[standard] uvicorn

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt

COPY download_encodec_nltk.py /app/download_encodec_nltk.py
RUN --mount=type=cache,target=/root/.cache/huggingface \
python /app/download_encodec_nltk.py

# espeak (for VALL-E)
RUN apt-get install -y espeak-ng

# amphion_service.py is the entrypoint
COPY amphion_service.py /app/amphion_service.py

# insert the following at the beginning of /app/Amphion/bins/tts/inference.py
# import sys\nsys.path.append('.')\n
RUN sed -i '1s/^/import sys\nsys.path.append(".")\n/' /app/Amphion/bins/tts/inference.py
RUN sed -i 's/self.args.text.replace(" ", "_", 100)/"output"/' /app/Amphion/models/tts/naturalspeech2/ns2_inference.py

RUN rm /app/Amphion/ckpts/tts/valle2/optimizer_valle_ar_mls_196000.bin

WORKDIR /app

ENV port=8000

CMD uvicorn amphion_service:app --host 0.0.0.0 --port $port --timeout-keep-alive 300