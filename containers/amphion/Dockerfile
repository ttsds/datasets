ARG gpus=all
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y git git-lfs

WORKDIR /app
RUN git clone https://github.com/open-mmlab/Amphion.git
WORKDIR Amphion
RUN git checkout 211e1d4
WORKDIR ..
RUN git clone https://huggingface.co/amphion/valle_librilight_6k
WORKDIR valle_librilight_6k
RUN git lfs pull
WORKDIR ..
RUN git clone https://huggingface.co/amphion/naturalspeech2_libritts
WORKDIR naturalspeech2_libritts
RUN git lfs pull
WORKDIR ..

RUN python --version >> python_version.txt
# copy to host machine
RUN pip install -U "huggingface-hub"

RUN mkdir -p Amphion/ckpts/tts/valle2
RUN huggingface-cli download amphion/valle SpeechTokenizer.pt config.json --local-dir Amphion/ckpts/tts/valle2
RUN huggingface-cli download amphion/valle valle_ar_mls_196000.bin valle_nar_mls_164000.bin --local-dir Amphion/ckpts/tts/valle2

WORKDIR Amphion
RUN sed -i '18s/.*/. ..\/..\/.venv\/Amphion\/bin\/activate/' egs/tts/VALLE/run.sh
RUN mkdir -p ckpts/tts
RUN ln -s  ../../../valle_librilight_6k  ckpts/tts/valle1
RUN ln -s  ../../../valle_libritts  ckpts/tts/valle1_libritts
RUN ln -s  ../../../naturalspeech2_libritts  ckpts/tts/naturalspeech2_libritts

# copy requirements.txt from current local directory to /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN apt-get update && apt-get install gcc build-essential g++-12 -y
RUN pip install fairseq

RUN pip install fastapi[standard] uvicorn

# amphion_service.py is the entrypoint
COPY amphion_service.py /app/amphion_service.py

WORKDIR /app

ENV port=8000

# CMD ["uvicorn", "amphion_service:app", "--host", "0.0.0.0", "--port", $port]
CMD uvicorn amphion_service:app --host 0.0.0.0 --port $port