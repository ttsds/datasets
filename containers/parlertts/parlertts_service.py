import os
import uuid
import tempfile

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoTokenizer, set_seed
from parler_tts import ParlerTTSForConditionalGeneration
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import soundfile as sf

app = FastAPI()

# Initialize the Parler-TTS model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup(model_id):
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    SAMPLING_RATE = model.config.sampling_rate
    return model, tokenizer, feature_extractor, SAMPLING_RATE

@app.post("/synthesize", response_class=FileResponse)
def synthesize(
    text: str = Form(...),
    version: str = Form(...),
    speaker_wav: UploadFile = File(...),
    speaker_txt: str = Form(...),
):
    if version not in ["Mini-v1", "Large-v1"]:
        raise ValueError("Invalid version")

    if version == "Mini-v1":
        model, tokenizer, feature_extractor, SAMPLING_RATE = setup("ylacombe/parler-tiny-v1") #setup("parler-tts/parler-tts-mini-v1")
    elif version == "Large-v1":
        model, tokenizer, feature_extractor, SAMPLING_RATE = setup("ylacombe/parler-tiny-v1") #setup("parler-tts/parler-tts-large-v1")

    raise NotImplementedError("This function is not implemented yet")

    # Create a directory to store results if it doesn't exist
    output_dir = "/results_parler"
    os.makedirs(output_dir, exist_ok=True)

    # Save the uploaded speaker reference audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(speaker_wav.file.read())
        init_audio_file = tmp.name

    init_prompt = speaker_txt
    prompt = text

    # Load and preprocess the initial audio
    init_audio, init_sr = torchaudio.load(init_audio_file)
    init_audio = torchaudio.functional.resample(init_audio, init_sr, SAMPLING_RATE)
    init_audio = init_audio.mean(0)  # Convert to mono if necessary

    # Encode the initial audio using the feature extractor
    input_values = feature_extractor(
        init_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_values.to(device)

    # Tokenize the description and prompts
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(
        init_prompt + " " + prompt, return_tensors="pt"
    ).input_ids.to(device)

    set_seed(2)  # For reproducibility

    # Generate the audio using the Parler-TTS model
    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        input_values=input_values,
    )

    # Save the generated audio to a unique file
    audio_arr = generation.cpu().numpy().squeeze()
    output_filename = f"output_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, output_filename)
    sf.write(output_path, audio_arr, SAMPLING_RATE)

    # Clean up the temporary file
    os.remove(init_audio_file)

    return FileResponse(output_path)

@app.get("/info")
def info():
    return {
        "versions": ["Mini-v1", "Large-v1"],
        "requires_text": [True],
    }

@app.get("/ready")
def ready():
    return "ready"
