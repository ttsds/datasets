sudo apt install ffmpeg
pip install -r requirements.txt

if [ $1 == "naturalspeech2" ]; then
    docker stop $(docker ps -a -q)
    cd containers/amphion && docker build -t amphion . && docker run -p 8000:8000 amphion &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/amphion_ns2/librittsr --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/amphion_ns2/emilia --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/amphion_ns2/librilatest --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/amphion_ns2/myst --tts_system amphion --tts_version "NaturalSpeech 2"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/amphion_ns2/torgo --tts_system amphion --tts_version "NaturalSpeech 2"
elif [ $1 == "valle" ]; then
    docker stop $(docker ps -a -q)
    cd containers/amphion && docker build -t amphion . && docker run -p 8000:8000 amphion &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/amphion_valle/librittsr --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/amphion_valle/emilia --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/amphion_valle/librilatest --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/amphion_valle/myst --tts_system amphion --tts_version "VALL-E v1"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/amphion_valle/torgo --tts_system amphion --tts_version "VALL-E v1"
elif [ $1 == "bark" ]; then
    docker stop $(docker ps -a -q)
    cd containers/bark && docker build -t bark . && docker run -p 8000:8000 bark &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/bark/librittsr --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/bark/emilia --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/bark/librilatest --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/bark/myst --tts_system bark --tts_version "Bark"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/bark/torgo --tts_system bark --tts_version "Bark"
elif [ $1 == "fish" ]; then
    docker stop $(docker ps -a -q)
    cd containers/fish && docker build -t fish . && docker run -p 8000:8000 fish &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/fish/librittsr --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/fish/emilia --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/fish/librilatest --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/fish/myst --tts_system fish --tts_version "Fish"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/fish/torgo --tts_system fish --tts_version "Fish"
elif [ $1 == "gpt-sovits" ]; then
    docker stop $(docker ps -a -q)
    cd containers/gpt-sovits && docker build -t gpt-sovits . && docker run -p 8000:8000 gpt-sovits &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/gpt-sovits/librittsr --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/gpt-sovits/emilia --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/gpt-sovits/librilatest --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/gpt-sovits/myst --tts_system gpt-sovits --tts_version "GPT-SoVITS"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/gpt-sovits/torgo --tts_system gpt-sovits --tts_version "GPT-SoVITS"
elif [ $1 == "hierspeechpp" ]; then
    docker stop $(docker ps -a -q)
    cd containers/hierspeechpp && docker build -t hierspeechpp . && docker run -p 8000:8000 hierspeechpp &
    python generate_data.py --source_audio_dir ../v2-evaluation/librittsr/ --output_dir ../v2-evaluation/tts/hierspeechpp/librittsr --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/emilia/ --output_dir ../v2-evaluation/tts/hierspeechpp/emilia --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/librilatest/ --output_dir ../v2-evaluation/tts/hierspeechpp/librilatest --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/myst/ --output_dir ../v2-evaluation/tts/hierspeechpp/myst --tts_system hierspeechpp --tts_version "v1.1"
    python generate_data.py --source_audio_dir ../v2-evaluation/torgo/ --output_dir ../v2-evaluation/tts/hierspeechpp/torgo --tts_system hierspeechpp --tts_version "v1.1"
fi