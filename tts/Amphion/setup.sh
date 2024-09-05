git clone https://github.com/open-mmlab/Amphion.git
cd Amphion && git checkout 211e1d4 && cd ..
git clone https://huggingface.co/amphion/valle_librilight_6k
cd valle_librilight_6k && git lfs pull && cd ..
git clone https://huggingface.co/amphion/naturalspeech2_libritts
cd naturalspeech2_libritts && git lfs pull && cd ..

mkdir -p Amphion/ckpts/tts/valle2 && huggingface-cli download amphion/valle SpeechTokenizer.pt config.json --local-dir Amphion/ckpts/tts/valle2
huggingface-cli download amphion/valle valle_ar_mls_196000.bin valle_nar_mls_164000.bin --local-dir Amphion/ckpts/tts/valle2

cd Amphion
# replace line 17 in egs/tts/VALLE/run.sh with ". ../../.venv/Amphion/bin/activate"
sed -i '18s/.*/. ..\/..\/.venv\/Amphion\/bin\/activate/' egs/tts/VALLE/run.sh
mkdir -p ckpts/tts
ln -s  ../../../valle_librilight_6k  ckpts/tts/valle1
ln -s  ../../../valle_libritts  ckpts/tts/valle1_libritts
ln -s  ../../../naturalspeech2_libritts  ckpts/tts/naturalspeech2_libritts
cd ..
pip install -r requirements.txt