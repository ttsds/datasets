git clone https://github.com/open-mmlab/Amphion.git
git clone https://huggingface.co/amphion/valle_librilight_6k
git clone https://huggingface.co/amphion/valle
git clone https://huggingface.co/amphion/naturalspeech2_libritts
cd Amphion
mkdir -p ckpts/tts
ln -s  ../../../valle_librilight_6k  ckpts/tts/valle1
ln -s  ../../../valle_libritts  ckpts/tts/valle1_libritts
ln -s  ../../../valle  ckpts/tts/valle2
ln -s  ../../../naturalspeech2_libritts  ckpts/tts/naturalspeech2_libritts