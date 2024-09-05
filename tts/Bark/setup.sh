git clone https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer.git bark-vc
cd bark-vc && git checkout 4f42e44
pip install -r requirements.txt
pip install git+https://github.com/suno-ai/bark.git@f4f32d4cd480dfec1c245d258174bc9bde3c2148
cd ..
touch .setup_done