git clone https://github.com/sh-lee-prml/HierSpeechpp.git
cd HierSpeechpp && git checkout 318c633
sed -i 's/+cu117//g' requirements.txt
pip install -r requirements.txt
cd ..
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1xMfhg4qeehGO0RN-zxq-hAnW-omXmpdq/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1JTi3OOhIFFElj1X1u5jBeNa3CPbVS_gk/view?usp=drive_link
mv hierspeechpp_v1.1_ckpt.pth models/main/hierspeechpp_v1.1_ckpt.pth
mv ttv_lt960_ckpt.pth models/ttv/ttv_lt960_ckpt.pth
touch .setup_done