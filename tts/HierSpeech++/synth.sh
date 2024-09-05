cd HierSpeechpp
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
    --input_prompt "../../test/001.wav" \
    --input_txt "../../test/001.txt" \
    --ckpt "../models/main/hierspeechpp_v1.1_ckpt.pth" \
    --ckpt_text2w2v "../models/ttv/ttv_lt960_ckpt.pth" \
    --output_dir "../results" \
    --noise_scale_vc "0.333" \
    --noise_scale_ttv "0.333" \
    --denoise_ratio "0"