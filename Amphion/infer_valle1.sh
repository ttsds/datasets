
sh Amphion/egs/tts/VALLE/run.sh --stage 3 --gpu "0" \
    --infer_expt_dir ckpts/tts/valle1 \
    --infer_output_dir ../results \
    --infer_mode "single" \
    --infer_text "This is a clip of generated speech with the given text from a TTS model." \
    --infer_text_prompt "Out of this incident, the officious sheriff managed most ingeniously to create an embroilment with the town of Lawrence, Buckley, who was alleged to have been accessory to the crime, obtained a peace warrant against Branson, a neighbor of the victim." \
    --infer_audio_prompt ../../test/003.wav