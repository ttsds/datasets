import os

import gradio as gr
import requests

g_systems = None


def synthesize(text, system, version, input_audio, input_text):
    system_attr = g_systems[system]
    result = requests.post(
        f"http://{system}:{system_attr['port']}/synthesize",
        data={
            "text": text,
            "version": version,
            "speaker_wav": input_audio,
            "speaker_txt": input_text,
        },
    )
    return result.content


def change_versions(system):
    versions = g_systems[system]["versions"]
    return gr.update(choices=versions, value=versions[0])


def gen_tts(input_text, system, version):
    pass


def start_gradio(systems):
    global g_systems
    app = gr.Blocks(title="TTS Interface")
    g_systems = systems
    with app:
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(lines=2, label="Text to synthesize")
                system = gr.Dropdown(
                    choices=list(g_systems.keys()), value="amphion", label="System"
                )
                version = gr.Dropdown(
                    choices=g_systems["amphion"]["versions"],
                    label="Version",
                    interactive=True,
                    value=g_systems["amphion"]["versions"][0],
                )
                input_audio = gr.File(
                    label="Reference audio", type="audio", accept=".wav"
                )
                input_text = gr.Textbox(lines=2, label="Reference text")
                system.change(change_versions, system, version)
                run_button = gr.Button("Generate Audio", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(
                    label="TTS generation", type="numpy", elem_id="audio_out"
                )

        inputs = [input_text, system, version, input_audio, input_text]
        outputs = [audio_out]
        run_button.click(fn=gen_tts, inputs=inputs, outputs=outputs, queue=True)

    app.queue()
    app.launch(server_name="0.0.0.0")
