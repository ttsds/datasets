import os
import io
import gradio as gr
import requests
import soundfile as sf

TTS_API = None
SAVE_FILE = "last_selection.txt"

# Function to save the last selected system and version to a file
def save_last_selection(system, version):
    with open(SAVE_FILE, "w") as f:
        f.write(f"{system},{version}")

# Function to read the last selected system and version from the file
def load_last_selection(systems):
    with open(SAVE_FILE, "r") as f:
        # check if empty
        data = f.read()
        if data:
            data = data.split(",")
            if len(data) == 2 and data[0] in systems and data[1] in systems[data[0]]["versions"]:
                return data[0], data[1]
    # If no valid data, return default values
    default_system = list(systems.keys())[0]
    default_version = systems[default_system]["versions"][0]
    return default_system, default_version

# Function to synthesize audio
def synthesize(text, system, version, input_audio, input_text=None):
    print(f"Synthesizing audio with {system} - {version}")
    save_last_selection(system, version)  # Save the selection before synthesizing
    return TTS_API.synthesize(text, system, version, input_audio, input_text)

# Function to update version dropdown when the system changes
def change_versions(system):
    versions = TTS_API.get_info(system)["versions"]
    return gr.update(choices=versions, value=versions[0])

# Function to start the Gradio app
def start_gradio(api):
    global TTS_API
    TTS_API = api
    systems = TTS_API.systems_info

    # Load the last saved system and version
    last_system, last_version = load_last_selection(systems)

    app = gr.Blocks(title="TTS Interface")
    with app:
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(lines=2, label="Text to synthesize")
                
                system = gr.Dropdown(
                    choices=list(systems.keys()), 
                    value=last_system,  # Set the last selected system
                    label="System"
                )
                
                version = gr.Dropdown(
                    choices=systems[last_system]["versions"],  # Set versions for the last selected system
                    label="Version",
                    interactive=True,
                    value=last_version,  # Set the last selected version
                )
                
                input_audio = gr.Audio(label="Reference audio", type="numpy", sources=["upload"])
                
                system.change(change_versions, system, version)

                # checkbox for trim silence
                trim_silence = gr.Checkbox(label="Trim silence", value=api.trim_silence)
                trim_silence.change(api.set_trim_silence, trim_silence)

                run_button = gr.Button("Generate Audio", variant="primary")
            
            with gr.Column():
                audio_out = gr.Audio(
                    label="TTS generation", type="numpy", elem_id="audio_out"
                )

        inputs = [text, system, version, input_audio]
        outputs = [audio_out]
        run_button.click(fn=synthesize, inputs=inputs, outputs=outputs, queue=True)

    app.queue()
    app.launch(server_name="0.0.0.0")

