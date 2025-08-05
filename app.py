# app.py

import sys
import os
import numpy as np
import librosa
from scipy.io import wavfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from agent import NewsReporterAgent, AgentState, add_messages
from langchain_core.messages import HumanMessage, AIMessage

#######################################
# Version check
import torch
import transformers
import gradio

# Now you can check the versions
print("--- 🔍 CHECKING LIBRARY VERSIONS 🔍 ---")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Gradio version: {gradio.__version__}")
print("------------------------------------")
#########################################


# --- 1. Initialize the Agent ---
# This loads the model once when the app starts.
agent = NewsReporterAgent()

# --- 2. Define Gradio Logic Handlers ---
# These functions orchestrate the agent's actions based on UI events.
def run_initial_generation(audio_path, image_path):
    """Handles the first step: processing inputs and generating the initial report."""
    if not audio_path and not image_path:
        return "Please provide an audio or image file.", None, gr.update(visible=False), None, None, None

    state = AgentState(audio_path=audio_path, 
                       image_path=image_path, 
                       news_report=[])
    
    state.update(agent.transcribe_audio(state))
    state.update(agent.describe_image(state))
    state.update(agent.create_report(state))

    latest_report = state["news_report"][-1].content
    transcribed_text = state.get('transcribed_text') or "No audio was provided to transcribe."
    image_description = state.get('image_description') or "No image was provided to describe."

    return latest_report, state, gr.update(visible=True), "", transcribed_text, image_description




def run_revision(feedback, current_state):
    """Handles the revision step based on user feedback."""
    if not feedback or not feedback.strip():
        # Re-populate UI fields if feedback is empty
        latest_report = next((msg.content for msg in reversed(current_state["news_report"]) if isinstance(msg, AIMessage)), "")
        transcribed_text = current_state.get('transcribed_text', "")
        image_description = current_state.get('image_description', "")
        return latest_report, current_state, "Please provide feedback.", transcribed_text, image_description

    current_state["news_report"] = add_messages(current_state["news_report"], [HumanMessage(content=feedback)])
    current_state.update(agent.revise_report(current_state))

    latest_report = current_state["news_report"][-1].content
    transcribed_text = current_state.get('transcribed_text') or "No audio was provided."
    image_description = current_state.get('image_description') or "No image was provided."

    return latest_report, current_state, "", transcribed_text, image_description

def run_save(current_state):
    """Handles the save step."""
    save_update = agent.save_report(current_state)
    return save_update["final_message"]

# --- 3. Define the Gradio UI ---
# ------------------------------------------------------------------
# Build examples: audio-only, image-only, and combined
# ------------------------------------------------------------------
# Define known file extensions
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
example_list = []

examples_dir = "examples"
if os.path.isdir(examples_dir):
    audio_files = sorted(
        f for f in os.listdir(examples_dir)
        if any(f.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)
    )
    image_files = sorted(
        f for f in os.listdir(examples_dir)
        if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    )

    # 1) audio-only
    for af in audio_files:
        example_list.append([os.path.join(examples_dir, af), None])

    # 2) image-only
    for imf in image_files:
        example_list.append([None, os.path.join(examples_dir, imf)])

    # 3) audio + image (pair first audio with first image, etc.)
    for af, imf in zip(audio_files, image_files):
        example_list.append([os.path.join(examples_dir, af),
                             os.path.join(examples_dir, imf)])



with gr.Blocks(theme=gr.themes.Soft(), title="Multimodal News Reporter") as demo:
    agent_state = gr.State(value=None)

    gr.Markdown("# 📰 Multimodal News Reporter AI")
    gr.Markdown("Upload an audio recording and/or a relevant image. The AI will generate a news report that you can then revise and save.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Audio Interview Evidence", type="filepath")
            image_input = gr.Image(label="Image Evidence", type="filepath")
            generate_btn = gr.Button("📝 Generate Initial Report", variant="primary")

            # Examples
            gr.Examples(
                examples=example_list,
                inputs=[audio_input, image_input],
                label="Click an example to test"
            )


        with gr.Column(scale=2):
            report_output = gr.Textbox(label="Generated News Report", lines=12, interactive=False)
            status_output = gr.Markdown(value="")
            
            with gr.Accordion("Show Source Information", open=False):
                transcribed_audio_output = gr.Textbox(label="🎤 Transcribed Audio", interactive=False, lines=5)
                image_description_output = gr.Textbox(label="🖼️ Image Description", interactive=False, lines=5)

            with gr.Group(visible=False) as revision_group:
                gr.Markdown("### ✍️ Provide Feedback for Revision")
                feedback_input = gr.Textbox(label="Your Feedback", placeholder="e.g., 'Make the tone more formal.'")
                with gr.Row():
                    revise_btn = gr.Button("🔄 Revise Report")
                    save_btn = gr.Button("💾 Save Final Report")

    # --- 4. Wire UI Components to Logic Handlers ---
    generate_btn.click(
        fn=run_initial_generation,
        inputs=[audio_input, image_input],
        outputs=[report_output, agent_state, revision_group, status_output, transcribed_audio_output, image_description_output]
    )
    revise_btn.click(
        fn=run_revision,
        inputs=[feedback_input, agent_state],
        outputs=[report_output, agent_state, status_output, transcribed_audio_output, image_description_output]
    ).then(fn=lambda: "", outputs=[feedback_input])
    save_btn.click(
        fn=run_save,
        inputs=[agent_state],
        outputs=[status_output]
    )

# --- 5. Launch the App ---
if __name__ == "__main__":
    demo.launch(debug=True)