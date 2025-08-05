# Multimodal News Reporter AI

This is a Gradio application that uses Google's `gemma-3n-e2b-it` model to function as a multimodal news reporter. The agent can take an audio file and/or a relevant image as input, transcribe the audio, describe the image, and synthesize the information into a cohesive news report. The user can then provide feedback to have the agent iteratively revise the report.

This application is designed to be deployed on platforms like Hugging Face Spaces.

## Features

-   **Multimodal Input**: Accepts both audio and image files.
-   **Audio Transcription**: Transcribes spoken content from audio files.
-   **Image Description**: Generates detailed descriptions of images.
-   **Report Generation**: Synthesizes all available information into a coherent news report.
-   **Iterative Revision**: Allows users to provide natural language feedback for report revision.
-   **Clickable Examples**: Includes sample audio and image files for quick testing and demonstration.
-   **Save Output**: Saves the final report to a local text file within the application environment.

## How to Use the Demo

1.  Launch the application.
2.  Under the input widgets, you will see a section labeled **"Click an example to test"**.
3.  Click on the example row. The sample audio and image will automatically load into the input boxes.
4.  Click the **"📝 Generate Initial Report"** button to see the AI in action.
5.  Once the report is generated, you can optionally provide feedback in the text box and click **"🔄 Revise Report"**.

## Project Structure

The repository is organized in a modular structure for clarity and maintainability.


/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py              # Main Gradio app entry point
├── examples/           # Contains sample files for the demo
│   ├── sample_audio.mp3
│   └── sample_image.jpg
├── init.py
└── agent.py        # Core agent logic, model loading, and nodes


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Alyxx-The-Sniper/AI_AGENT_REPORTER_DEMO_ONLY.git
    cd gemma_3n_challenge
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run Locally

To start the Gradio application, run the `app.py` file from the project's root directory:

```bash
python app.py

The application will start, load the model from the Hugging Face Hub (this may take a few moments on the first run), and provide a local URL (e.g., http://127.0.0.1:7860) that you can open in your browser.

HuggingFace example: 