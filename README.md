# Multimodal News Reporter AI

This is a Gradio application that uses Google's Gemma 3n model to function as a multimodal news reporter. The agent can take an audio file and/or an image as input, transcribe the audio, describe the image, and synthesize the information into a news report. The user can then provide feedback to have the agent revise the report.

## Features

-   **Multimodal Input**: Accepts audio and image files.
-   **Audio Transcription**: Transcribes spoken content from audio files.
-   **Image Description**: Generates detailed descriptions of images.
-   **Report Generation**: Synthesizes information into a coherent news report.
-   **Iterative Revision**: Allows users to provide feedback for report revision.
-   **Save Output**: Saves the final report to a local text file.

## Project Structure

```
/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py              # Main Gradio app entry point
└── src/
    ├── __init__.py
    └── agent.py        # Core agent logic and model
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd multimodal-reporter
    ```

2.  **Set up Kaggle API (Required for model download):**
    -   Go to your Kaggle account, select "Account" from the profile menu.
    -   Click "Create New Token" to download `kaggle.json`.
    -   Place the `kaggle.json` file in `~/.kaggle/` on Linux/macOS or `C:\Users\<Your-Username>\.kaggle\` on Windows.

3.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To start the Gradio application, run the `app.py` file from the root directory:

```bash
python app.py
```

The application will start, load the model (this may take a few moments), and provide a local URL (e.g., `http://127.0.0.1:7860`) that you can open in your browser.