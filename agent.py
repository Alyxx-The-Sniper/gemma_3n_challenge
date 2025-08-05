# /agent.py

import torch
import gc
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Optional, Sequence
from typing_extensions import TypedDict

# Helper function to mimic LangGraph's add_messages
def add_messages(left: Sequence[BaseMessage], right: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Concatenates two sequences of messages."""
    return left + right

class AgentState(TypedDict):
    """Defines the state of our agent."""
    audio_path: Optional[str]
    image_path: Optional[str]
    transcribed_text: Optional[str]
    image_description: Optional[str]
    news_report: Sequence[BaseMessage]
    final_message: Optional[str]


class NewsReporterAgent:
    def __init__(self):
        """Initializes the agent by loading the model and processor from the Hub."""
        print("--- 🚀 INITIALIZING MODEL (this may take a moment) ---")
        
        # Define the Hugging Face Hub model ID
        model_id = "google/gemma-3n-E2B-it" 
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  > Using device: {self.device}")
        
        # Load directly from the Hugging Face Hub
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype="auto"
        ).to(self.device)
        
        print("--- ✅ MODEL READY ---")

    def _generate(self, messages: list) -> str:
        """Private helper to run model inference and handle memory."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=self.model.dtype)

        outputs = self.model.generate(**inputs, max_new_tokens=1024, disable_compile=True)
        text = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

        del inputs
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        return text

    def transcribe_audio(self, state: AgentState) -> dict:
        """Transcribes the audio file specified in the state."""
        print("--- 🎤 TRANSCRIBING AUDIO ---")
        audio_path = state.get('audio_path')
        if not audio_path:
            return {}
        messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": "Transcribe the following audio. Provide only the transcribed text."}]}]
        transcribed_text = self._generate(messages)
        print("  > Transcription generated.")
        return {"transcribed_text": transcribed_text}

    def describe_image(self, state: AgentState) -> dict:
        """Generates a description for the image specified in the state."""
        print("--- 🖼️ DESCRIBING IMAGE ---")
        image_path = state.get('image_path')
        if not image_path:
            return {}
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": "Describe this image in detail."}]}]
        image_description = self._generate(messages)
        print("  > Description generated.")
        return {"image_description": image_description}

    # This is the agent
    def create_report(self, state: AgentState) -> dict:
        """Generates a news report from transcription and/or image description."""
        print("--- ✍️ GENERATING NEWS REPORT ---")
        context_parts = ["You are an expert news reporter. Your task is to write a clear, concise, and factual news report...", "Synthesize all available information into a single, coherent story..."]
        transcribed_text = state.get('transcribed_text')
        image_description = state.get('image_description')
        if not transcribed_text and not image_description:
            return {"news_report": [AIMessage(content="No input provided to generate a report.")]}
        if transcribed_text:
            context_parts.append(f"--- Transcribed Audio ---\n\"{transcribed_text}\"")
        if image_description:
            context_parts.append(f"--- Image Description ---\n\"{image_description}\"")
        prompt = "\n\n".join(context_parts)
        report_content = self._generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        print("  > Report generated successfully.")
        return {"news_report": [AIMessage(content=report_content)]}

    def revise_report(self, state: AgentState) -> dict:
        """Revises the news report based on the latest human feedback."""
        print("--- 🔄 REVISING REPORT ---")
        # Extract context from state
        transcribed = state.get("transcribed_text", "Not available.")
        image_desc = state.get("image_description", "Not available.")
        human_feedback = next((msg.content for msg in reversed(state["news_report"]) if isinstance(msg, HumanMessage)), None)
        last_ai_report = next((msg.content for msg in reversed(state["news_report"]) if isinstance(msg, AIMessage)), None)
        
        prompt = f"""You are a professional news editor. Revise the news report to address the feedback...
                    **Original Source Information:**
                    --- Transcribed Audio ---
                    "{transcribed}"
                    --- Image Description ---
                    "{image_desc}"
                    **Current Draft of News Report:**
                    "{last_ai_report}"
                    **Latest Human Feedback:**
                    "{human_feedback}"
                    Provide only the full, revised news report..."""
        revised_content = self._generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        print("  > Revision complete.")
        return {"news_report": add_messages(state["news_report"], [AIMessage(content=revised_content)])}

    def save_report(self, state: AgentState) -> dict:
        """Saves the latest AI-generated news report to a text file."""
        print("--- 💾 SAVING REPORT ---")
        latest_report_msg = next((msg for msg in reversed(state["news_report"]) if isinstance(msg, AIMessage)), None)
        if not latest_report_msg:
            return {"final_message": "Error: No report available to save."}
        output_dir = "saved_reports"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"news_report_{len(os.listdir(output_dir)) + 1}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(latest_report_msg.content)
        final_message = f"✅ Report saved to: **{filename}**"
        print(f"  > {final_message}")
        return {"final_message": final_message}