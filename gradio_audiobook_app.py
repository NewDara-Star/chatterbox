"""
Audiobook Converter - Gradio UI

Web interface for the audiobook converter system.
Allows users to upload documents, manage voices, and generate audiobooks.
"""

import gradio as gr
import os
from pathlib import Path
from audiobook_generator import AudiobookGenerator
from voice_manager import VoiceManager
from audiobook_utils import estimate_processing_time

# Initialize components
generator = AudiobookGenerator()
voice_manager = VoiceManager()

def get_voice_choices():
    """Get list of available voices for dropdown."""
    voices = voice_manager.list_voices()
    choices = [v['name'] for v in voices]
    return choices

def update_voice_list():
    """Update the voice dropdown choices."""
    return gr.Dropdown(choices=get_voice_choices())

def save_new_voice(audio_file, name, description):
    """Save a new voice reference."""
    if not audio_file:
        return "Please upload an audio file first.", update_voice_list()
    
    if not name:
        return "Please provide a name for the voice.", update_voice_list()
    
    try:
        voice_manager.save_voice(audio_file, name, description)
        return f"Successfully saved voice: {name}", update_voice_list()
    except Exception as e:
        return f"Error saving voice: {str(e)}", update_voice_list()

def analyze_document(file):
    """Analyze uploaded document and provide estimates."""
    if not file:
        return "No file uploaded", "", ""
    
    try:
        text, metadata = generator.parser.parse_document(file.name)
        
        info = f"""
        **Document Analysis:**
        - Pages: {metadata['page_count']}
        - Words: {metadata['word_count']}
        - Characters: {metadata['char_count']}
        - Format: {metadata['format']}
        """
        
        mins, secs = estimate_processing_time(metadata['char_count'])
        estimate = f"⏱️ Estimated processing time: {mins}m {secs}s"
        
        return info, estimate, file.name
        
    except Exception as e:
        return f"Error analyzing document: {str(e)}", "", ""

def generate_audiobook_ui(file_path, voice_name, custom_voice_file, progress=gr.Progress()):
    """Generate audiobook from UI."""
    if not file_path:
        raise gr.Error("Please upload and analyze a document first.")
    
    output_path = "generated_audiobook.wav"
    
    try:
        # Determine which voice to use
        voice_path = None
        if custom_voice_file:
            voice_path = custom_voice_file
        elif not voice_name:
            gr.Info("No voice selected, using default model voice.")
        
        # Progress callback wrapper
        def update_progress(p, msg):
            progress(p, desc=msg)
        
        # Generate
        result_path = generator.generate_audiobook(
            input_path=file_path,
            output_path=output_path,
            voice_name=voice_name if not custom_voice_file else None,
            voice_path=voice_path,
            progress_callback=update_progress
        )
        
        return result_path, f"Audiobook generated successfully! Saved to {result_path}"
        
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# Build UI
with gr.Blocks(title="Chatterbox Audiobook Converter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Chatterbox Audiobook Converter")
    gr.Markdown("Convert PDF and DOCX files to audiobooks using your cloned voice.")
    
    with gr.Tabs():
        # Tab 1: Create Audiobook
        with gr.Tab("Create Audiobook"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Step 1: Document
                    gr.Markdown("### 1. Upload Document")
                    file_input = gr.File(
                        label="Upload PDF or DOCX",
                        file_types=[".pdf", ".doc", ".docx"]
                    )
                    analyze_btn = gr.Button("Analyze Document", variant="secondary")
                    doc_info = gr.Markdown()
                    time_estimate = gr.Markdown()
                    
                    # Hidden state for file path
                    file_path_state = gr.State()
                    
                with gr.Column(scale=1):
                    # Step 2: Voice
                    gr.Markdown("### 2. Select Voice")
                    
                    with gr.Group():
                        voice_dropdown = gr.Dropdown(
                            label="Saved Voices",
                            choices=get_voice_choices(),
                            value=None
                        )
                        refresh_btn = gr.Button("🔄 Refresh Voices", size="sm")
                    
                    gr.Markdown("--- OR ---")
                    
                    custom_voice = gr.Audio(
                        label="Upload One-time Voice Reference",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
            
            # Step 3: Generate
            gr.Markdown("### 3. Generate")
            generate_btn = gr.Button("🚀 Generate Audiobook", variant="primary", size="lg")
            
            # Output
            output_audio = gr.Audio(label="Generated Audiobook", type="filepath")
            status_msg = gr.Textbox(label="Status", interactive=False)
            
            # Events
            analyze_btn.click(
                analyze_document,
                inputs=[file_input],
                outputs=[doc_info, time_estimate, file_path_state]
            )
            
            refresh_btn.click(
                update_voice_list,
                outputs=[voice_dropdown]
            )
            
            generate_btn.click(
                generate_audiobook_ui,
                inputs=[file_path_state, voice_dropdown, custom_voice],
                outputs=[output_audio, status_msg]
            )
        
        # Tab 2: Manage Voices
        with gr.Tab("Manage Voices"):
            gr.Markdown("### Save New Voice")
            with gr.Row():
                with gr.Column():
                    new_voice_audio = gr.Audio(
                        label="Record or Upload Voice",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                with gr.Column():
                    new_voice_name = gr.Textbox(label="Voice Name (e.g., 'My Narrator Voice')")
                    new_voice_desc = gr.Textbox(label="Description (optional)")
                    save_voice_btn = gr.Button("Save Voice", variant="primary")
                    save_status = gr.Textbox(label="Status", interactive=False)
            
            save_voice_btn.click(
                save_new_voice,
                inputs=[new_voice_audio, new_voice_name, new_voice_desc],
                outputs=[save_status, voice_dropdown]  # Update dropdown in other tab too
            )
            
            gr.Markdown("### Saved Voices")
            # We could add a dataframe here to list voices, but for now the dropdown is enough

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="127.0.0.1", server_port=7861)
