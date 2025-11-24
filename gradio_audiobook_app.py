"""
Audiobook Converter - Gradio UI

Web interface for the audiobook converter system.
Allows users to upload documents, manage voices, and generate audiobooks.
"""

import gradio as gr
import os
import threading
import time
from pathlib import Path
from audiobook_generator import AudiobookGenerator
from voice_manager import VoiceManager
from audiobook_utils import estimate_processing_time

# Initialize components
# The generator is now initialized within _run_job to handle dynamic parameters like use_llm_cleanup
voice_manager = VoiceManager()

class JobManager:
    """Manages background generation jobs."""
    def __init__(self):
        self.status = "IDLE"  # IDLE, RUNNING, COMPLETED, FAILED
        self.progress = 0.0
        self.message = "Ready"
        self.result_path = None
        self.error = None
        self._lock = threading.Lock()
        
    def start_job(self, file_path, voice_name, custom_voice_file, exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty, use_llm_cleanup=False, detect_sfx=False):
        """Start a generation job in a background thread."""
        with self._lock:
            if self.status == "RUNNING":
                return False, "Job already running"
            
            self.status = "RUNNING"
            self.progress = 0.0
            self.message = "Starting..."
            self.result_path = None
            self.error = None
            
        # Start thread
        thread = threading.Thread(
            target=self._run_job,
            args=(file_path, voice_name, custom_voice_file, exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty, use_llm_cleanup, detect_sfx)
        )
        thread.daemon = True
        thread.start()
        return True, "Job started"

    def _run_job(self, file_path, voice_name, custom_voice_file, exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty, use_llm_cleanup=False, detect_sfx=False):
        """Internal method to run the job."""
        output_path = "generated_audiobook.wav"
        
        try:
            # Initialize generator here to pass use_llm_cleanup dynamically
            generator = AudiobookGenerator(use_llm_cleanup=use_llm_cleanup)

            # Determine which voice to use
            voice_path = None
            if custom_voice_file:
                voice_path = custom_voice_file
            elif not voice_name:
                print("No voice selected, using default model voice.")
            
            # Progress callback wrapper
            def update_progress(p, msg):
                with self._lock:
                    self.progress = p
                    self.message = msg
            
            # Generate
            result_path = generator.generate_audiobook(
                input_path=file_path,
                output_path=output_path,
                voice_name=voice_name if not custom_voice_file else None,
                voice_path=voice_path,
                progress_callback=update_progress,
                detect_sfx=detect_sfx,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            with self._lock:
                self.status = "COMPLETED"
                self.progress = 1.0
                self.message = "Generation complete!"
                self.result_path = result_path
                
        except Exception as e:
            print(f"Job failed: {e}")
            with self._lock:
                self.status = "FAILED"
                self.error = str(e)
                self.message = f"Failed: {str(e)}"

    def get_status(self):
        """Get current job status."""
        with self._lock:
            return {
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "result": self.result_path,
                "error": self.error
            }

# Global job manager
job_manager = JobManager()

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
        # Create a parser just for analysis (no LLM cleanup needed here)
        from audiobook_utils import DocumentParser
        parser = DocumentParser(use_llm_cleanup=False)
        
        text, metadata = parser.parse_document(file.name)
        
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

def start_generation(file_path, voice_name, custom_voice_file, exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty, use_llm_cleanup, detect_sfx):
    """Trigger the background generation job."""
    if not file_path:
        raise gr.Error("Please upload and analyze a document first.")
    
    # Defensive check: if a voice_name is selected, verify it exists
    if voice_name and not custom_voice_file:
        available_voices = [v['name'] for v in voice_manager.list_voices()]
        if voice_name not in available_voices:
            raise gr.Error(
                f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}. "
                f"Try refreshing the voice list or re-uploading the voice."
            )
        
    success, msg = job_manager.start_job(
        file_path, voice_name, custom_voice_file, 
        exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty,
        use_llm_cleanup, detect_sfx
    )
    
    if not success:
        raise gr.Error(msg)
        
    return gr.update(active=True), "Job started..."

def check_progress():
    """Poll for progress updates."""
    status = job_manager.get_status()
    
    # Update progress bar
    # Note: Gradio's progress bar is tricky with polling, so we use a slider or just text for now
    # Ideally we'd use the proper Progress component but it's tied to the function call
    
    msg = f"Status: {status['status']} - {status['message']}"
    
    if status['status'] == "COMPLETED":
        return (
            gr.update(value=status['result'], visible=True), # Audio output
            msg,
            gr.update(active=False) # Stop timer
        )
    elif status['status'] == "FAILED":
        return (
            None,
            msg,
            gr.update(active=False) # Stop timer
        )
    else:
        return (
            None,
            msg,
            gr.update(active=True) # Keep timer running
        )

# Build UI
with gr.Blocks(title="Chatterbox Audiobook Converter") as demo:
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
            
            # Advanced Settings
            with gr.Accordion("Advanced Audio Settings", open=False):
                with gr.Row():
                    exaggeration = gr.Slider(0.25, 2.0, step=0.05, label="Exaggeration", value=0.5, info="Higher = more dramatic (can be unstable)")
                    cfg_weight = gr.Slider(0.0, 1.0, step=0.05, label="CFG/Pace", value=0.5, info="Classifier-free guidance weight")
                with gr.Row():
                    temperature = gr.Slider(0.05, 5.0, step=0.05, label="Temperature", value=0.8, info="Higher = more random/creative")
                    repetition_penalty = gr.Slider(1.0, 2.0, step=0.1, label="Repetition Penalty", value=1.2, info="Prevents repeating phrases")
                with gr.Row():
                    min_p = gr.Slider(0.0, 1.0, step=0.01, label="Min P", value=0.05, info="Lower bound for sampling")
                    top_p = gr.Slider(0.0, 1.0, step=0.01, label="Top P", value=1.0, info="Cumulative probability cutoff")


            # AI Text Processing (Tier 2)
            with gr.Accordion("🧠 AI Text Processing (Experimental)", open=False):
                gr.Markdown("""
                **Advanced text enhancement using local AI models:**
                - **AI Text Cleanup**: Uses Qwen2.5-1.5B to intelligently remove headers/footers and fix OCR errors
                - **Sound Effect Detection**: Analyzes text and suggests ambient sounds to enhance the audiobook
                
                ⚠️ **Note**: These features are slower but significantly improve quality. First use will download ~3GB model.
                """)
                
                use_llm_cleanup = gr.Checkbox(
                    label="Enable AI text cleanup",
                    value=False,
                    info="Slower but removes artifacts more intelligently"
                )
                
                detect_sfx = gr.Checkbox(
                    label="Detect and suggest sound effects",
                    value=False,
                    info="Saves suggestions to sfx_suggestions.json for review"
                )

            # Step 3: Generate
            gr.Markdown("### 3. Generate")
            generate_btn = gr.Button("🚀 Generate Audiobook", variant="primary", size="lg")
            
            # Output Area
            status_msg = gr.Textbox(label="Status", interactive=False, value="Ready")
            output_audio = gr.Audio(label="Generated Audiobook", type="filepath", visible=False)
            
            # Timer for polling (initially hidden/inactive)
            timer = gr.Timer(1.0, active=False)
            
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
            
            # Start generation
            generate_btn.click(
                start_generation,
                inputs=[
                    file_path_state, 
                    voice_dropdown, 
                    custom_voice,
                    exaggeration,
                    temperature,
                    cfg_weight,
                    min_p,
                    top_p,
                    repetition_penalty,
                    use_llm_cleanup,
                    detect_sfx
                ],
                outputs=[timer, status_msg] # Enable timer
            )
            
            # Poll for updates
            timer.tick(
                check_progress,
                outputs=[output_audio, status_msg, timer]
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
