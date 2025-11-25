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
                    label="Generate and mix sound effects (Experimental)",
                    value=False,
                    info="Analyzes text for scene context and mixes in sound effects (rain, thunder, footsteps, etc.)"
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
        
        # Tab 2: Review Chapters (The Editor)
        with gr.Tab("Review Chapters"):
            gr.Markdown("### 1. Prepare Chapters")
            gr.Markdown("Clean text and split into chapters for review.")
            
            with gr.Row():
                review_file_input = gr.File(label="Upload Document", file_types=[".pdf", ".doc", ".docx"])
                with gr.Column():
                    review_use_llm = gr.Checkbox(label="Use AI Cleanup (Qwen)", value=False)
                    prepare_btn = gr.Button("Clean & Split Chapters", variant="primary")
            
            chapter_list_state = gr.State([])
            current_chapter_path = gr.State()
            
            gr.Markdown("### 2. Edit Chapters")
            with gr.Row():
                chapter_dropdown = gr.Dropdown(label="Select Chapter", choices=[], interactive=True)
                refresh_chapters_btn = gr.Button("🔄", size="sm")
            
            chapter_content = gr.TextArea(label="Chapter Text", lines=20, interactive=True)
            save_chapter_btn = gr.Button("Save Changes", variant="secondary")
            save_status_msg = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 3. The Director (Claude)")
            with gr.Row():
                api_key_input = gr.Textbox(label="Anthropic API Key (Optional if set in env)", type="password")
                direct_btn = gr.Button("🎬 Direct Scene", variant="primary")
            
            director_output = gr.JSON(label="Direction Sheet (JSON)")
            
            # Helper functions for this tab
            def prepare_chapters_ui(file, use_llm):
                if not file:
                    return gr.update(choices=[]), [], "Please upload a file first."
                
                try:
                    # Initialize generator just for this task
                    gen = AudiobookGenerator(use_llm_cleanup=use_llm)
                    output_dir = "chapter_drafts"
                    files = gen.prepare_chapters(file.name, output_dir)
                    
                    # Get basenames for dropdown
                    choices = [os.path.basename(f) for f in files]
                    return gr.update(choices=choices, value=choices[0] if choices else None), files, f"Generated {len(files)} chapters."
                except Exception as e:
                    return gr.update(choices=[]), [], f"Error: {str(e)}"

            def load_chapter_text(selected_filename, all_files):
                if not selected_filename or not all_files:
                    return "", None
                
                # Find full path
                full_path = next((f for f in all_files if os.path.basename(f) == selected_filename), None)
                if not full_path:
                    return "Error: File not found", None
                
                with open(full_path, 'r') as f:
                    text = f.read()
                return text, full_path

            def save_chapter_text(text, full_path):
                if not full_path:
                    return "No file selected."
                try:
                    with open(full_path, 'w') as f:
                        f.write(text)
                    return f"Saved {os.path.basename(full_path)}"
                except Exception as e:
                    return f"Error saving: {e}"

            def run_director(chapter_path, api_key):
                if not chapter_path:
                    return "Please select a chapter first."
                
                try:
                    gen = AudiobookGenerator()
                    json_path = gen.analyze_chapter_with_director(chapter_path, api_key=api_key if api_key else None)
                    
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    return data
                except Exception as e:
                    return {"error": str(e)}

            # Events
            prepare_btn.click(
                prepare_chapters_ui,
                inputs=[review_file_input, review_use_llm],
                outputs=[chapter_dropdown, chapter_list_state, save_status_msg]
            )
            
            chapter_dropdown.change(
                load_chapter_text,
                inputs=[chapter_dropdown, chapter_list_state],
                outputs=[chapter_content, current_chapter_path]
            )
            
            save_chapter_btn.click(
                save_chapter_text,
                inputs=[chapter_content, current_chapter_path],
                outputs=[save_status_msg]
            )
            
            direct_btn.click(
                run_director,
                inputs=[current_chapter_path, api_key_input],
                outputs=[director_output]
            )

            direct_btn.click(
                run_director,
                inputs=[current_chapter_path, api_key_input],
                outputs=[director_output]
            )

        # Tab 3: Scene Manager (Visual Production)
        with gr.Tab("Scene Manager"):
            gr.Markdown("### 1. Load Direction Sheet")
            with gr.Row():
                scene_json_file = gr.File(label="Upload Direction Sheet (JSON)", file_types=[".json"])
                load_scenes_btn = gr.Button("Load Scenes", variant="secondary")
            
            scenes_state = gr.State([])
            current_scene_idx = gr.State(0)
            
            gr.Markdown("### 2. Production Board")
            with gr.Row():
                with gr.Column(scale=1):
                    scene_list = gr.Dropdown(label="Select Scene", choices=[], interactive=True)
                    scene_text = gr.Textbox(label="Scene Text", lines=3, interactive=False)
                    scene_mood = gr.Textbox(label="Mood & Pacing", interactive=False)
                    scene_prompt = gr.TextArea(label="Visual Prompt (Copy this!)", lines=3, interactive=False)
                    copy_prompt_btn = gr.Button("📋 Copy Prompt")
                
                with gr.Column(scale=1):
                    scene_image = gr.Image(label="Scene Image", type="filepath", sources=["upload"], interactive=True)
                    save_image_btn = gr.Button("Save Image to Scene", variant="primary")
                    scene_status = gr.Textbox(label="Status", interactive=False)
            
            # Helper functions
            def load_scenes(file):
                if not file:
                    return [], gr.update(choices=[]), "Please upload a JSON file."
                
                try:
                    import json
                    with open(file.name, 'r') as f:
                        data = json.load(f)
                    
                    scenes = data.get("scenes", [])
                    choices = [f"Scene {s.get('id', i+1)}: {s.get('text_segment', '')[:30]}..." for i, s in enumerate(scenes)]
                    return scenes, gr.update(choices=choices, value=choices[0] if choices else None), f"Loaded {len(scenes)} scenes."
                except Exception as e:
                    return [], gr.update(choices=[]), f"Error: {e}"

            def select_scene(evt: gr.SelectData, scenes):
                # This might be tricky with dropdown, let's use index
                # For now, just use the value string to find index
                # Actually, dropdown change event passes the value
                pass

            def update_scene_view(selected_label, scenes):
                if not selected_label or not scenes:
                    return "", "", "", None, 0
                
                # Find index
                idx = 0
                for i, s in enumerate(scenes):
                    label = f"Scene {s.get('id', i+1)}: {s.get('text_segment', '')[:30]}..."
                    if label == selected_label:
                        idx = i
                        break
                
                scene = scenes[idx]
                text = scene.get("text_segment", "")
                mood = f"Mood: {scene.get('mood', 'N/A')} | Pacing: {scene.get('pacing', 'N/A')}"
                prompt = scene.get("visual_prompt", "")
                img_path = scene.get("image_path", None)
                
                return text, mood, prompt, img_path, idx

            def save_scene_image(img, idx, scenes, json_file):
                if not json_file:
                    return scenes, "No JSON file loaded."
                
                if img is None:
                    return scenes, "No image uploaded."
                
                # Update scene data
                scenes[idx]["image_path"] = img
                
                # Save back to JSON file
                # We need to read the full original JSON to preserve other fields
                import json
                with open(json_file.name, 'r') as f:
                    full_data = json.load(f)
                
                full_data["scenes"] = scenes
                
                with open(json_file.name, 'w') as f:
                    json.dump(full_data, f, indent=2)
                
                return scenes, f"Saved image for Scene {idx+1}"

            # Events
            load_scenes_btn.click(
                load_scenes,
                inputs=[scene_json_file],
                outputs=[scenes_state, scene_list, scene_status]
            )
            
            scene_list.change(
                update_scene_view,
                inputs=[scene_list, scenes_state],
                outputs=[scene_text, scene_mood, scene_prompt, scene_image, current_scene_idx]
            )
            
            save_image_btn.click(
                save_scene_image,
                inputs=[scene_image, current_scene_idx, scenes_state, scene_json_file],
                outputs=[scenes_state, scene_status]
            )

        # Tab 4: Manage Voices
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
