"""
Audiobook Converter - Gradio UI

Web interface for the audiobook converter system.
Allows users to upload documents, manage voices, and generate audiobooks.
"""

import gradio as gr
import os
import threading
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
        # self.status = "IDLE"  # IDLE, RUNNING, COMPLETED, FAILED
        # self.progress = 0.0
        # self.message = "Ready"
        # self.result_path = None
        # self.error = None
        # self._lock = threading.Lock()
        self.jobs = {}  # Stores status for multiple jobs
        self.lock = threading.Lock()
        self.current_job_id = None # To track the currently active job for UI polling

    def start_job(self, file_path, voice_name, custom_voice_file, exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty, use_llm_cleanup=False, detect_sfx=False):
        """Start a new background job."""
        job_id = str(uuid.uuid4())
        
        # Create thread
        thread = threading.Thread(
            target=self._run_job,
            args=(file_path, voice_name, custom_voice_file, exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty, use_llm_cleanup, detect_sfx),
            kwargs={'job_id': job_id}
        )
        
        with self.lock:
            self.jobs[job_id] = {
                'status': 'starting',
                'progress': 0.0,
                'message': 'Initializing...',
                'result': None,
                'thread': thread,
                'start_time': time.time(),
                'error': None
            }
            self.current_job_id = job_id
            
        thread.start()
        return 0.0, "Starting job..."

    def _run_job(self, file_path, voice_name, custom_voice_file, exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty, use_llm_cleanup=False, detect_sfx=False, job_id=None):
        """Worker function."""
        try:
            # Initialize generator here to pass parameters dynamically
            generator = AudiobookGenerator()
            
            # Determine which voice to use
            voice_path = None
            if custom_voice_file:
                voice_path = custom_voice_file
            elif not voice_name:
                print("No voice selected, using default model voice.")
            
            def progress_callback(p, msg):
                with self.lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]['progress'] = p
                        self.jobs[job_id]['message'] = msg
            
            # Ensure output_file is defined, e.g., using a temporary file or a fixed name
            # For now, let's assume a fixed name as in the original generate_audiobook call
            output_file = Path("generated_audiobook.wav")

            output_path = generator.generate_audiobook(
                input_path=file_path,
                output_path=str(output_file),
                voice_name=voice_name if not custom_voice_file else None,
                voice_path=custom_voice_file,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_llm_cleanup=use_llm_cleanup, # Changed from self.use_llm_cleanup
                detect_sfx=detect_sfx,
                progress_callback=progress_callback
            )
            
            with self.lock: # Changed from self._lock to self.lock
                if job_id in self.jobs:
                    self.jobs[job_id]['status'] = "COMPLETED"
                    self.jobs[job_id]['progress'] = 1.0
                    self.jobs[job_id]['message'] = "Generation complete!"
                    self.jobs[job_id]['result'] = output_path # Changed from result_path
                
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
        parser = DocumentParser()
        
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
        
    progress_val, msg = job_manager.start_job(
        file_path, voice_name, custom_voice_file, 
        exaggeration, temperature, cfg_weight, min_p, top_p, repetition_penalty,
        detect_sfx
    )
    
    return gr.update(value=progress_val, visible=True), msg

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
        # Tab 0: Book Analysis (NEW)
        with gr.Tab("📚 Book Analysis"):
            gr.Markdown("""
            ### Analyze Full Book to Generate Character Bible
            Upload your complete book to extract character descriptions, visual style, and validate text quality.
            This creates a "Character Bible" for consistent image generation across all chapters.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    book_file_input = gr.File(
                        label="Upload Full Book (PDF, DOCX, or TXT)",
                        file_types=[".pdf", ".doc", ".docx", ".txt"]
                    )
                    book_title_input = gr.Textbox(
                        label="Book Title",
                        placeholder="e.g., The Dawn of Yangchen"
                    )
                    provider_radio = gr.Radio(
                        choices=["Anthropic", "OpenAI"],
                        value="Anthropic",
                        label="AI Provider"
                    )
                    api_key_input = gr.Textbox(
                        label="API Key (Optional if set in .env)",
                        type="password",
                        placeholder="Leave blank to use .env key"
                    )
                    analyze_book_btn = gr.Button("🔍 Analyze Book & Generate Character Bible", variant="primary")
                
                with gr.Column(scale=1):
                    book_analysis_status = gr.Textbox(
                        label="Status",
                        value="Ready",
                        interactive=False
                    )
                    character_bible_output = gr.JSON(
                        label="Character Bible Preview",
                        visible=False
                    )
                    extraction_report_output = gr.JSON(
                        label="Text Quality Report",
                        visible=False
                    )
                    download_bible_btn = gr.DownloadButton(
                        label="📥 Download Character Bible",
                        visible=False
                    )
            
            def analyze_full_book(book_file, book_title, provider, api_key):
                """Analyze full book to generate Character Bible."""
                if not book_file:
                    return (
                        "❌ Please upload a book file",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                
                if not provider:
                    return (
                        "❌ Please select an AI provider (Anthropic or OpenAI)",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                
                try:
                    from director import Director
                    from audiobook_utils import DocumentParser
                    
                    # Parse document
                    parser = DocumentParser()
                    book_text, metadata = parser.parse_document(book_file.name)
                    
                    # Map provider name to lowercase
                    provider_name = provider.lower()
                    
                    # Analyze with Director Mode 1
                    director = Director(
                        provider=provider_name,
                        api_key=api_key if api_key else None
                    )
                    result = director.analyze_full_book(book_text, book_title or "Untitled")
                    
                    # Save Character Bible
                    bible_path = Path("chapter_drafts/character_bible.json")
                    bible_path.parent.mkdir(exist_ok=True)
                    
                    import json
                    with open(bible_path, 'w') as f:
                        json.dump(result["character_bible"], f, indent=2)
                    
                    return (
                        f"✅ Analysis complete! Found {len(result['character_bible'].get('characters', []))} characters. Quality score: {result['extraction_report']['quality_score']}/100",
                        gr.update(value=result["character_bible"], visible=True),
                        gr.update(value=result["extraction_report"], visible=True),
                        gr.update(value=str(bible_path), visible=True)
                    )
                    
                except Exception as e:
                    return (
                        f"❌ Error: {str(e)}",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            
            analyze_book_btn.click(
                fn=analyze_full_book,
                inputs=[book_file_input, book_title_input, provider_radio, api_key_input],
                outputs=[book_analysis_status, character_bible_output, extraction_report_output, download_bible_btn]
            )
        
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
            
            gr.Markdown("### 3. The Director (AI Creative Lead)")
            with gr.Row():
                provider_radio = gr.Radio(
                    choices=["Anthropic", "OpenAI"],
                    value=None,
                    label="AI Provider"
                )
                api_key_input = gr.Textbox(label="API Key (Optional if set in env)", type="password")
                direct_btn = gr.Button("🎬 Direct Scene", variant="primary")
            
            director_output = gr.JSON(label="Direction Sheet (JSON)")
            
            # Helper functions for this tab
            def prepare_chapters_ui(file):
                if not file:
                    return gr.update(choices=[]), [], "Please upload a file first."
                
                try:
                    # Initialize generator just for this task
                    gen = AudiobookGenerator()
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

            def run_director(chapter_path, api_key, provider):
                print(f"DEBUG: run_director called with provider={provider}, chapter={chapter_path}")
                if not chapter_path:
                    return "Please select a chapter first."
                
                try:
                    gen = AudiobookGenerator()
                    json_path = gen.analyze_chapter_with_director(
                        chapter_path, 
                        api_key=api_key if api_key else None,
                        provider=provider
                    )
                    
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    return data
                except Exception as e:
                    return {"error": str(e)}

            # Events
            prepare_btn.click(
                prepare_chapters_ui,
                inputs=[review_file_input],
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
                inputs=[current_chapter_path, api_key_input, provider_radio],
                outputs=[director_output]
            )



        # Tab 3: Scene Manager (Visual Production)
        with gr.Tab("Scene Manager"):
            gr.Markdown("### 1. Load Direction Sheet")
            with gr.Row():
                with gr.Column():
                    scene_json_file = gr.File(label="Upload JSON File", file_types=[".json"])
                with gr.Column():
                    scene_json_text = gr.TextArea(label="Or Paste JSON Here", lines=5, placeholder='{"scenes": [...]}')
            
            load_scenes_btn = gr.Button("Load Scenes", variant="secondary")
            
            scene_editor_container = gr.Column(visible=False)
            with scene_editor_container:
                scenes_state = gr.State([])
                current_scene_idx = gr.State(0)
                current_json_path = gr.State()
                
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
                        download_json_btn = gr.DownloadButton("Download Updated JSON", label="Download JSON")

            # Helper functions
            def load_scenes(file, text_input):
                json_path = None

                if file:
                    json_path = file.name
                elif text_input:
                    # Save pasted text to a temp file
                    try:
                        import json
                        # Validate JSON first
                        data = json.loads(text_input)

                        # Save to a file so we can update it later
                        output_dir = Path("chapter_drafts")
                        output_dir.mkdir(exist_ok=True)
                        json_path = output_dir / "pasted_scenes.json"

                        with open(json_path, 'w') as f:
                            json.dump(data, f, indent=2)
                        json_path = str(json_path)
                    except Exception as e:
                        return [], 0, gr.update(choices=[]), "", "", "", None, f"Invalid JSON: {e}", gr.update(visible=False), None

                if not json_path:
                    return [], 0, gr.update(choices=[]), "", "", "", None, "Please upload a file or paste JSON.", gr.update(visible=False), None

                try:
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    scenes = data.get("scenes", [])
                    if not scenes:
                        return [], 0, gr.update(choices=[]), "", "", "", None, "No scenes found in JSON.", gr.update(visible=False), None

                    # Prepare dropdown choices
                    choices = [f"Scene {s.get('id')}: {s.get('text_segment')[:30]}..." for s in scenes]

                    # Load first scene
                    first_scene = scenes[0]

                    return (
                        scenes, # state
                        0, # idx
                        gr.update(choices=choices, value=choices[0]), # dropdown
                        first_scene.get("text_segment", ""),
                        f"Mood: {first_scene.get('mood')} | Pacing: {first_scene.get('pacing')}",
                        first_scene.get("visual_prompt", ""),
                        first_scene.get("image_path"), # Image
                        f"Loaded {len(scenes)} scenes from {os.path.basename(json_path)}",
                        gr.update(visible=True), # Show editor
                        json_path # Update current_json_path state
                    )
                except Exception as e:
                    return [], 0, gr.update(choices=[]), "", "", "", None, f"Error loading JSON: {e}", gr.update(visible=False), None

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

            def save_scene_image(img, idx, scenes, json_path):
                if not json_path:
                    return scenes, "No JSON file loaded."
                
                if img is None:
                    return scenes, "No image uploaded."
                
                # Update scene data
                scenes[idx]["image_path"] = img
                
                # Save back to JSON file
                import json
                try:
                    with open(json_path, 'r') as f:
                        full_data = json.load(f)
                    
                    full_data["scenes"] = scenes
                    
                    with open(json_path, 'w') as f:
                        json.dump(full_data, f, indent=2)
                    
                    return scenes, f"Saved image for Scene {idx+1}"
                except Exception as e:
                    return scenes, f"Error saving: {e}"

            # Events
            load_scenes_btn.click(
                load_scenes,
                inputs=[scene_json_file, scene_json_text],
                outputs=[scenes_state, current_scene_idx, scene_list, scene_text, scene_mood, scene_prompt, scene_image, scene_status, scene_editor_container, current_json_path]
            )
            
            scene_list.change(
                update_scene_view,
                inputs=[scene_list, scenes_state],
                outputs=[scene_text, scene_mood, scene_prompt, scene_image, current_scene_idx]
            )
            
            save_image_btn.click(
                save_scene_image,
                inputs=[scene_image, current_scene_idx, scenes_state, current_json_path],
                outputs=[scenes_state, scene_status]
            )
            
            # Download Logic
            download_json_btn.click(
                lambda path: path,
                inputs=[current_json_path],
                outputs=[download_json_btn]
            )
            
            # Copy Prompt Logic
            copy_prompt_btn.click(
                None,
                inputs=[scene_prompt],
                outputs=None,
                js="(text) => { navigator.clipboard.writeText(text); return text; }"
            )

        # Tab 4: Video Assembly (The Studio)
        with gr.Tab("Render Video"):
            gr.Markdown("### 4. Assemble Movie")
            gr.Markdown("Turn your Direction Sheet + Images into a final video.")
            
            with gr.Row():
                render_json_file = gr.File(label="Upload Completed Direction Sheet (JSON)", file_types=[".json"])
                voice_for_video = gr.Dropdown(label="Narrator Voice", choices=[v['name'] for v in voice_manager.list_voices()], value="af_bella")
            
            render_btn = gr.Button("🎥 Render Full Movie", variant="primary")
            video_output = gr.Video(label="Final Movie")
            render_status = gr.Textbox(label="Status", interactive=False)
            
            def render_movie(json_file, voice):
                if not json_file:
                    return None, "Please upload a JSON file first."
                
                try:
                    from video_editor import VideoEditor
                    editor = VideoEditor()
                    
                    # Update status
                    output_path = editor.assemble_video(json_file.name, voice_name=voice)
                    return output_path, "Render Complete!"
                except Exception as e:
                    return None, f"Error: {str(e)}"

            render_btn.click(
                render_movie,
                inputs=[render_json_file, voice_for_video],
                outputs=[video_output, render_status]
            )

        # Tab 5: Manage Voices
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
