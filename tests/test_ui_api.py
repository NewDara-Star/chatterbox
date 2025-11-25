from gradio_client import Client, handle_file
import os
import time

def test_editor_workflow():
    print("Connecting to Gradio app...")
    client = Client("http://127.0.0.1:7861")
    
    # 1. Prepare Chapters
    print("\n1. Testing 'Clean & Split Chapters'...")
    # The API expects a file object. gradio_client.handle_file handles this.
    test_file = "/Users/Star/chatterbox/alice_test.docx"
    
    # predict(file, use_llm, api_name="/prepare_chapters_ui")
    result = client.predict(
        file=handle_file(test_file),
        use_llm=False,
        api_name="/prepare_chapters_ui"
    )
    
    # Result is (dropdown_update, status_msg)
    # The first element is the dropdown update dict
    dropdown_update = result[0]
    print(f"Split Result Choices: {dropdown_update['choices']}")
    
    # Pick the second chapter (Chapter 1) to test
    # choices is a list of [value, label] pairs or just strings
    first_choice = dropdown_update['choices'][1] 
    chapter_filename = first_choice[0] if isinstance(first_choice, list) else first_choice
    print(f"Selected Chapter: {chapter_filename}")
    
    # 2. Load Chapter Text
    print(f"\n2. Testing 'Load Chapter Text' for {chapter_filename}...")
    
    try:
        # We must pass the filename exactly as it appears in the choices
        text = client.predict(
            selected_filename=chapter_filename,
            api_name="/load_chapter_text"
        )
        print(f"Loaded Text (first 50 chars): {text[:50]}...")
        
        # 3. Save Chapter Text
        print("\n3. Testing 'Save Chapter Text'...")
        new_text = "[EDITED VIA API] " + text
        status = client.predict(
            text=new_text,
            api_name="/save_chapter_text"
        )
        print(f"Save Status: {status}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_editor_workflow()
