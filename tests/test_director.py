"""
Test script for Director class structure.
"""
import os
from director import Director

def test_director_init():
    print("Testing Director initialization...")
    
    # Test without API key (should warn but not crash)
    director = Director(api_key=None)
    print("Director initialized (no key).")
    
    # Test with dummy key
    director_with_key = Director(api_key="dummy_key")
    print("Director initialized (with key).")
    
    # Verify method existence
    assert hasattr(director, "analyze_chapter")
    print("Method analyze_chapter exists.")
    
    # Test OpenAI init
    print("Testing OpenAI initialization...")
    director_openai = Director(api_key="dummy", provider="openai")
    assert director_openai.provider == "openai"
    print("Director initialized (OpenAI).")
    
    print("SUCCESS: Director class structure verified.")

if __name__ == "__main__":
    test_director_init()
