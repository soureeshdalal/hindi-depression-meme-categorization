#!/usr/bin/env python3
"""
Setup and test script for Gemini Vision meme analysis
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages"""
    print("Installing Google GenAI SDK...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
        print("✅ Google GenAI SDK installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Google GenAI SDK: {e}")
        return False
    return True

def test_gemini_connection(api_key: str):
    """Test Gemini API connection"""
    try:
        from google import genai
        from google.genai import types
        
        print("Testing Gemini API connection...")
        
        client = genai.Client(api_key=api_key)
        
        # Simple text test with Gemini 3 Pro
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=['Say "Hello, I am working!" in exactly those words.']
        )
        
        print(f"✅ Gemini API test successful!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def check_dataset():
    """Check if dataset exists"""
    dataset_path = Path("../dataset")
    if dataset_path.exists():
        test_images = list(dataset_path.glob("**/*.jpg"))
        val_images = list(dataset_path.glob("**/*.png"))
        total_images = len(test_images) + len(val_images)
        
        print(f"✅ Dataset found: {total_images} images")
        print(f"   Test images: {len(list(dataset_path.glob('test/*.jpg')))}")
        print(f"   Validation images: {len(list(dataset_path.glob('validation/*.jpg')))}")
        return True
    else:
        print(f"❌ Dataset not found at {dataset_path}")
        return False

def main():
    print("=== Gemini Vision Meme Analyzer Setup ===\n")
    
    # Step 1: Install dependencies
    if not install_dependencies():
        return
    
    # Step 2: Check dataset
    if not check_dataset():
        print("Please ensure the dataset folder is in the parent directory")
        return
    
    # Step 3: Test API connection
    api_key = input("\nEnter your Gemini API key for testing: ").strip()
    if not api_key:
        print("Skipping API test (no key provided)")
    else:
        if test_gemini_connection(api_key):
            print("\n🎉 Setup complete! Ready to analyze memes.")
            print("\nNext steps:")
            print("1. Run: python gemini_meme_analyzer.py")
            print("2. Enter your API key when prompted")
            print("3. Choose sample size (recommend 20-50 for first test)")
        else:
            print("\n❌ Setup incomplete. Please check your API key.")

if __name__ == "__main__":
    main()