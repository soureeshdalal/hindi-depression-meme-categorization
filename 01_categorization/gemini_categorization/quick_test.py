#!/usr/bin/env python3
"""
Quick test of Gemini Vision with a single meme
"""

import os
import json
from pathlib import Path
from google import genai
from google.genai import types

def test_single_meme(api_key: str, image_path: Path):
    """Test analysis of a single meme"""
    
    # Setup client
    client = genai.Client(api_key=api_key)
    
    # Analysis prompt
    prompt = """
Analyze this mental health meme for cultural adaptation from English to Hindi for an Indian audience with ZERO Western cultural knowledge.

Extract:
1. All visible text
2. Who is in the image (celebrity name or "generic person")
3. Cultural elements that might be barriers
4. Rate cultural dependency 0-100%
5. Suggest adaptation strategy

Respond in JSON format:
{
    "extracted_text": "text from meme",
    "person_identified": "name or generic",
    "cultural_barriers": ["list of barriers"],
    "dependency_score": 0-100,
    "adaptation_needed": "strategy"
}
"""
    
    try:
        print(f"Analyzing: {image_path.name}")
        
        # Load image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Create image part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg'
        )
        
        # Call Gemini 3 Pro
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[image_part, prompt]
        )
        
        print("Raw response:")
        print(response.text)
        print("\n" + "="*50 + "\n")
        
        # Try to parse JSON
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        try:
            result = json.loads(response_text)
            print("Parsed JSON:")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError:
            print("Could not parse as JSON, but got response!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Get API key
    api_key = input("Enter Gemini API key: ").strip()
    if not api_key:
        print("API key required")
        return
    
    # Find a test image
    dataset_path = Path("../dataset")
    test_images = list(dataset_path.glob("**/*.jpg"))[:5]  # First 5 images
    
    if not test_images:
        print("No test images found")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test first image
    success = test_single_meme(api_key, test_images[0])
    
    if success:
        print("\n✅ Test successful! Gemini Vision is working.")
        print("Ready to run full analysis with gemini_meme_analyzer.py")
    else:
        print("\n❌ Test failed. Check API key and connection.")

if __name__ == "__main__":
    main()