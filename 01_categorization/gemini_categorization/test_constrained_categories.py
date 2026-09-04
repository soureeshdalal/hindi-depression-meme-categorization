#!/usr/bin/env python3
"""
Test the new constrained 6-category prompt with a small sample
"""

import os
import sys
sys.path.append('.')
from gemini_meme_analyzer import GeminiMemeAnalyzer
from pathlib import Path

def test_constrained_categories():
    """Test the new prompt with 10 sample memes"""
    
    # Get API key
    api_key = input("Enter Gemini API key: ").strip()
    if not api_key:
        print("API key required")
        return
    
    # Initialize analyzer with new constrained prompt
    analyzer = GeminiMemeAnalyzer(api_key)
    
    # Get sample images
    dataset_dir = Path("../dataset")
    test_images = sorted(list(dataset_dir.glob("**/*.jpg")))[:10]  # First 10 images
    
    print(f"Testing constrained categories with {len(test_images)} sample memes...")
    
    results = []
    categories_found = set()
    
    for i, image_path in enumerate(test_images, 1):
        try:
            print(f"\nAnalyzing {i}/10: {image_path.name}")
            result = analyzer.analyze_meme_with_gemini(image_path)
            results.append(result)
            
            category = result.get('discovered_category', 'Unknown')
            categories_found.add(category)
            
            print(f"  Category: {category}")
            print(f"  Dependency: {result.get('cultural_dependency_score', 0)}%")
            print(f"  Difficulty: {result.get('adaptation_difficulty', 'Unknown')}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Successfully analyzed: {len(results)}/10 memes")
    print(f"Categories discovered: {len(categories_found)}")
    print(f"Categories found: {sorted(categories_found)}")
    
    # Check if all categories are from our predefined set
    expected_categories = {
        'UNIVERSAL_HUMAN',
        'CELEBRITY_POPCULTURE', 
        'LANGUAGE_SLANG',
        'BRAND_PRODUCT',
        'FORMAT_CONTEXT',
        'COMPLEX_HYBRID'
    }
    
    unexpected_categories = categories_found - expected_categories
    if unexpected_categories:
        print(f"⚠️  Unexpected categories found: {unexpected_categories}")
        print("The prompt may need further refinement.")
    else:
        print("✅ All categories match the predefined 6-category framework!")
    
    return results, categories_found

if __name__ == "__main__":
    test_constrained_categories()