#!/usr/bin/env python3
"""
Test constrained 6-category system with 50 sample memes
"""

import os
import sys
sys.path.append('.')
from gemini_meme_analyzer import GeminiMemeAnalyzer
from pathlib import Path
import json

def test_50_sample():
    """Test with 50 memes to validate category consistency"""
    
    # API key (using the one from previous test)
    api_key = "AIzaSyCE7Tn19v0m6BpiaESw90uhjDos0HrhB7U"
    
    # Initialize analyzer
    analyzer = GeminiMemeAnalyzer(api_key)
    
    # Run 50-sample analysis
    dataset_dir = Path("../dataset")
    print("Running 50-sample analysis to validate constrained categories...")
    
    analyzer.analyze_sample_batch(dataset_dir, sample_size=50)
    
    # Save results
    output_dir = Path("analysis_output")
    analyzer.save_results(output_dir)
    
    # Analyze results
    print(f"\n=== 50-SAMPLE TEST RESULTS ===")
    print(f"Successfully analyzed: {len(analyzer.results)}/50 memes")
    print(f"Processing errors: {len(analyzer.errors)}")
    print(f"Categories discovered: {len(analyzer.discovered_categories)}")
    
    # Check category compliance
    expected_categories = {
        'UNIVERSAL_HUMAN',
        'CELEBRITY_POPCULTURE', 
        'LANGUAGE_SLANG',
        'BRAND_PRODUCT',
        'FORMAT_CONTEXT',
        'COMPLEX_HYBRID'
    }
    
    unexpected_categories = analyzer.discovered_categories - expected_categories
    if unexpected_categories:
        print(f"⚠️  Unexpected categories found: {unexpected_categories}")
        print("The prompt needs further refinement.")
        return False
    else:
        print("✅ All categories match the predefined 6-category framework!")
    
    # Show category distribution
    category_counts = {}
    for result in analyzer.results:
        category = result.get('discovered_category', 'Unknown')
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\n=== CATEGORY DISTRIBUTION ===")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(analyzer.results)) * 100
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    # Calculate average dependency score
    dependency_scores = []
    for result in analyzer.results:
        score = result.get('cultural_dependency_score', 0)
        if isinstance(score, (int, float)):
            dependency_scores.append(score)
    
    if dependency_scores:
        avg_dependency = sum(dependency_scores) / len(dependency_scores)
        print(f"\nAverage Cultural Dependency Score: {avg_dependency:.1f}%")
    
    print(f"\nDetailed results saved to: {output_dir}")
    return True

if __name__ == "__main__":
    success = test_50_sample()
    if success:
        print("\n🎉 Ready for full 1,023 meme analysis!")
    else:
        print("\n❌ Need to fix category constraints before full analysis.")