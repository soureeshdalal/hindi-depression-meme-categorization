#!/usr/bin/env python3
"""
Run full 1,023 meme analysis with organic category discovery
"""

import os
import sys
sys.path.append('.')
from gemini_meme_analyzer import GeminiMemeAnalyzer
from pathlib import Path
import time

def run_full_analysis():
    """Run complete analysis on all 1,023 memes"""
    
    # API key (using the one from previous tests)
    api_key = "AIzaSyCE7Tn19v0m6BpiaESw90uhjDos0HrhB7U"
    
    # Initialize analyzer
    analyzer = GeminiMemeAnalyzer(api_key)
    
    # Run full analysis
    dataset_dir = Path("../dataset")
    print("Starting FULL 1,023 meme analysis.")
    
    # Get total count for confirmation
    all_images = list(dataset_dir.glob("**/*.jpg"))
    print(f"Total images found: {len(all_images)}")
    
    if len(all_images) != 1023:
        print(f"Expected 1,023 images, found {len(all_images)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run analysis with all images (sample_size = total count)
    analyzer.analyze_sample_batch(dataset_dir, sample_size=len(all_images))
    
    # Save results
    output_dir = Path("analysis_output")
    analyzer.save_results(output_dir)
    
    # Final summary
    print(f"\nFULL ANALYSIS COMPLETE!")
    print(f"Successfully analyzed: {len(analyzer.results)}/{len(all_images)} memes")
    print(f"Processing errors: {len(analyzer.errors)}")
    print(f"Categories discovered: {len(analyzer.discovered_categories)}")
    
    # Show category distribution
    category_counts = {}
    for result in analyzer.results:
        category = result.get('discovered_category', 'Unknown')
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\n=== FINAL CATEGORY DISTRIBUTION ===")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(analyzer.results)) * 100
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    # Calculate research insights
    dependency_scores = []
    easy_adaptation = 0
    
    for result in analyzer.results:
        score = result.get('cultural_dependency_score', 0)
        if isinstance(score, (int, float)):
            dependency_scores.append(score)
        
        if result.get('adaptation_difficulty') == 'Easy':
            easy_adaptation += 1
    
    if dependency_scores:
        avg_dependency = sum(dependency_scores) / len(dependency_scores)
        print(f"\n=== RESEARCH INSIGHTS ===")
        print(f"Average Cultural Dependency: {avg_dependency:.1f}%")
        print(f"Easy Adaptation (text-only): {easy_adaptation} ({easy_adaptation/len(analyzer.results)*100:.1f}%)")
        print(f"Complex Adaptation needed: {len(analyzer.results)-easy_adaptation} ({(len(analyzer.results)-easy_adaptation)/len(analyzer.results)*100:.1f}%)")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Check category_discovery_report.txt for detailed analysis")
    print("CSV file ready for statistical analysis")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("MENTAL HEALTH MEME CULTURAL ADAPTATION ANALYSIS")
    print("Full Dataset: 1,023 memes")
    print("Model: Gemini 3 Pro Vision")
    print("Organic Category Discovery (5-10 categories)")
    print("=" * 60)
    
    start_time = time.time()
    success = run_full_analysis()
    end_time = time.time()
    
    if success:
        duration = (end_time - start_time) / 60  # Convert to minutes
        print(f"\n Total processing time: {duration:.1f} minutes")
        print("🎉 Ready for research publication!")
    else:
        print("Analysis incomplete. Check logs for issues.")