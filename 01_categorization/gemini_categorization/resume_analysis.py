#!/usr/bin/env python3
"""
Resume analysis from where it stopped - process remaining 863 memes
"""

import os
import sys
import json
sys.path.append('.')
from gemini_meme_analyzer import GeminiMemeAnalyzer
from pathlib import Path
import time

def resume_analysis():
    """Resume processing remaining memes"""
    
    # API key
    api_key = "AIzaSyCE7Tn19v0m6BpiaESw90uhjDos0HrhB7U"
    
    # Load existing results to see what's already processed
    output_dir = Path("analysis_output")
    results_file = output_dir / "gemini_analysis_results.json"
    
    processed_meme_ids = set()
    existing_results = []
    existing_errors = []
    
    if results_file.exists():
        print("Loading existing results...")
        with open(results_file, 'r') as f:
            data = json.load(f)
            existing_results = data.get('results', [])
            existing_errors = data.get('errors', [])
            
            # Track which memes are already processed
            for result in existing_results:
                processed_meme_ids.add(result['meme_id'])
        
        print(f"Found {len(existing_results)} already processed memes")
    
    # Initialize analyzer
    analyzer = GeminiMemeAnalyzer(api_key)
    
    # Restore existing results
    analyzer.results = existing_results
    analyzer.errors = existing_errors
    
    # Get all memes
    dataset_dir = Path("../dataset")
    all_images = sorted(list(dataset_dir.glob("**/*.jpg")))
    
    # Filter to only unprocessed memes
    remaining_images = [img for img in all_images if img.stem not in processed_meme_ids]
    
    print(f"\n{'='*60}")
    print(f"📊 RESUME ANALYSIS STATUS")
    print(f"{'='*60}")
    print(f"Total memes: {len(all_images)}")
    print(f"Already processed: {len(processed_meme_ids)}")
    print(f"Remaining to process: {len(remaining_images)}")
    print(f"{'='*60}\n")
    
    if not remaining_images:
        print("✅ All memes already processed!")
        return
    
    # Process remaining memes
    print(f"Starting to process {len(remaining_images)} remaining memes...")
    print("Note: Will hit API quota again. Run this script daily until complete.\n")
    
    for i, image_path in enumerate(remaining_images, 1):
        try:
            print(f"Processing {i}/{len(remaining_images)}: {image_path.name}")
            
            analysis = analyzer.analyze_meme_with_gemini(image_path)
            analyzer.results.append(analysis)
            
            # Track discovered categories
            if 'discovered_category' in analysis:
                analyzer.discovered_categories.add(analysis['discovered_category'])
            
            # Save progress every 10 memes
            if i % 10 == 0:
                print(f"  💾 Saving progress... ({len(analyzer.results)} total processed)")
                analyzer.save_results(output_dir)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if quota exceeded
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                print(f"\n⚠️  API Quota Exceeded at {i}/{len(remaining_images)}")
                print(f"✅ Progress saved: {len(analyzer.results)} total memes processed")
                print(f"📅 Run this script again tomorrow to continue")
                
                # Save final progress
                analyzer.save_results(output_dir)
                break
            
            # Other errors - log and continue
            print(f"  ❌ Error: {error_msg}")
            analyzer.errors.append({
                'meme_id': image_path.stem,
                'file_path': str(image_path),
                'error': error_msg,
                'timestamp': time.time()
            })
            continue
    
    # Final save
    analyzer.save_results(output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {len(analyzer.results)}/{len(all_images)}")
    print(f"Remaining: {len(all_images) - len(analyzer.results)}")
    print(f"Categories discovered: {len(analyzer.discovered_categories)}")
    print(f"{'='*60}")
    
    if len(analyzer.results) == len(all_images):
        print("\n🎉 ALL 1,023 MEMES PROCESSED!")
        print("✅ Ready for research publication!")
    else:
        print(f"\n📅 Run this script again to process remaining {len(all_images) - len(analyzer.results)} memes")

if __name__ == "__main__":
    resume_analysis()
