#!/usr/bin/env python3

import os
import json
import csv
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional
import logging
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiMemeAnalyzer:
    
    def __init__(self, api_key: str):
        # Configure Gemini with new SDK
        os.environ['GEMINI_API_KEY'] = api_key
        self.client = genai.Client(api_key=api_key)
        
        # Storage
        self.results = []
        self.errors = []
        self.discovered_categories = set()
        
        # Analysis prompt - organic category discovery with research-focused constraints
        self.analysis_prompt = """
You are a cultural adaptation AI analyzing mental health memes for cross-cultural translation from English to Hindi for an Indian audience with ZERO Western cultural knowledge.

RESEARCH MISSION: Categorize memes based on their PRIMARY cultural adaptation barrier to understand what percentage need cultural adaptation vs simple translation.

CATEGORY DISCOVERY GUIDELINES:
- Create 5-10 BROAD categories based on PRIMARY adaptation barriers
- Focus on barriers to understanding, not content themes
- Use consistent category names across similar adaptation challenges
- Categories should guide practical adaptation strategies
- CONSOLIDATE similar barriers under the same category name

EXAMPLE BARRIER TYPES (create your own category names but keep them BROAD):
- Universal emotions/reactions that need no cultural context → "UNIVERSAL_HUMAN"
- Western celebrities where recognition affects humor → "CELEBRITY_DEPENDENT"  
- Western brands/products central to meaning → "BRAND_DEPENDENT"
- English slang/wordplay that doesn't translate → "LANGUAGE_DEPENDENT"

IMPORTANT: Use the SAME category name for memes with similar adaptation barriers. Don't create variations like "UNIVERSAL_RELATABILITY" and "UNIVERSAL_EMOTIONAL_RELATABILITY" - just use "UNIVERSAL_HUMAN" for both.

ANALYSIS PROTOCOL:

STEP 1: VISUAL ANALYSIS
- Who is in the image? (celebrity, generic person, cartoon character)
- What objects/brands are visible?
- What emotions/expressions are shown?
- What is the setting/location?

STEP 2: TEXT EXTRACTION
- Extract ALL visible text from the meme
- Note if text is subtitle style, overlaid, or part of original image

STEP 3: CULTURAL DEPENDENCY ASSESSMENT
Ask: "Would an Indian person with ZERO Western cultural knowledge understand this meme?"
- Does humor require recognizing the celebrity?
- Does it need knowing specific meme formats?
- Does it reference Western cultural concepts?
- Rate dependency: 0-100% (0=universal, 100=requires deep Western knowledge)

STEP 4: HUMOR MECHANISM IDENTIFICATION
What makes this funny? (self-deprecating, situational irony, relatable anxiety, etc.)

STEP 5: CATEGORIZATION DECISION
- Identify the PRIMARY barrier to Indian audience understanding
- Create/assign a BROAD category name for this barrier type
- Ensure category guides adaptation strategy

STEP 6: ADAPTATION STRATEGY
Based on cultural dependency score:
- 0-30%: "Text translation only" (most cost-effective)
- 30-70%: Mixed approach (text + optional visual changes)
- 70-100%: "Cultural adaptation required" (face swap, object replacement, etc.)

RESPOND IN THIS EXACT JSON FORMAT:
{
    "extracted_text": "all visible text from meme",
    "visual_description": "detailed description of visual elements",
    "people_identified": "celebrity names or 'generic person' or 'cartoon character'",
    "cultural_elements": ["list", "of", "western", "cultural", "elements"],
    "cultural_dependency_score": number_0_to_100,
    "humor_mechanism": "what makes this funny (self-deprecating, ironic, relatable, etc.)",
    "discovered_category": "YOUR_BROAD_CATEGORY_NAME",
    "category_rationale": "why this category represents the PRIMARY adaptation barrier",
    "adaptation_strategy": "recommended approach based on dependency score",
    "adaptation_difficulty": "Easy/Medium/Hard/Impossible",
    "requires_face_swap": "Yes/No/Optional",
    "requires_object_replacement": "Yes/No/Optional", 
    "requires_cultural_reconceptualization": "Yes/No",
    "indian_equivalent_suggestions": ["suggestions", "for", "cultural", "replacements"],
    "notes": "additional observations for research"
}

CRITICAL CONSTRAINTS:
- Create 5-10 BROAD categories maximum across all memes
- CONSOLIDATE similar barriers under the same category name
- Focus on adaptation barriers, not content themes  
- Use consistent category names for similar barriers
- Categories must guide practical adaptation strategies
- Rate cultural dependency objectively (Indian audience perspective)
- Don't create category variations - use the SAME name for similar barriers
"""

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_meme_with_gemini(self, image_path: Path) -> Dict:
        """Analyze single meme with Gemini Vision"""
        try:
            logger.info(f"Analyzing {image_path.name}...")
            
            # Load image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Create image part for Gemini 2.5
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg'
            )
            
            # Call Gemini 1.5 Pro (free tier)
            response = self.client.models.generate_content(
                model='gemini-1.5-pro',
                contents=[
                    image_part,
                    self.analysis_prompt
                ]
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean up response (remove markdown formatting and extra text)
            if '```json' in response_text:
                # Extract JSON from markdown code block
                start_idx = response_text.find('```json') + 7
                end_idx = response_text.find('```', start_idx)
                if end_idx != -1:
                    response_text = response_text[start_idx:end_idx]
                else:
                    response_text = response_text[start_idx:]
            elif response_text.startswith('```') and response_text.endswith('```'):
                response_text = response_text[3:-3]
            
            # Remove any leading text before JSON
            if '{' in response_text:
                json_start = response_text.find('{')
                response_text = response_text[json_start:]
            
            # Parse JSON
            analysis = json.loads(response_text)
            
            # Add metadata
            analysis['meme_id'] = image_path.stem
            analysis['file_path'] = str(image_path)
            analysis['processing_timestamp'] = time.time()
            
            # Track discovered categories
            if 'discovered_category' in analysis:
                self.discovered_categories.add(analysis['discovered_category'])
            
            logger.info(f"✅ Successfully analyzed {image_path.name}")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {image_path.name}: {e}")
            logger.error(f"Raw response: {response.text}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing {image_path.name}: {e}")
            raise

    def analyze_sample_batch(self, dataset_dir: Path, sample_size: int = 50) -> None:
        """Analyze a sample batch of memes for category discovery"""
        
        # Get all image files
        image_files = sorted(list(dataset_dir.glob("**/*.jpg")) + list(dataset_dir.glob("**/*.png")))
        
        if not image_files:
            logger.error(f"No image files found in {dataset_dir}")
            return
        
        # Select sample
        if sample_size > len(image_files):
            sample_size = len(image_files)
            logger.warning(f"Sample size reduced to {sample_size} (total available)")
        
        # Take evenly distributed sample
        step = len(image_files) // sample_size
        sample_files = image_files[::step][:sample_size]
        
        logger.info(f"Analyzing {len(sample_files)} sample memes from {len(image_files)} total")
        
        # Process sample
        for i, image_path in enumerate(sample_files, 1):
            try:
                logger.info(f"Processing {i}/{len(sample_files)}: {image_path.name}")
                
                analysis = self.analyze_meme_with_gemini(image_path)
                self.results.append(analysis)
                
                # Rate limiting - be respectful to API
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Failed to analyze {image_path.name}: {str(e)}"
                logger.error(error_msg)
                self.errors.append({
                    'meme_id': image_path.stem,
                    'file_path': str(image_path),
                    'error': error_msg,
                    'timestamp': time.time()
                })
                
                # Continue processing other memes
                continue
        
        logger.info(f"Sample analysis complete. Processed: {len(self.results)}, Errors: {len(self.errors)}")

    def save_results(self, output_dir: Path) -> None:
        """Save analysis results"""
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        results_path = output_dir / "gemini_analysis_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'errors': self.errors,
                'discovered_categories': list(self.discovered_categories),
                'total_processed': len(self.results),
                'total_errors': len(self.errors),
                'analysis_timestamp': time.time()
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {results_path}")
        
        # Save as CSV for easy analysis
        if self.results:
            csv_path = output_dir / "gemini_analysis_results.csv"
            
            # Get all possible keys from results
            all_keys = set()
            for result in self.results:
                all_keys.update(result.keys())
            
            fieldnames = sorted(list(all_keys))
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    # Handle list fields by converting to string
                    row = {}
                    for key in fieldnames:
                        value = result.get(key, '')
                        if isinstance(value, list):
                            row[key] = '; '.join(str(v) for v in value)
                        else:
                            row[key] = value
                    writer.writerow(row)
            
            logger.info(f"CSV results saved to {csv_path}")
        
        # Save category discovery report
        self.generate_category_report(output_dir)
        
        # Save error log
        if self.errors:
            error_path = output_dir / "analysis_errors.json"
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, indent=2)
            logger.info(f"Error log saved to {error_path}")

    def generate_category_report(self, output_dir: Path) -> None:
        """Generate category discovery analysis report"""
        
        if not self.results:
            logger.warning("No results to analyze for category report")
            return
        
        report_path = output_dir / "category_discovery_report.txt"
        
        # Analyze discovered categories
        category_counts = {}
        difficulty_counts = {}
        dependency_scores = []
        
        for result in self.results:
            # Count categories
            category = result.get('discovered_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count difficulty levels
            difficulty = result.get('adaptation_difficulty', 'Unknown')
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            # Collect dependency scores
            score = result.get('cultural_dependency_score', 0)
            if isinstance(score, (int, float)):
                dependency_scores.append(score)
        
        # Calculate statistics
        avg_dependency = sum(dependency_scores) / len(dependency_scores) if dependency_scores else 0
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== GEMINI VISION MEME CATEGORIZATION DISCOVERY REPORT ===\n\n")
            
            f.write(f"Sample Size: {len(self.results)} memes\n")
            f.write(f"Processing Errors: {len(self.errors)}\n")
            f.write(f"Average Cultural Dependency Score: {avg_dependency:.1f}%\n\n")
            
            f.write("=== DISCOVERED CATEGORIES ===\n")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.results)) * 100
                f.write(f"{category}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n=== ADAPTATION DIFFICULTY DISTRIBUTION ===\n")
            for difficulty, count in sorted(difficulty_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.results)) * 100
                f.write(f"{difficulty}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n=== CULTURAL DEPENDENCY ANALYSIS ===\n")
            if dependency_scores:
                low_dep = len([s for s in dependency_scores if s < 30])
                med_dep = len([s for s in dependency_scores if 30 <= s < 70])
                high_dep = len([s for s in dependency_scores if s >= 70])
                
                f.write(f"Low Dependency (0-30%): {low_dep} ({low_dep/len(dependency_scores)*100:.1f}%)\n")
                f.write(f"Medium Dependency (30-70%): {med_dep} ({med_dep/len(dependency_scores)*100:.1f}%)\n")
                f.write(f"High Dependency (70-100%): {high_dep} ({high_dep/len(dependency_scores)*100:.1f}%)\n")
            
            f.write(f"\n=== SAMPLE CATEGORY EXAMPLES ===\n")
            # Show examples for each category
            category_examples = {}
            for result in self.results:
                category = result.get('discovered_category', 'Unknown')
                if category not in category_examples:
                    category_examples[category] = result
            
            for category, example in category_examples.items():
                f.write(f"\n{category}:\n")
                f.write(f"  Example: {example.get('meme_id', 'Unknown')}\n")
                f.write(f"  Text: {example.get('extracted_text', 'N/A')[:100]}...\n")
                f.write(f"  Rationale: {example.get('category_rationale', 'N/A')[:150]}...\n")
        
        logger.info(f"Category discovery report saved to {report_path}")

def main():
    """Main execution function"""
    
    # Get API key
    api_key = input("Enter your Gemini API key: ").strip()
    if not api_key:
        logger.error("API key is required")
        return
    
    # Setup paths
    dataset_dir = Path("../dataset")  # Relative to gemini_categorization folder
    output_dir = Path("analysis_output")
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Initialize analyzer
    analyzer = GeminiMemeAnalyzer(api_key)
    
    # Start with sample analysis
    sample_size = int(input("Enter sample size for initial analysis (recommended: 50): ") or "50")
    
    logger.info("Starting Gemini Vision meme analysis...")
    logger.info("This will discover categories organically from actual meme content")
    
    # Analyze sample
    analyzer.analyze_sample_batch(dataset_dir, sample_size)
    
    # Save results
    analyzer.save_results(output_dir)
    
    # Summary
    logger.info("=== ANALYSIS COMPLETE ===")
    logger.info(f"Successfully analyzed: {len(analyzer.results)} memes")
    logger.info(f"Discovered categories: {len(analyzer.discovered_categories)}")
    logger.info(f"Processing errors: {len(analyzer.errors)}")
    
    if analyzer.discovered_categories:
        logger.info("Categories discovered:")
        for category in sorted(analyzer.discovered_categories):
            logger.info(f"  - {category}")
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Review the category_discovery_report.txt for detailed insights!")

if __name__ == "__main__":
    main()