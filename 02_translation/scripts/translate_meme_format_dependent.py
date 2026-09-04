#!/usr/bin/env python3
"""
MEME_FORMAT_DEPENDENT Meme Translation Pipeline
Translates memes that use specific Western meme formats/templates
"""

import os
import json
import csv
import time
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = "AIzaSyCE7Tn19v0m6BpiaESw90uhjDos0HrhB7U"
GEMINI_MODEL = "gemini-3-pro-image-preview"

CSV_PATH = "gemini_categorization/analysis_output/gemini_analysis_results.csv"
SOURCE_IMAGES_DIR = "categorized_memes/MEME_FORMAT_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "meme_format_dependent_adaptations.csv"
CHECKPOINT_FILE = "meme_format_dependent_progress.json"

TARGET_CATEGORY = "MEME_FORMAT_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# MEME_FORMAT_DEPENDENT TRANSLATION PROMPT
# ============================================================================

MEME_FORMAT_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - MEME_FORMAT_DEPENDENT AUTOMATION PROMPT

MISSION
Translate mental health memes that use specific Western meme formats/templates by preserving the format structure while translating text to Hinglish in DEVANAGARI SCRIPT.

CRITICAL REQUIREMENTS

REQUIREMENT #1: DEVANAGARI SCRIPT ONLY
ALL Hindi text MUST be in Devanagari, NEVER Romanized.
WRONG: "Hamesha se tha" 
CORRECT: "हमेशा से था"

REQUIREMENT #2: PRESERVE MEME FORMAT STRUCTURE
Keep the visual template/layout, only translate/adapt the text.
These memes rely on recognizing a specific format - don't destroy it!

FORMAT REPLACEMENT GUIDE

FORMATS TO KEEP (translate text only):

Format: "Always has been"
- Original Text: "Always have been"
- Hindi Translation: "हमेशा से था"
- Why: Format recognizable in India

Format: Trade Offer
- Original Text: "I receive / You receive"
- Hindi Translation: "मुझे मिलता है / तुम्हें मिलता है"
- Why: TikTok format popular

Format: Girls vs Boys
- Original Text: "Girls: / Boys:"
- Hindi Translation: "लड़कियाँ: / मैं:"
- Why: Universal comparison

Format: Spotify Wrapped
- Original Text: Stats card
- Hindi Translation: Keep stats style, translate text
- Why: Spotify popular in India

FORMATS TO ADAPT (replace Western elements):

Format: Skyrim skill tree
- Western Element: "SPEECH 100"
- Indian Replacement: "बोलचाल 100" or "बात Level MAX"
- Why: Game interface less known

Format: Pawn Stars
- Western Element: Shop owner
- Indian Replacement: Indian shopkeeper / Shark Tank India
- Why: Pawn Stars not known

Format: Deep-fried meme
- Western Element: Soup, baseball bat
- Indian Replacement: Chai, cricket bat / belan
- Why: Replace Western items

Format: IQ Bell Curve
- Western Element: Wojak characters
- Indian Replacement: Generic Indian characters
- Why: Wojak unfamiliar

COMPLETE EXAMPLES

EXAMPLE 1: "Always Has Been" Format
ORIGINAL: "Wait you are Depressed? / Always have been"
HINDI: "रुको, तुम Depressed हो? / हमेशा से था"
FORMAT: Keep astronaut setup, translate text
RECOGNITION: Medium (seen in Indian meme circles)

EXAMPLE 2: Skyrim Skill Tree
ORIGINAL: "The important thing is not size / SPEECH 100"
HINDI: "असली बात size नहीं, person है / बोलचाल 100"
FORMAT: Keep game interface style
ADAPTATION: "SPEECH 100" to "बोलचाल 100"

EXAMPLE 3: Trade Offer
ORIGINAL: "TRADE OFFER / I receive: Hours thinking about food / You receive: Everyone judging you"
HINDI: "सौदा / मुझे मिलता है: सारा समय खाने के बारे में सोचना / तुम्हें मिलता है: सब तुम्हें judge करते हैं"
FORMAT: Keep split screen layout

EXAMPLE 4: "Best I Can Do Is..."
ORIGINAL: "Can I be happy with recovery? / Best I can do is unhappy with both"
HINDI: "क्या मैं recovery में खुश हो सकता हूँ? / बस इतना दे सकता हूँ: दोनों में नाखुश"
FORMAT: Keep negotiation structure
OPTIONAL: Replace with Indian shopkeeper instead of Pawn Stars

EXAMPLE 5: Deep-Fried Surrealist
ORIGINAL: "soup time" (with baseball bat)
HINDI: "Chai का समय" (with cricket bat or belan)
FORMAT: Keep chaotic/distorted aesthetic
ADAPTATION: Soup to Chai, Baseball bat to Cricket bat/Belan

FORMAT PRESERVATION QUICK REFERENCE

Keep These Formats (translate only):
- "Always has been" to "हमेशा से था"
- Trade Offer split screen
- Girls vs Boys comparison
- Spotify Wrapped stats

Adapt These Elements:
- "SPEECH 100" to "बोलचाल 100"
- Pawn Stars to Indian shopkeeper
- Soup to Chai
- Baseball bat to Cricket bat / Belan
- Wojak characters to Generic Indian characters

DEVANAGARI ENFORCEMENT

BEFORE generating:
- Is translation in Devanagari? (हमेशा से vs "hamesha se")
- Can see Hindi characters (ह, म, े, श, ा)?
- NO Romanization anywhere?

AFTER generating:
- Text in image is Devanagari? (रुको तुम)
- NOT Romanized? (not "ruko tum")

QUALITY CHECKLIST
- Hindi in Devanagari - CRITICAL
- NO Romanization (no "hai", "mujhe", "ko" in Latin)
- Meme format structure preserved (layout intact)
- Timing/punchline maintained
- Humor intact (same comedic impact)
- Hinglish natural (conversational)
- Matras correct

REMEMBER
Goal: Preserve meme format/template structure while translating text to Devanagari. The format IS the joke - don't destroy it!

Success markers:
- Devanagari (हमेशा से था, NOT "hamesha se tha")
- Format preserved (layout/structure intact)
- Timing maintained (punchline works)
- Humor intact (same impact)

Key insight: These memes rely on RECOGNIZING A FORMAT. Keep the visual template, translate the words. Examples: "Always has been" = astronaut betrayal, "Trade Offer" = TikTok split screen, "SPEECH 100" = Skyrim skill tree.

CRITICAL: Generate the translated meme image maintaining the format structure with Devanagari script.

YOUR 8 MEMES
Meme ID: TE-140 - Format: Skyrim skill tree - Key Translation: बोलचाल 100
Meme ID: TE-196 - Format: Always has been - Key Translation: हमेशा से था
Meme ID: TE-484 - Format: Spotify Wrapped - Key Translation: Therapy लो
Meme ID: TE-550 - Format: Best I can do - Key Translation: बस इतना दे सकता हूँ
Meme ID: TE-589 - Format: Deep-fried - Key Translation: Chai का समय
Meme ID: TE-591 - Format: IQ Bell Curve - Key Translation: काम शुरू करो
Meme ID: VA-197 - Format: Trade Offer - Key Translation: मुझे मिलता है / तुम्हें मिलता है
Meme ID: VA-81 - Format: Girls vs Boys - Key Translation: लड़कियाँ / मैं

Time estimate: 20-30 minutes (8 memes)
Cost estimate: $0.40-$0.80
"""


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress():
    """Load existing progress or create new"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "last_processed_index": -1,
            "total_processed": 0,
            "total_to_process": 0,
            "last_checkpoint": None,
            "completed_memes": {},
            "failed_memes": {}
        }

def save_progress(progress):
    """Save progress to checkpoint file"""
    progress["last_checkpoint"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    print(f"✅ Checkpoint saved: {progress['total_processed']}/{progress['total_to_process']} completed")

def is_already_processed(meme_id, progress):
    """Check if meme already processed"""
    return meme_id in progress["completed_memes"]

# ============================================================================
# CSV OPERATIONS
# ============================================================================

def load_memes_to_process(progress):
    """Load MEME_FORMAT_DEPENDENT memes from CSV"""
    print("📖 Loading memes from CSV...")
    
    # First, check what's already translated in the output folder
    already_translated = set()
    translated_dir = Path(OUTPUT_DIR) / TARGET_CATEGORY
    
    for split in ['test', 'validation']:
        split_dir = translated_dir / split
        if split_dir.exists():
            for img_file in split_dir.glob('*.jpg'):
                meme_id = img_file.stem
                already_translated.add(meme_id)
    
    print(f"🔍 Found {len(already_translated)} already translated memes in output folder")
    
    memes_data = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('discovered_category', '').strip() == TARGET_CATEGORY:
                memes_data.append(row)
    
    print(f"📊 Total {TARGET_CATEGORY} memes in CSV: {len(memes_data)}")
    
    # Filter out already processed AND already translated
    memes_to_process = []
    skipped_translated = []
    
    for meme in memes_data:
        meme_id = meme.get('meme_id', '').strip()
        if not meme_id:
            continue
            
        # Skip if already translated in output folder
        if meme_id in already_translated:
            skipped_translated.append(meme_id)
            continue
            
        # Skip if already in progress file
        if is_already_processed(meme_id, progress):
            continue
            
        memes_to_process.append(meme)
    
    print(f"✅ Already translated (skipping): {len(skipped_translated)}")
    print(f"✅ Already processed in this run: {len(progress['completed_memes'])}")
    print(f"⏳ Remaining to process: {len(memes_to_process)}")
    
    progress['total_to_process'] = len(memes_data)
    
    return memes_to_process

def save_to_csv(progress):
    """Save all completed adaptations to CSV"""
    print("💾 Saving results to CSV...")
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'meme_id', 'original_text', 'format_type',
            'hindi_translation', 'rationale',
            'status', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write completed memes
        for meme_id, data in progress['completed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'format_type': data.get('format_type', ''),
                'hindi_translation': data.get('hindi_translation', ''),
                'rationale': data.get('rationale', ''),
                'status': 'SUCCESS',
                'error': ''
            })
        
        # Write failed memes
        for meme_id, data in progress['failed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'format_type': '',
                'hindi_translation': '',
                'rationale': '',
                'status': 'ERROR',
                'error': data.get('error', 'Unknown error')
            })
    
    print(f"✅ CSV saved: {OUTPUT_CSV}")

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def setup_output_directories():
    """Create output directory structure"""
    print("📁 Setting up output directory structure...")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    category_path = output_path / TARGET_CATEGORY
    (category_path / "test").mkdir(parents=True, exist_ok=True)
    (category_path / "validation").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure: {category_path}")
    return category_path

# ============================================================================
# GEMINI API
# ============================================================================

def setup_gemini_api():
    """Initialize Gemini API"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model

def parse_response_data(response_text):
    """Extract structured data from Gemini response"""
    import re
    
    data = {
        'format_type': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract format type
    format_match = re.search(r'Format:\s*(.+)', response_text)
    if format_match:
        data['format_type'] = format_match.group(1).strip()
    
    # Extract selected translation
    selected_match = re.search(r'(?:Selected|HINDI):\s*["\']?(.+?)["\']?\s*(?:\n|Rationale|FORMAT)', response_text, re.DOTALL)
    if selected_match:
        data['hindi_translation'] = selected_match.group(1).strip()
    
    # Extract rationale
    rationale_match = re.search(r'Rationale:\s*(.+?)(?:\n\n|STEP|===|$)', response_text, re.DOTALL)
    if rationale_match:
        data['rationale'] = rationale_match.group(1).strip()
    
    return data

def translate_meme_with_gemini(model, meme_data, image_path):
    """Translate meme using Gemini API"""
    try:
        # Prepare prompt with meme context
        meme_specific_prompt = f"""{MEME_FORMAT_DEPENDENT_PROMPT}

---
## MEME TO PROCESS:

**Meme ID:** {meme_data['meme_id']}
**Extracted English Text:**
{meme_data.get('extracted_text', 'N/A')}

**Visual Description:**
{meme_data.get('visual_description', 'N/A')}

**Cultural Elements:**
{meme_data.get('cultural_elements', 'N/A')}

**Humor Mechanism:**
{meme_data.get('humor_mechanism', 'N/A')}

**Existing Translation Suggestions:**
{meme_data.get('indian_equivalent_suggestions', 'N/A')}

**Adaptation Strategy:**
{meme_data.get('adaptation_strategy', 'N/A')}

**Notes:**
{meme_data.get('notes', 'N/A')}

---
BEGIN PROCESSING NOW. Provide analysis first, then generate the translated image maintaining the format structure.
"""

        print(f"📡 Calling Gemini {GEMINI_MODEL} for {meme_data['meme_id']}...")
        
        # Load image
        try:
            image = Image.open(image_path)
            print(f"✓ Image loaded: {image.size}")
        except Exception as e:
            print(f"✗ Error loading image: {e}")
            return False, f"Error loading image: {e}", None
        
        # Generate content
        print("⟳ Sending to Gemini API...")
        response = model.generate_content(
            [meme_specific_prompt, image],
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 4096,
            }
        )
        
        print("✓ Response received!")
        
        # Extract text analysis
        response_text = ""
        try:
            if hasattr(response, 'text'):
                response_text = response.text
                print(f"📝 Analysis length: {len(response_text)} characters")
        except Exception as e:
            print(f"⚠ Could not extract text: {e}")
        
        # Parse structured data from response
        parsed_data = parse_response_data(response_text)
        
        # Extract image from response
        image_data = None
        if hasattr(response, 'parts'):
            print(f"Response has parts: {len(response.parts)}")
            for i, part in enumerate(response.parts):
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    print(f"✓ Found image in part {i}")
                    break
        
        if image_data:
            return True, parsed_data, image_data
        else:
            print("⚠ No image generated")
            return True, parsed_data, None  # Still success if we got analysis
        
    except Exception as e:
        error_msg = str(e)
        
        if "quota" in error_msg.lower() or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return False, "QUOTA_EXCEEDED", None
        
        print(f"✗ Error: {e}")
        return False, error_msg, None

def save_translated_image(image_data, output_path):
    """Save translated image"""
    try:
        print(f"💾 Saving image to: {output_path}")
        
        if hasattr(image_data, 'save'):
            image_data.save(output_path)
            print(f"✅ Saved PIL Image")
            return True
        
        if isinstance(image_data, bytes):
            with open(output_path, 'wb') as f:
                f.write(image_data)
            print(f"✅ Saved binary data")
            return True
        
        if isinstance(image_data, str):
            import base64
            image_bytes = base64.b64decode(image_data)
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            print(f"✅ Saved base64 data")
            return True
        
        print(f"❌ Unknown image data type: {type(image_data)}")
        return False
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return False

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_memes():
    """Main processing loop"""
    print("🚀 Starting MEME_FORMAT_DEPENDENT meme translation pipeline...")
    print("=" * 80)
    
    # Setup
    progress = load_progress()
    print(f"📂 Loaded progress: {progress['total_processed']} memes already completed")
    
    category_output_path = setup_output_directories()
    memes_to_process = load_memes_to_process(progress)
    
    if len(memes_to_process) == 0:
        print("✅ All memes already processed!")
        save_to_csv(progress)
        return
    
    # Setup Gemini
    model = setup_gemini_api()
    
    # Process memes
    processed_since_checkpoint = 0
    
    for idx, meme_data in enumerate(memes_to_process):
        meme_id = meme_data.get('meme_id', '').strip()
        
        print(f"\n{'='*80}")
        print(f"🎯 Processing: {meme_id} ({progress['total_processed']+1}/{progress['total_to_process']})")
        print(f"{'='*80}")
        
        # Determine paths
        if meme_id.startswith('TE-'):
            source_image_path = Path(SOURCE_IMAGES_DIR) / "test" / f"{meme_id}.jpg"
            output_subfolder = "test"
        elif meme_id.startswith('VA-'):
            source_image_path = Path(SOURCE_IMAGES_DIR) / "validation" / f"{meme_id}.jpg"
            output_subfolder = "validation"
        else:
            print(f"❌ Unknown meme ID format: {meme_id}")
            continue
        
        if not source_image_path.exists():
            print(f"❌ Source image not found: {source_image_path}")
            continue
        
        # Call Gemini
        success, result, image_data = translate_meme_with_gemini(model, meme_data, source_image_path)
        
        if success:
            # Save image if generated
            output_filename = f"{meme_id}.jpg"
            output_path = category_output_path / output_subfolder / output_filename
            
            image_saved = False
            if image_data:
                image_saved = save_translated_image(image_data, output_path)
            
            # Update progress
            progress['completed_memes'][meme_id] = {
                'status': 'completed',
                'output_path': str(output_path) if image_saved else 'NO_IMAGE',
                'timestamp': datetime.now().isoformat(),
                'original_text': meme_data.get('extracted_text', ''),
                'format_type': result.get('format_type', ''),
                'hindi_translation': result.get('hindi_translation', ''),
                'rationale': result.get('rationale', '')
            }
            progress['total_processed'] += 1
            processed_since_checkpoint += 1
            
            print(f"✅ Success! Analysis completed" + (" + Image saved" if image_saved else " (no image)"))
            
            # Checkpoint
            if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
                save_progress(progress)
                save_to_csv(progress)
                processed_since_checkpoint = 0
        else:
            error_msg = result
            
            if error_msg == "QUOTA_EXCEEDED":
                print(f"❌ API QUOTA EXCEEDED - Stopping")
                progress['failed_memes'][meme_id] = {
                    'status': 'failed',
                    'error': 'API quota exceeded',
                    'timestamp': datetime.now().isoformat(),
                    'original_text': meme_data.get('extracted_text', ''),
                    'retry_count': 0
                }
                save_progress(progress)
                save_to_csv(progress)
                return
            
            progress['failed_memes'][meme_id] = {
                'status': 'failed',
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'original_text': meme_data.get('extracted_text', ''),
                'retry_count': 0
            }
            print(f"❌ Error: {error_msg}")
        
        time.sleep(BATCH_DELAY)
    
    # Final save
    save_progress(progress)
    save_to_csv(progress)
    
    print(f"\n{'='*80}")
    print(f"🎉 PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Total processed: {progress['total_processed']}/{progress['total_to_process']}")
    print(f"❌ Failed: {len(progress['failed_memes'])}")
    print(f"📁 Output directory: {category_output_path}")
    print(f"📄 CSV file: {OUTPUT_CSV}")
    print(f"{'='*80}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def show_progress_summary():
    """Show current progress summary"""
    if not os.path.exists(CHECKPOINT_FILE):
        print("No progress file found. Run the pipeline first.")
        return
    
    progress = load_progress()
    
    print(f"\n{'='*60}")
    print(f"📊 MEME_FORMAT_DEPENDENT TRANSLATION PROGRESS")
    print(f"{'='*60}")
    print(f"Total to process: {progress['total_to_process']}")
    print(f"Completed: {progress['total_processed']}")
    print(f"Remaining: {progress['total_to_process'] - progress['total_processed']}")
    print(f"Failed: {len(progress['failed_memes'])}")
    if progress['total_processed'] + len(progress['failed_memes']) > 0:
        success_rate = progress['total_processed']/(progress['total_processed']+len(progress['failed_memes']))*100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"Last checkpoint: {progress['last_checkpoint']}")
    print(f"{'='*60}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_progress_summary()
    else:
        process_memes()
