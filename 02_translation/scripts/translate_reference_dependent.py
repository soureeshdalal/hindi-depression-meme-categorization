#!/usr/bin/env python3
"""
REFERENCE_DEPENDENT Meme Translation Pipeline
Translates memes that require specific contextual knowledge
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
SOURCE_IMAGES_DIR = "categorized_memes/REFERENCE_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "reference_dependent_adaptations.csv"
CHECKPOINT_FILE = "reference_dependent_progress.json"

TARGET_CATEGORY = "REFERENCE_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# REFERENCE_DEPENDENT TRANSLATION PROMPT
# ============================================================================

REFERENCE_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - REFERENCE_DEPENDENT AUTOMATION PROMPT

MISSION
Translate mental health memes that require specific contextual knowledge (historical events, movie quotes, tax systems, games) by replacing obscure Western references with Indian equivalents using Hinglish in DEVANAGARI SCRIPT.

CRITICAL REQUIREMENTS

REQUIREMENT #1: DEVANAGARI SCRIPT ONLY
ALL Hindi text MUST be in Devanagari, NEVER Romanized.
WRONG: "Dil pe lagi" 
CORRECT: "दिल पे लगी"

REQUIREMENT #2: REPLACE OBSCURE REFERENCES WITH INDIAN EQUIVALENTS
Don't just translate - replace with contextually equivalent Indian references.

REFERENCE REPLACEMENT GUIDE

HISTORICAL EVENTS:

Western Reference: Great Depression (1920)
Indian Replacement: Engineering students / Medical students
Why: Same high stress parallel

Western Reference: Great Emu War (Australia)
Indian Replacement: British vs Mosquitoes / Anxiety exploding back
Why: Absurd losing battle

MOVIE/MEDIA QUOTES:

Western Reference: "Personally victimized" (Mean Girls)
Indian Replacement: "बस कर पगले, रुलाएगा क्या?" (Baburao)
Why: Bollywood equivalent

Western Reference: Breaking Bad "cooking meth"
Indian Replacement: "Desi daru कैसे बनाएं" (moonshine)
Why: Indian illegal activity

SYSTEMS/INSTITUTIONS:

Western Reference: US Tax Form 1040
Indian Replacement: ITR / GST form
Why: Indian tax system

Western Reference: Tax accountants stress
Indian Replacement: CA during March closing
Why: Indian context

Western Reference: ED treatment center
Indian Replacement: Army recruitment / Nani's house
Why: Weight checking context

GAMES/ACTIVITIES:

Western Reference: Tabletop RPG
Indian Replacement: Board game / Card game
Why: Generic gaming

Western Reference: Animal Crossing
Indian Replacement: Generic cute game
Why: No direct equivalent

Western Reference: Counting sheep
Indian Replacement: Counting files / rupees
Why: Idiom not common in India

ADULT CONTENT REFERENCES:

Western Reference: Pornhub casting couch
Indian Replacement: Viva exam panel / Corporate interview
Why: Anxiety/trapped feeling

Western Reference: "Cooking meth" search
Indian Replacement: "Desi daru kaise banaye"
Why: Indian illegal activity

COMPLETE EXAMPLES

EXAMPLE 1: Great Depression Reference
ORIGINAL: "great depression / 1920 / 2020 / Sad boy year"
HINDI: "Engineering Students / Medical Students / दोनों high stress में हैं"
REPLACEMENT: Replace historical depression with student stress
RECOGNITION: Everyone knows student pressure

EXAMPLE 2: Mean Girls Quote
ORIGINAL: "Too depressed for nap? Welcome to Existential Crisis! / Raise your hand if you've felt personally victimized"
HINDI: "Depression Nap के लिए बहुत depressed? / Existential Crisis में welcome! / 'बस कर पगले, रुलाएगा क्या?'"
REPLACEMENT: Mean Girls quote to Baburao (Hera Pheri) quote
ALTERNATIVE: "दिल पे सीधा लगी" (Hit straight to heart)

EXAMPLE 3: US Tax Form 1040
ORIGINAL: "1040 1040 1040 (sheep counting) / Annual sleep disorder for tax accountants"
HINDI: "ITR ITR ITR (counting files) / CA की March closing के समय नींद नहीं आती"
REPLACEMENT: 1040 form to ITR, Tax accountants to CA
CONTEXT: March = Financial year end in India

EXAMPLE 4: ED Treatment Weigh-In
ORIGINAL: "how i pull up to my weigh-in at the ED treatment center after relapsing"
HINDI: "Army medical में weight check के लिए जब पहुँचता हूँ"
ALTERNATIVE: "Nani के घर जब जाता हूँ और वो check करती है"
REPLACEMENT: ED center to Army recruitment / Grandmother
CONTEXT: Weight checking situations in Indian context

EXAMPLE 5: Great Emu War
ORIGINAL: "Australia / Emus (with laser eyes)"
HINDI: "मैं anxiety को suppress करने की कोशिश / Anxiety (वापस explode होती है)"
ALTERNATIVE: "British Empire / Mosquitoes in India"
REPLACEMENT: Absurd historical war to Mental health battle
CONTEXT: Losing battle metaphor

EXAMPLE 6: Breaking Bad Meth Search
ORIGINAL: "Google how to cook meth"
HINDI: "Google: Desi daru कैसे बनाएं"
ALTERNATIVE: "Ganja कैसे उगाएं"
REPLACEMENT: Meth cooking to Indian illegal activity
CONTEXT: Dark humor about illegal activities

EXAMPLE 7: Pornhub Casting Couch
ORIGINAL: "Porn hub (Animal Crossing character on casting couch)"
HINDI: "Viva exam panel में जब बैठता हूँ"
ALTERNATIVE: "Corporate interview में panic"
REPLACEMENT: Adult content anxiety to Academic/job anxiety
CONTEXT: Extreme nervousness/trapped feeling

REFERENCE REPLACEMENT QUICK REFERENCE

Historical:
- Great Depression to Student stress
- Emu War to Anxiety battles

Movie Quotes:
- Mean Girls to Baburao quotes
- Breaking Bad to Desi alternatives

Systems:
- Tax Form 1040 to ITR/GST
- ED center to Army medical / Nani's house

Cultural:
- Counting sheep to Counting files/rupees
- Casting couch to Viva panel/interview

DEVANAGARI ENFORCEMENT

BEFORE generating:
- Is translation in Devanagari? (दिल पे लगी vs "dil pe lagi")
- Can see Hindi characters (द, ि, ल)?
- NO Romanization anywhere?

AFTER generating:
- Text in image is Devanagari? (बस कर पगले)
- NOT Romanized? (not "bas kar pagle")

QUALITY CHECKLIST
- Hindi in Devanagari - CRITICAL
- NO Romanization (no "dil", "pe", "lagi" in Latin)
- Obscure reference replaced with Indian equivalent
- Context preserved (same emotional situation)
- Humor intact (same comedic impact)
- Hinglish natural (conversational)
- Matras correct

REMEMBER
Goal: Replace obscure Western references (historical events, movie quotes, systems) with contextually equivalent Indian references that preserve the emotional situation.

Success markers:
- Devanagari (दिल पे लगी, NOT "dil pe lagi")
- Reference replaced (ITR not 1040, Baburao not Mean Girls)
- Context intact (same situation/emotion)
- Widely recognizable (70%+ Indians understand)

Key insight: Unlike POP_CULTURE (recognizing characters), this is about understanding CONTEXT that requires specific knowledge. Replace the reference with an Indian equivalent that creates the same understanding.

CRITICAL: Generate the translated meme image with Indian contextual references and Devanagari script.

YOUR 8 MEMES
Meme ID: TE-32 - Reference Type: Great Depression (1920) - Indian Replacement: Engineering/Medical students
Meme ID: TE-413 - Reference Type: ED treatment weigh-in - Indian Replacement: Army medical / Nani's house
Meme ID: TE-44 - Reference Type: Great Emu War - Indian Replacement: Anxiety battle metaphor
Meme ID: TE-442 - Reference Type: Mean Girls quote - Indian Replacement: बस कर पगले, रुलाएगा क्या
Meme ID: TE-602 - Reference Type: US Tax Form 1040 - Indian Replacement: ITR / CA की March closing
Meme ID: TE-644 - Reference Type: RPG/Goth culture - Indian Replacement: Bheja fry / Dimaag ka dahi
Meme ID: VA-244 - Reference Type: Pornhub casting couch - Indian Replacement: Viva exam panel
Meme ID: VA-302 - Reference Type: Breaking Bad meth - Indian Replacement: Desi daru कैसे बनाएं

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
    """Load REFERENCE_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'reference_type', 'indian_replacement',
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
                'reference_type': data.get('reference_type', ''),
                'indian_replacement': data.get('indian_replacement', ''),
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
                'reference_type': '',
                'indian_replacement': '',
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
        'reference_type': '',
        'indian_replacement': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract reference type
    ref_match = re.search(r'Reference Type:\s*(.+)', response_text)
    if ref_match:
        data['reference_type'] = ref_match.group(1).strip()
    
    # Extract replacement
    replacement_match = re.search(r'(?:Indian Replacement|REPLACEMENT):\s*(.+)', response_text)
    if replacement_match:
        data['indian_replacement'] = replacement_match.group(1).strip()
    
    # Extract selected translation
    selected_match = re.search(r'(?:Selected|HINDI):\s*["\']?(.+?)["\']?\s*(?:\n|Rationale|REPLACEMENT)', response_text, re.DOTALL)
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
        meme_specific_prompt = f"""{REFERENCE_DEPENDENT_PROMPT}

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
BEGIN PROCESSING NOW. Identify the reference, replace with Indian equivalent, then generate the translated image.
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
    print("🚀 Starting REFERENCE_DEPENDENT meme translation pipeline...")
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
                'reference_type': result.get('reference_type', ''),
                'indian_replacement': result.get('indian_replacement', ''),
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
    print(f"📊 REFERENCE_DEPENDENT TRANSLATION PROGRESS")
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
