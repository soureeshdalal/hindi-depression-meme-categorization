#!/usr/bin/env python3
"""
NICHE_SUBCULTURE_DEPENDENT Meme Translation Pipeline
Translates memes by replacing niche Western subculture references with Indian equivalents
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
SOURCE_IMAGES_DIR = "categorized_memes/NICHE_SUBCULTURE_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "niche_subculture_dependent_adaptations.csv"
CHECKPOINT_FILE = "niche_subculture_dependent_progress.json"

TARGET_CATEGORY = "NICHE_SUBCULTURE_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# NICHE_SUBCULTURE_DEPENDENT TRANSLATION PROMPT
# ============================================================================

NICHE_SUBCULTURE_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - NICHE_SUBCULTURE_DEPENDENT AUTOMATION PROMPT

MISSION
Translate mental health memes by replacing niche Western subculture references (anime/weeb/otaku culture) with Indian fandom equivalents using Hinglish in DEVANAGARI SCRIPT (देवनागरी).

CRITICAL REQUIREMENTS

REQUIREMENT #1: DEVANAGARI SCRIPT ONLY
ALL Hindi text MUST be in Devanagari (देवनागरी), NEVER Romanized.
WRONG: "Tera hero real nahi hai" 
CORRECT: "तेरा hero real नहीं है"

REQUIREMENT #2: REPLACE NICHE SUBCULTURE WITH INDIAN EQUIVALENTS

Western Subculture → Indian Replacement:
- Weeb/Otaku → Bollywood fanatic / Cricket fanatic
- Waifu → Bollywood hero/heroine crush
- Isekai (rebirth genre) → Escape from reality / Rebirth desire
- Weeb trash → Nalla (नल्ला - jobless/useless)
- Fictional character obsession → Celebrity parasocial relationship
- Anime debate → Bollywood debate / Cricket debate

SUBCULTURE REPLACEMENT GUIDE

WEEB/OTAKU CULTURE → BOLLYWOOD/CRICKET FANDOM:
- Weeb → Bollywood fanatic (Bollywood का दीवाना)
- Otaku → Die-hard fan (पागल fan)
- Weeb trash → Nalla (नल्ला)

WAIFU/FICTIONAL CRUSH → BOLLYWOOD CRUSH:
- Waifu → Bollywood hero/heroine (तेरी heroine)
- Your waifu isn't real → Hero acting kar raha hai (Hero तो acting कर रहा है)
- Waifu won't notice you → Hero real में ignore karega (Real life में देखेगा भी नहीं)

ISEKAI/ESCAPISM → REBIRTH/ESCAPE:
- Isekai (rebirth genre) → Dobaara janam (दोबारा जनम)
- Want to die & restart → Escape from life (ज़िन्दगी से भागना)
- Fantasy world → Dream world (सपनों की दुनिया)

ANIME DEBATES → BOLLYWOOD/CRICKET DEBATES:
- Rem vs Megumin → SRK vs Salman (SRK या Salman)
- Best waifu → Best actor/actress (Best actor कौन)
- Isekai preference → Marriage type (Arranged या Love)

SOCIAL ISOLATION:
- Friends can't insult you → Relatives can't judge (रिश्तेदार बेइज़्ज़त नहीं करेंगे)
- No friends → No social life (बाहर नहीं निकलना)

THREE-STEP TRANSLATION PROCESS

STEP 1: SUBCULTURE ANALYSIS

ORIGINAL SUBCULTURE: "Fucking Weeb / Your Waifu isn't real"

Western Elements:
- Weeb (anime obsessive insult)
- Waifu (fictional female character crush)
- Otaku subculture mockery
- Parasocial relationship criticism

Indian Replacement:
- Bollywood fanatic (obsessive fan)
- Hero/heroine crush (Bollywood celebrity)
- Same parasocial dynamic

Why It Works:
Both: Obsessive fandom of fictional/distant figures
Both: Social mockery for unrealistic attachment
Weeb recognition: Very Low (niche internet)
Bollywood fan: Very High (universal in India)

STEP 2: DEVANAGARI TRANSLATION

ORIGINAL ENGLISH:
"Fucking Weeb / Your Waifu isn't real / 
If your waifu met you in real life, she wouldn't even look at you"

DEVANAGARI OPTIONS:
1. "पागल Bollywood fan
   तेरा hero real नहीं है
   Real life में मिले तो देखेगा भी नहीं तुझे"

2. "Bollywood का दीवाना
   Hero तो acting कर रहा है
   असल में मिले तो ignore करेगा"

3. "तू Bollywood के पीछे पागल है
   वो सब फ़र्ज़ी है
   तुझे जानता भी नहीं"

SELECTED: Option 2
"Bollywood का दीवाना
Hero तो acting कर रहा है
असल में मिले तो ignore करेगा"

RATIONALE:
Devanagari for Hindi (का, दीवाना, तो, कर, रहा, है, असल, में, मिले, ignore, करेगा)
"Bollywood" and "Hero" kept in English (common terms) Natural Hinglish flow
Same mockery of parasocial relationship
"Acting kar raha hai" = "isn't real" equivalent
Relatable to all Indians (Bollywood fandom universal)

STEP 3: VALIDATION

DEVANAGARI CHECK:
   - All Hindi in देवनागरी? YES (का, दीवाना, तो, कर, रहा, है, असल, में, मिले, करेगा)
   - NO Romanization? YES (not "ka", "deewana", "to")

SUBCULTURE REPLACEMENT:
   - Weeb → Bollywood fan 
   - Waifu → Hero 
   - Otaku culture → Bollywood fandom 
   - Mockery preserved 

HUMOR PRESERVED:
   - Social mockery? YES 
   - Parasocial relationship critique? YES 
   - Self-aware depression humor? YES 

NATURAL HINGLISH:
   - Sounds conversational? YES 
   - Not too formal? YES 

VALIDATED TRANSLATION:
"Bollywood का दीवाना
Hero तो acting कर रहा है
असल में मिले तो ignore करेगा"

IMAGE GENERATION

Generate new meme image with:
- Same visual layout (keep composition)
- Replaced Western subculture with Indian equivalents:
  * Anime imagery → Bollywood posters/imagery
  * Weeb references → Bollywood fan references
  * Otaku culture → Indian fandom culture
- Text in Devanagari (का दीवाना...)
- Same color scheme and style
- Font: Noto Sans Devanagari, bold, white with shadow

SUBCULTURE REPLACEMENT QUICK REFERENCE

WEEB/OTAKU:
- Weeb → Bollywood fanatic, Cricket fanatic
- Otaku → Die-hard fan (पागल fan)
- Weeb trash → Nalla (नल्ला)

WAIFU/CRUSH:
- Waifu → Hero/Heroine (तेरी heroine)
- Waifu isn't real → Acting kar raha hai (तो acting कर रहा है)
- Won't notice you → Ignore karega (ignore करेगा)

ISEKAI/ESCAPISM:
- Isekai → Rebirth (दोबारा जनम)
- Want to die → Escape life (ज़िन्दगी से भागना)
- Fantasy world → Dream world (सपनों की दुनिया)

DEBATES:
- Anime character debates → SRK vs Salman, Arranged vs Love
- Best waifu → Best actor (Best actor कौन)

SOCIAL:
- Friends → Relatives (रिश्तेदार)
- No friends → No social life (बाहर नहीं निकलना)

DEVANAGARI ENFORCEMENT

BEFORE generating:
- Is translation in Devanagari? (का ✅ vs "ka" ❌)
- Can see Hindi characters (क, ा)?
- NO Romanization anywhere?

AFTER generating:
- Text in image is Devanagari? (तेरा hero ✅)
- NOT Romanized? (not "tera hero" ❌)

QUALITY CHECKLIST
[ ] Hindi in Devanagari (देवनागरी) ← CRITICAL
[ ] NO Romanization (no "ka", "to", "hai" in Latin)
[ ] Niche subculture replaced (Weeb → Bollywood fan, Waifu → Hero)
[ ] Indian replacement relatable (70%+ Indians)
[ ] Theme preserved (parasocial, escapism, social mockery)
[ ] Humor intact (same self-aware depression)
[ ] Hinglish natural (conversational)
[ ] Matras correct (ा, ि, ी, ु, ू, े, ै, ो, ौ)

✨ REMEMBER
Goal: Replace ultra-niche Western anime/otaku subculture with universal Indian fandom/social dynamics that 80%+ of Indians recognize.

Success markers:
Devanagari (का दीवाना, NOT "ka deewana")
Indian equivalent (Bollywood not anime, Hero not Waifu, Nalla not weeb trash)
Theme intact (parasocial, escapism, social mockery)
No anime knowledge needed

Key insight: Anime subculture has ~5% recognition in India. Bollywood fandom, family pressure, cricket debates have ~95% recognition. This category requires the most aggressive cultural substitution.

CRITICAL: Generate the translated meme image with Indian fandom context and Devanagari script.
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
    """Load NICHE_SUBCULTURE_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'western_subculture', 'indian_replacement',
            'hindi_translation', 'rationale', 'status', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write completed memes
        for meme_id, data in progress['completed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'western_subculture': data.get('western_subculture', ''),
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
                'western_subculture': '',
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
        'western_subculture': '',
        'indian_replacement': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract western subculture
    west_match = re.search(r'Western:\s*(.+)', response_text)
    if west_match:
        data['western_subculture'] = west_match.group(1).strip()
    
    # Extract Indian replacement
    indian_match = re.search(r'Indian:\s*(.+)', response_text)
    if indian_match:
        data['indian_replacement'] = indian_match.group(1).strip()
    
    # Extract Hindi translation
    hindi_match = re.search(r'Hindi:\s*["\']?(.+?)["\']?\s*(?:\n|Rationale)', response_text, re.DOTALL)
    if hindi_match:
        data['hindi_translation'] = hindi_match.group(1).strip()
    
    # Extract rationale
    rationale_match = re.search(r'Rationale:\s*(.+?)(?:\n\n|STEP|===|$)', response_text, re.DOTALL)
    if rationale_match:
        data['rationale'] = rationale_match.group(1).strip()
    
    return data

def translate_meme_with_gemini(model, meme_data, image_path):
    """Translate meme using Gemini API"""
    try:
        # Prepare prompt with meme context
        meme_specific_prompt = f"""{NICHE_SUBCULTURE_DEPENDENT_PROMPT}

---
## 🎯 MEME TO PROCESS:

**Meme ID:** {meme_data['meme_id']}
**Extracted English Text:**
{meme_data.get('extracted_text', 'N/A')}

**Visual Description:**
{meme_data.get('visual_description', 'N/A')}

**Cultural Elements:**
{meme_data.get('cultural_elements', 'N/A')}

**Requires Cultural Reconceptualization:**
{meme_data.get('requires_cultural_reconceptualization', 'N/A')}

**Humor Mechanism:**
{meme_data.get('humor_mechanism', 'N/A')}

**Existing Indian Equivalent Suggestions:**
{meme_data.get('indian_equivalent_suggestions', 'N/A')}

**Adaptation Strategy:**
{meme_data.get('adaptation_strategy', 'N/A')}

**Notes:**
{meme_data.get('notes', 'N/A')}

---
BEGIN PROCESSING NOW. Provide subculture analysis first, then generate the translated image with Indian fandom context in Devanagari script.
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
    print("🚀 Starting NICHE_SUBCULTURE_DEPENDENT meme translation pipeline...")
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
                'western_subculture': result.get('western_subculture', ''),
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
    print(f"📊 NICHE_SUBCULTURE_DEPENDENT TRANSLATION PROGRESS")
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
