#!/usr/bin/env python3
"""
VISUAL_SYMBOL_DEPENDENT Meme Translation Pipeline
Translates memes that rely on Western visual symbols, gestures, or iconography
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
SOURCE_IMAGES_DIR = "categorized_memes/VISUAL_SYMBOL_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "visual_symbol_dependent_adaptations.csv"
CHECKPOINT_FILE = "visual_symbol_dependent_progress.json"

TARGET_CATEGORY = "VISUAL_SYMBOL_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# VISUAL_SYMBOL_DEPENDENT TRANSLATION PROMPT
# ============================================================================

VISUAL_SYMBOL_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - VISUAL_SYMBOL_DEPENDENT AUTOMATION PROMPT

MISSION
Translate mental health memes that rely on Western visual symbols, gestures, or iconography by replacing them with Indian equivalents using Hinglish in DEVANAGARI SCRIPT.

CRITICAL REQUIREMENTS

REQUIREMENT #1: DEVANAGARI SCRIPT ONLY
ALL Hindi text MUST be in Devanagari, NEVER Romanized.
WRONG: "Main udaas hoon" 
CORRECT: "मैं उदास हूँ"

REQUIREMENT #2: REPLACE WESTERN VISUAL SYMBOLS WITH INDIAN EQUIVALENTS
Don't just translate text - replace visual symbols, gestures, and iconography.

VISUAL SYMBOL REPLACEMENT GUIDE

RELIGIOUS/MYTHOLOGICAL SYMBOLS:

Western Symbol: Devil/Satan (red, horns)
Indian Replacement: Shakuni Mama / Rakshasa
Why: Indian tempter/demon figure

Western Symbol: Grim Reaper (black robe, scythe)
Indian Replacement: Yamraj (gold crown, mace)
Why: Indian god of death

Western Symbol: Shoulder devil
Indian Replacement: Rakshasa on shoulder
Why: Same temptation concept

GESTURES:

Western Gesture: "Rock on" hand sign
Indian Replacement: Thumbs up
Why: Universal "okay" irony

Western Gesture: Finger guns
Indian Replacement: Peace sign / Thumbs up
Why: Less aggressive

CLOTHING/FORMALITY:

Western Clothing: Tuxedo progression
Indian Replacement: T-shirt to Kurta to Sherwani
Why: Indian formality hierarchy

Western Clothing: Top hat/monocle
Indian Replacement: Maharaja turban/attire
Why: Indian aristocracy

Western Clothing: Black funeral clothes
Indian Replacement: White kurta/saree
Why: Indian mourning color

FUNERAL/DEATH RITUALS:

Western Ritual: Casket/burial
Indian Replacement: Cremation ground (Shamshan Ghat)
Why: Indian death ritual

Western Ritual: Standing funeral service
Indian Replacement: Sitting mourning ceremony
Why: Indian custom

Western Ritual: Black mourning
Indian Replacement: White mourning
Why: Indian tradition

SASS/ATTITUDE SYMBOLS:

Western Symbol: LV bag + hoop earrings
Indian Replacement: South Delhi girl shopping
Why: Indian "diva" aesthetic

Western Symbol: Baddie walk away
Indian Replacement: Pallu flip / Komolika walk
Why: Indian TV drama sass

BIRTHDAY/CELEBRATION:

Western Symbol: Betty White (Western icon)
Indian Replacement: Indian Dadi / Ratna Pathak Shah
Why: Indian grandmother figure

Western Symbol: Cake with candles
Indian Replacement: Generic celebration (keep)
Why: Universal enough

COMPLETE EXAMPLES

EXAMPLE 1: Devil on Shoulder
ORIGINAL: Devil whispering "Just sleep 10 more minutes"
HINDI: "बस 10 मिनट और सो लो, late नहीं होगे"
VISUAL REPLACEMENT:
- Devil to Shakuni Mama or Rakshasa
- Red skin/horns to Indian demon imagery
- Same shoulder position
CONTEXT: Temptation to oversleep

EXAMPLE 2: Grim Reaper
ORIGINAL: Grim Reaper (black robe, scythe) in mundane situation
HINDI: Yamraj (यमराज) casual scene में
VISUAL REPLACEMENT:
- Black robe to Traditional Yamraj attire (gold/red)
- Scythe to Mace (gada)
- Same mundane situation (bus stop, street food)
CONTEXT: Death personified in everyday life

EXAMPLE 3: "Rock On" Gesture Irony
ORIGINAL: "When you got jumped by anxiety, depression, bills..." (Person giving "rock on" hand sign while suffering)
HINDI: "जब anxiety, depression, bills ने घेर लिया / और कोई पूछे 'Theek ho?'" (Person giving thumbs up while suffering)
VISUAL REPLACEMENT:
- "Rock on" gesture to Thumbs up
- Same ironic "I'm okay" while clearly not okay
CONTEXT: Fake "I'm fine" gesture

EXAMPLE 4: Tuxedo Progression (Winnie Pooh)
ORIGINAL: Winnie Pooh in T-shirt to Tuxedo to Top hat/monocle / "I am depressed" to "I have the big sad" to "Dressed to impress but stressed"
HINDI: "मैं उदास हूँ" / "मुझे बहुत दुख है" / "दिखता हूँ fancy, पर अंदर depressed"
VISUAL REPLACEMENT:
- T-shirt to Regular T-shirt
- Tuxedo to Kurta or Sherwani
- Top hat/monocle to Maharaja attire (turban)
- Keep character progression style
CONTEXT: Escalating formality of sadness description

EXAMPLE 5: Western Funeral
ORIGINAL: Black-suited people standing at casket / "Suicidal memes were cries for help"
HINDI: "Suicidal memes सच में help के लिए थे / पर सबने edgy समझा"
VISUAL REPLACEMENT:
- Black suits to White kurtas/sarees
- Casket to Cremation ground or sitting mourners
- Standing to Sitting ceremony
CONTEXT: Funeral scene for dark humor

EXAMPLE 6: Betty White Birthday
ORIGINAL: Betty White at birthday party with mental health issues as guests
HINDI: "तुम, नींद की कमी, Depression, Anxiety"
VISUAL REPLACEMENT:
- Betty White to Indian Dadi or Ratna Pathak Shah
- Keep birthday setup
- Same joke structure
CONTEXT: Sassy grandmother dealing with problems

EXAMPLE 7: LV Bag Sass
ORIGINAL: Character with LV bag + hoop earrings walking away sassily
HINDI: "मैं जा रही हूँ"
VISUAL REPLACEMENT:
- LV bag to Generic shopping bags or designer bag
- Hoop earrings to Keep (common in India too)
- OR: Replace with Komolika-style pallu flip
- OR: South Delhi girl aesthetic
CONTEXT: Dramatic sassy exit

VISUAL SYMBOL QUICK REFERENCE

Replace These Symbols:
- Devil to Shakuni / Rakshasa
- Grim Reaper to Yamraj
- "Rock on" to Thumbs up
- Tuxedo to Sherwani
- Black funeral to White mourning
- Betty White to Indian Dadi
- LV bag sass to South Delhi girl

Keep Text Short, Focus on Visuals:
The main work is visual replacement, not text.

DEVANAGARI ENFORCEMENT

BEFORE generating:
- Is translation in Devanagari? (मैं उदास हूँ vs "main udaas hoon")
- Can see Hindi characters (म, ै, ं)?
- NO Romanization anywhere?

AFTER generating:
- Text in image is Devanagari? (बस 10 मिनट)
- NOT Romanized? (not "bas 10 minute")

QUALITY CHECKLIST
- Hindi in Devanagari - CRITICAL
- NO Romanization (no "main", "udaas", "hoon" in Latin)
- Western visual symbol replaced with Indian equivalent
- Symbol culturally appropriate (Yamraj not Grim Reaper)
- Context preserved (same emotional situation)
- Humor intact (same comedic impact)
- Hinglish natural (conversational)
- Matras correct

REMEMBER
Goal: Replace Western visual symbols, gestures, and iconography with Indian equivalents while preserving the emotional context.

Success markers:
- Devanagari (मैं उदास हूँ, NOT "main udaas hoon")
- Symbol replaced (Yamraj not Grim Reaper, Thumbs up not Rock on)
- Context intact (same situation/emotion)
- Visually appropriate for Indian audience

Key insight: Unlike POP_CULTURE (recognizing characters) or FORMAT (template structure), this is about VISUAL SYMBOLS that carry cultural meaning. Devil = evil tempter, Grim Reaper = death, "Rock on" = ironic coolness. Replace the symbol with Indian equivalent.

CRITICAL: Generate the translated meme image with Indian visual symbols and Devanagari script.

YOUR 7 MEMES
Meme ID: TE-205 - Western Symbol: Devil on shoulder - Indian Replacement: Shakuni Mama / Rakshasa
Meme ID: TE-276 - Western Symbol: Grim Reaper - Indian Replacement: Yamraj (gold crown, mace)
Meme ID: TE-461 - Western Symbol: "Rock on" gesture - Indian Replacement: Thumbs up
Meme ID: TE-512 - Western Symbol: Betty White birthday - Indian Replacement: Indian Dadi
Meme ID: TE-87 - Western Symbol: Black funeral (casket) - Indian Replacement: White mourning (cremation)
Meme ID: VA-2 - Western Symbol: Winnie Pooh tuxedo - Indian Replacement: T-shirt to Kurta to Sherwani
Meme ID: VA-305 - Western Symbol: LV bag + hoop earrings - Indian Replacement: South Delhi girl / Komolika

Time estimate: 15-25 minutes (7 memes)
Cost estimate: $0.35-$0.70
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
    """Load VISUAL_SYMBOL_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'western_symbol', 'indian_replacement',
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
                'western_symbol': data.get('western_symbol', ''),
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
                'western_symbol': '',
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
        'western_symbol': '',
        'indian_replacement': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract western symbol
    symbol_match = re.search(r'Western Symbol:\s*(.+)', response_text)
    if symbol_match:
        data['western_symbol'] = symbol_match.group(1).strip()
    
    # Extract replacement
    replacement_match = re.search(r'(?:Indian Replacement|REPLACEMENT):\s*(.+)', response_text)
    if replacement_match:
        data['indian_replacement'] = replacement_match.group(1).strip()
    
    # Extract selected translation
    selected_match = re.search(r'(?:Selected|HINDI):\s*["\']?(.+?)["\']?\s*(?:\n|Rationale|VISUAL)', response_text, re.DOTALL)
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
        meme_specific_prompt = f"""{VISUAL_SYMBOL_DEPENDENT_PROMPT}

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
BEGIN PROCESSING NOW. Identify the Western symbol, replace with Indian equivalent, then generate the translated image.
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
    print("🚀 Starting VISUAL_SYMBOL_DEPENDENT meme translation pipeline...")
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
                'western_symbol': result.get('western_symbol', ''),
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
    print(f"📊 VISUAL_SYMBOL_DEPENDENT TRANSLATION PROGRESS")
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
