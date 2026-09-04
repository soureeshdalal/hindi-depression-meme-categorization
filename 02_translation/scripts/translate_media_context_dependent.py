#!/usr/bin/env python3
"""
MEDIA_CONTEXT_DEPENDENT Meme Translation Pipeline
Translates memes by replacing Western media references with Indian media equivalents
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
SOURCE_IMAGES_DIR = "categorized_memes/MEDIA_CONTEXT_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "media_context_dependent_adaptations.csv"
CHECKPOINT_FILE = "media_context_dependent_progress.json"

TARGET_CATEGORY = "MEDIA_CONTEXT_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# MEDIA_CONTEXT_DEPENDENT TRANSLATION PROMPT
# ============================================================================

MEDIA_CONTEXT_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - MEDIA_CONTEXT_DEPENDENT AUTOMATION PROMPT

MISSION
Translate mental health memes by replacing Western media references (specific anime, Marvel movies, TV shows) with Indian media equivalents using Hinglish in DEVANAGARI SCRIPT.

CRITICAL REQUIREMENTS

REQUIREMENT #1: DEVANAGARI SCRIPT ONLY
ALL Hindi text MUST be in Devanagari, NEVER Romanized.
WRONG: "Yeh depression jaisa nahi dikhta" 
CORRECT: "यह depression जैसा नहीं दिखता"

REQUIREMENT #2: REPLACE WESTERN MEDIA WITH INDIAN MEDIA

Western Media to Indian Replacement:
- Anime characters to Bollywood comedic failures (Rajpal Yadav, Crime Master Gogo)
- Thor: Ragnarok (Marvel) to Baahubali (epic divine power scene)
- My Hero Academia to Generic Bollywood dramatic scene
- SpongeBob/Squidward to Tired Indian office worker / student
- Breaking Bad (Hank) to SRK screaming (Devdas) / Baburao (Hera Pheri)
- Clarinet sounds to Shehnai sounds / sad background music

MEDIA REPLACEMENT GUIDE

ANIME TO BOLLYWOOD:

Western Anime: Anime "useless" characters
Indian Replacement: Rajpal Yadav, Crime Master Gogo
Devanagari Example: "बेकार characters"
Why: Comedic failure archetypes

Western Anime: My Hero Academia
Indian Replacement: Dramatic Bollywood scene
Devanagari Example: "देखो यह bridge"
Why: Over-dramatic reactions

Western Anime: Multiple anime references
Indian Replacement: Generic Bollywood fails
Devanagari Example: "सब बेकार हैं"
Why: Universal failure theme

MARVEL/SUPERHERO TO INDIAN EPIC:

Western: Thor: Ragnarok
Indian: Baahubali
Devanagari Example: "यह depression जैसा नहीं दिखता"
Why: Divine power/epic scale

Western: Mjolnir (hammer)
Indian: Baahubali's weapon
Devanagari Example: "दिव्य शक्ति"
Why: Mythical weapon parallel

Western: MCU
Indian: Mahabharata TV series
Devanagari Example: "महाभारत का scene"
Why: Epic mythology

TV SHOWS TO BOLLYWOOD FILMS:

Western: SpongeBob/Squidward
Indian: Tired office worker
Devanagari Example: "सुबह उठना"
Why: Relatable everyday person

Western: Breaking Bad (Hank screaming)
Indian: SRK screaming (Devdas) / Baburao
Devanagari Example: "हाँक नहीं, 'अंजली'"
Why: Dramatic Bollywood scream

SOUNDS/MUSIC:

Western: Clarinet noises
Indian: Shehnai / sad background music
Devanagari Example: "दुखद संगीत"
Why: Indian instrument/music

THREE-STEP TRANSLATION PROCESS

STEP 1: MEDIA REFERENCE ANALYSIS

ORIGINAL REFERENCE: "Thor: Ragnarok - Mjolnir hammer scene"

Western Context:
- Marvel Cinematic Universe
- Thor tries to lift hammer, can't
- Shows weakness/loss of power
- "Smiling depression" metaphor

Indian Replacement:
- Baahubali (2015-2017 Indian epic)
- Divine weapon/power scene
- Same "looks strong but struggling" theme
- Massive recognition in India

Why It Works:
- Both: Epic scale with divine weapons
- Both: Character facing internal struggle
- Thor recognition: Medium (urban youth)
- Baahubali recognition: Very High (pan-India blockbuster)

STEP 2: DEVANAGARI TRANSLATION

ORIGINAL ENGLISH: "BUT THAT'S NOT WHAT DEPRESSION LOOKS LIKE / DARLING, YOU'VE NO IDEA WHAT DEPRESSION LOOKS LIKE"

DEVANAGARI OPTIONS:
1. "लेकिन depression तो ऐसा नहीं दिखता / प्यारे, तुम्हें पता नहीं depression कैसा दिखता है"
2. "यह depression जैसा नहीं लगता / तुम्हें क्या पता depression क्या होता है"
3. "Depression ऐसा नहीं होता / तुम समझ नहीं सकते यह कैसा है"

SELECTED: Option 2
"यह depression जैसा नहीं लगता / तुम्हें क्या पता depression क्या होता है"

RATIONALE:
- Devanagari for Hindi
- Natural Hinglish flow
- Maintains condescending tone
- "Smiling depression" theme preserved

STEP 3: VALIDATION

DEVANAGARI CHECK:
- All Hindi in देवनागरी? YES
- NO Romanization? YES

MEDIA REPLACEMENT:
- Thor to Baahubali
- Mjolnir to Divine weapon
- Same metaphor (hidden struggle)

HUMOR PRESERVED:
- "Smiling depression" concept? YES
- Condescending tone? YES

NATURAL HINGLISH:
- Sounds conversational? YES
- Not too formal? YES

COMPLETE EXAMPLES

EXAMPLE 1: Thor to Baahubali (TE-98)

STEP 1: MEDIA ANALYSIS
Western: Thor: Ragnarok (Marvel) - trying to lift Mjolnir
Theme: Looks strong but struggling internally (smiling depression)
Indian: Baahubali - divine power/weapon scene
Why: Same epic scale, same internal struggle theme
Recognition: Thor (Medium) to Baahubali (Very High)

STEP 2: DEVANAGARI TRANSLATION
Original: "BUT THAT'S NOT WHAT DEPRESSION LOOKS LIKE / DARLING, YOU'VE NO IDEA WHAT DEPRESSION LOOKS LIKE"
Hindi: "यह depression जैसा नहीं लगता / तुम्हें क्या पता depression क्या होता है"

STEP 3: VALIDATION
- Devanagari: यह, जैसा, नहीं, लगता, तुम्हें, क्या, पता, होता, है
- Media: Thor to Baahubali
- Theme: Smiling depression preserved
- Recognition: Very High (Baahubali massive hit)

EXAMPLE 2: Anime Characters to Bollywood Failures (TE-294)

STEP 1: MEDIA ANALYSIS
Western: 8 different anime "useless" characters (alignment chart)
Theme: Everyone is useless/failures (depression metaphor)
Indian: Bollywood comedic failure characters
Options: Rajpal Yadav, Crime Master Gogo, Generic "bekaar" characters
Why: Same "useless" theme, universally recognized

STEP 2: DEVANAGARI TRANSLATION
Original: "The Useless Elites / The Useless Goddess / The Useless Demon / The Useless Hero / The Useless Villains / The Useless MC / The Useless Princess / The Useless Weeb"
Hindi: "बेकार Elite / बेकार देवी / बेकार दुश्मन / बेकार Hero / बेकार Villains / बेकार Main Character / बेकार Princess / बेकार हम सब"

STEP 3: VALIDATION
- Devanagari: बेकार, देवी, दुश्मन, सब
- Media: Anime characters to Generic Bollywood archetypes
- Theme: "Everyone useless" depression preserved
- Recognition: Universal (no anime knowledge needed)

EXAMPLE 3: SpongeBob to Tired Indian (VA-127)

STEP 1: MEDIA ANALYSIS
Western: Squidward (SpongeBob) + clarinet noises
Theme: Waking up miserable, no will to live
Indian: Generic tired Indian office worker/student
Sound: Shehnai or sad background music
Why: No SpongeBob needed, universal morning misery

STEP 2: DEVANAGARI TRANSLATION
Original: "Me waking up in the morning without any will to live / *sad clarinet noises*"
Hindi: "सुबह उठते हुए बिना किसी जीने की इच्छा के / *दुखद संगीत बजता है*"
Alternative: "सुबह जब आँख खुलती है *पीछे से दुखी शहनाई*"
SELECTED: Option 1 (more universal)

STEP 3: VALIDATION
- Devanagari: सुबह, उठते, हुए, बिना, किसी, जीने, की, इच्छा, के, दुखद, संगीत, बजता, है
- Media: Squidward to Generic tired person
- Theme: Morning misery preserved
- Sound: Clarinet to दुखद संगीत (sad music)

EXAMPLE 4: Breaking Bad to Bollywood Screaming (VA-202)

STEP 1: MEDIA ANALYSIS
Western: Breaking Bad - "HAAAANK" screaming scene
Theme: Desperate screaming/panic
Indian: SRK screaming "ANJALI" (KKHH) OR Baburao screaming (Hera Pheri)
Why: Bollywood has iconic dramatic screaming scenes
Options: Shah Rukh Khan screaming "ANJALI" or "RAHUL", Baburao screaming frantically, Devdas SRK screaming
SELECTED: Baburao (Hera Pheri) - most meme-able

STEP 2: DEVANAGARI TRANSLATION
Original: "HAAAAAAAANK / GET THE food / THE food NOT water"
Hindi: "बाबूराव / खाना लाओ / खाना, पानी नहीं"
Alternative: "अंजली / खाना चाहिए / पानी नहीं, खाना"
SELECTED: Option 1 (Baburao reference - iconic)

STEP 3: VALIDATION
- Devanagari: बाबूराव, खाना, लाओ, पानी, नहीं
- Media: Hank (Breaking Bad) to Baburao (Hera Pheri)
- Theme: Desperate panic preserved
- Recognition: Hera Pheri = cult classic in India

EXAMPLE 5: My Hero Academia to Bollywood Drama (VA-104)

STEP 1: MEDIA ANALYSIS
Western: My Hero Academia (anime) - character fascinated by bridge
Theme: Anti-humor (dramatic reaction to mundane thing)
Indian: Bollywood over-dramatic reaction to flyover/bridge
Character: Generic Bollywood hero (or SRK signature pose)
Why: Bollywood famous for over-dramatic reactions

STEP 2: DEVANAGARI TRANSLATION
Original: "Todoroki: *is a fan of bridges* / Todoroki: Wow look at this bridge"
Hindi: "Civil Engineer: / वाह देखो यह bridge"
Alternative: "Shah Rukh Khan: देखो क्या खूबसूरत flyover है"
SELECTED: Option 2 (SRK signature = recognizable)

STEP 3: VALIDATION
- Devanagari: देखो, क्या, खूबसूरत, है
- Media: Anime character to SRK/Bollywood hero
- Theme: Over-dramatic reaction to mundane thing
- Recognition: SRK signature pose = universally known

QUALITY CHECKLIST
- Hindi in Devanagari - CRITICAL
- NO Romanization (no "yeh", "nahi", "hai" in Latin)
- Western media replaced (Thor to Baahubali, anime to Bollywood)
- Indian replacement recognizable (70%+ Indians)
- Theme/metaphor preserved (epic struggle, uselessness, panic)
- Humor intact (same emotional impact)
- Hinglish natural (conversational)
- Matras correct

DEVANAGARI ENFORCEMENT

BEFORE generating:
- Is translation in Devanagari? (यह depression vs "yeh depression")
- Can see Hindi characters?
- NO Romanization anywhere?

AFTER generating:
- Text in image is Devanagari? (तुम्हें क्या पता)
- NOT Romanized? (not "tumhe kya pata")

REMEMBER
Goal: Replace niche Western media references with Indian media equivalents that 70%+ of Indians recognize, preserving humor and using Devanagari script.

Success markers:
- Devanagari (यह depression, NOT "yeh depression")
- Indian media (Baahubali not Thor, Baburao not Hank)
- Humor intact (same emotional resonance)
- Widely recognizable (no anime/Western TV knowledge needed)

CRITICAL: Generate the translated meme image with Indian media context and Devanagari script.
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
    """Load MEDIA_CONTEXT_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'western_media', 'indian_replacement',
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
                'western_media': data.get('western_media', ''),
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
                'western_media': '',
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
        'western_media': '',
        'indian_replacement': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract western media
    media_match = re.search(r'Western(?:\s+Media)?:\s*(.+)', response_text)
    if media_match:
        data['western_media'] = media_match.group(1).strip()
    
    # Extract replacement
    replacement_match = re.search(r'(?:Indian(?:\s+Replacement)?|REPLACEMENT):\s*(.+)', response_text)
    if replacement_match:
        data['indian_replacement'] = replacement_match.group(1).strip()
    
    # Extract selected translation
    selected_match = re.search(r'(?:Selected|HINDI):\s*["\']?(.+?)["\']?\s*(?:\n|Rationale|STEP)', response_text, re.DOTALL)
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
        meme_specific_prompt = f"""{MEDIA_CONTEXT_DEPENDENT_PROMPT}

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
BEGIN PROCESSING NOW. Identify the Western media reference, replace with Indian equivalent, then generate the translated image.
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
    print("🚀 Starting MEDIA_CONTEXT_DEPENDENT meme translation pipeline...")
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
                'western_media': result.get('western_media', ''),
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
    print(f"📊 MEDIA_CONTEXT_DEPENDENT TRANSLATION PROGRESS")
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
