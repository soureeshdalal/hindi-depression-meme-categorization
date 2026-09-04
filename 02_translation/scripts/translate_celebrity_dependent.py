#!/usr/bin/env python3
"""
CELEBRITY_DEPENDENT Meme Translation Pipeline
Translates memes by replacing Western celebrities with Indian celebrities
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
SOURCE_IMAGES_DIR = "categorized_memes/CELEBRITY_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "celebrity_dependent_adaptations.csv"
CHECKPOINT_FILE = "celebrity_dependent_progress.json"

TARGET_CATEGORY = "CELEBRITY_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# CELEBRITY_DEPENDENT TRANSLATION PROMPT
# ============================================================================

CELEBRITY_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - CELEBRITY_DEPENDENT MEME TRANSLATION PROMPT

MISSION
Translate CELEBRITY_DEPENDENT mental health memes by replacing Western celebrities with Indian celebrities while preserving expression, pose, and emotional context.

CATEGORY OVERVIEW
- Main challenge: Face swapping
- Key task: Replace Western celebrity faces with recognizable Indian equivalents
- Preserve: Expression, pose, emotional context, humor mechanism

THREE-STEP PROCESS

STEP 1: CELEBRITY MATCHING DECISION
For each Western celebrity, find Indian equivalent based on:
- Recognition level (A-list Bollywood > Regional > Niche)
- Emotional archetype (Sad Keanu → Resilient Yuvraj)
- Cultural context (Reality TV villain → Roadies judge)
- Expression match (Crying face → Crying Bollywood scene)

Celebrity Replacement Guide:
- Will Smith (emotional) → Shah Rukh Khan, Ranbir Kapoor
- Dr. Phil (authority) → Raghu Ram (Roadies), Angry uncle
- Keanu Reeves (stoic resilience) → Yuvraj Singh, Rajesh Khanna
- Game of Thrones characters → Mirzapur characters, Gabbar Singh
- Billie Eilish (sad aesthetic) → Deepika Padukone, Alia Bhatt
- Post Malone (disheveled) → Shahid Kapoor (Kabir Singh)
- The Undertaker (wrestler) → The Great Khali

Recognition Tiers (Prioritize higher):
Tier 1: National Icons (Shah Rukh Khan, Amitabh Bachchan, Sachin Tendulkar)
Tier 2: Popular Bollywood (Ranbir Kapoor, Deepika Padukone, Alia Bhatt)
Tier 3: TV/Regional (Raghu Ram, TV serial characters)
Tier 4: Character Types ("Angry uncle", "Strict teacher")

Decision Output Format:
```
CELEBRITY MATCHING:
  Original: Keanu Reeves (stoic, hopeful despite sadness)
  Indian Match: Yuvraj Singh (cancer survivor, still playing)
  Archetype: Resilience, hope, perseverance
  Recognition: High (cricket legend)
  Expression: Smiling despite hardship
  Decision: REPLACE with Yuvraj Singh
```

STEP 2: TRANSLATION
Translate text to Hinglish (Devanagari):

Original: "WHEN DEATH KEEPS VISITING YOUR LIFE BUT SOMEHOW REMAIN HOPEFUL"
Options:
- "जब मौत बार-बार आए पर फिर भी hope बना रहे"
- "Death से मिलते रहो पर hopeful रहो"
- "मौत करीब आए पर हम हार नहीं मानते"

Selected: "जब मौत बार-बार आए पर फिर भी hope बना रहे"
Rationale: Maintains dark humor + hope parallel

STEP 3: IMAGE GENERATION WITH FACE SWAP

Critical face swap requirements:

A. Match Expression EXACTLY
- Original: Keanu looking sad but smiling
- Replacement: Yuvraj with same sad smile

B. Match Pose & Framing
- Head angle (looking left/right/center?)
- Body posture (sitting/standing?)
- Camera angle (close-up/medium shot?)
- Background elements

C. Maintain Context
- If celebrity in specific setting, keep similar setting
- If holding object, replace with culturally appropriate equivalent
- If wearing specific clothing, adapt to Indian equivalent

FACE SWAP SCENARIOS

SCENARIO 1: Direct Face Swap (Simple)
When: Celebrity is just a face/expression, no cultural context needed
Example: Post Malone disheveled → Shahid Kapoor (Kabir Singh) disheveled
Process:
1. Identify exact expression (tired, sad, angry)
2. Find Indian celebrity known for that expression
3. Generate image with Indian celebrity in same pose
4. Keep background, lighting, composition identical
5. Translate text

SCENARIO 2: Character Replacement (Medium)
When: Western character from show/movie → Indian character
Example: Robert Baratheon (GoT) → Kaleen Bhaiya (Mirzapur)
Process:
1. Understand character archetype (king, villain, judge)
2. Find Indian equivalent character
3. Use scene from Indian show with similar emotional tone
4. Adapt text to fit new character context

SCENARIO 3: Controversial Figure (Complex)
When: Celebrity is controversial/political (Trump, Epstein)
Options:
- Option A: Replace with generic "politician" or "controversial figure" (no face)
- Option B: Use recognizable but less controversial Indian equivalent
- Option C: Skip face swap, focus on emotion/context
IMAGE GENERATION INSTRUCTIONS

For each face swap:

Generate a meme image with these specifications:

FACE SWAP:
- Original celebrity: [Name]
- Indian replacement: [Name]
- Expression required: [Sad but smiling, hopeful]
- Setting: [Generic background / Press conference / Movie scene]
- Pose: [Head tilted slightly, looking at camera]
- Lighting: [Soft, natural light]
- Shot type: [Medium close-up, shoulders visible]

MAINTAIN FROM ORIGINAL:
- Background style: [same type]
- Composition: [same framing]
- Color grading: [same mood/tone]
- Text position: [same location]

TEXT:
- Original: "[original text]"
- Hindi: "[Hindi translation]"
- Position: [top/bottom/center]
- Font: [bold white with black outline]
- Size: [large, readable]

CRITICAL: The Indian celebrity's face should look natural, not "pasted on".
Match lighting, shadows, and perspective to background.

QUALITY CHECKLIST

Before submitting:
[ ] Indian celebrity is recognizable to Indian audience
[ ] Expression matches original emotion
[ ] Pose and framing identical to original
[ ] Face looks natural (not obviously edited)
[ ] Text translated accurately to Hinglish
[ ] Cultural context preserved (humor mechanism intact)
[ ] No controversial figures named (if sensitive topic)

SPECIAL CASES

Case 1: Multiple Celebrities in One Meme
- Replace ALL Western celebrities with Indian equivalents
- Maintain relationships (if two people arguing, keep that dynamic)

Case 2: Celebrity Not Recognizable
- If you don't know the Western celebrity, describe their expression/role
- Match by archetype (judge, musician, athlete, actor)

Case 3: Optional Face Swap
If requires_face_swap: Optional, you can:
- Option A: Keep face but adapt text heavily
- Option B: Do face swap for better cultural resonance

Case 4: No Good Indian Match
- Use generic "Bollywood actor/actress" without specifying who
- Or use character archetype (sad person, angry uncle, etc.)

REMEMBER
Your goal: Make it look like the meme was originally created FOR Indians, not translated FROM Western content.

Quality over speed: Face swaps are complex. Take time to match expression, lighting, and pose correctly.

Cultural sensitivity: Avoid naming controversial figures. When in doubt, use generic archetypes.

CRITICAL: You MUST generate the translated meme image with the face swap. This is not optional.
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
    """Load CELEBRITY_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'original_celebrity', 'indian_celebrity',
            'archetype', 'hindi_translation', 'rationale',
            'status', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write completed memes
        for meme_id, data in progress['completed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'original_celebrity': data.get('original_celebrity', ''),
                'indian_celebrity': data.get('indian_celebrity', ''),
                'archetype': data.get('archetype', ''),
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
                'original_celebrity': '',
                'indian_celebrity': '',
                'archetype': '',
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
        'original_celebrity': '',
        'indian_celebrity': '',
        'archetype': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract original celebrity
    orig_match = re.search(r'Original:\s*(.+?)(?:\(|$)', response_text)
    if orig_match:
        data['original_celebrity'] = orig_match.group(1).strip()
    
    # Extract Indian match
    indian_match = re.search(r'Indian Match:\s*(.+?)(?:\(|$)', response_text)
    if indian_match:
        data['indian_celebrity'] = indian_match.group(1).strip()
    
    # Extract archetype
    archetype_match = re.search(r'Archetype:\s*(.+)', response_text)
    if archetype_match:
        data['archetype'] = archetype_match.group(1).strip()
    
    # Extract selected translation
    selected_match = re.search(r'Selected:\s*["\']?(.+?)["\']?\s*(?:\n|Rationale)', response_text, re.DOTALL)
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
        meme_specific_prompt = f"""{CELEBRITY_DEPENDENT_PROMPT}

---
## 🎯 MEME TO PROCESS:

**Meme ID:** {meme_data['meme_id']}
**Extracted English Text:**
{meme_data.get('extracted_text', 'N/A')}

**Visual Description:**
{meme_data.get('visual_description', 'N/A')}

**People Identified:**
{meme_data.get('people_identified', 'N/A')}

**Requires Face Swap:**
{meme_data.get('requires_face_swap', 'N/A')}

**Humor Mechanism:**
{meme_data.get('humor_mechanism', 'N/A')}

**Existing Indian Equivalent Suggestions:**
{meme_data.get('indian_equivalent_suggestions', 'N/A')}

**Adaptation Strategy:**
{meme_data.get('adaptation_strategy', 'N/A')}

**Notes:**
{meme_data.get('notes', 'N/A')}

---
BEGIN PROCESSING NOW. Provide celebrity matching analysis first, then generate the translated image with face swap.
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
    print("🚀 Starting CELEBRITY_DEPENDENT meme translation pipeline...")
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
                'original_celebrity': result.get('original_celebrity', ''),
                'indian_celebrity': result.get('indian_celebrity', ''),
                'archetype': result.get('archetype', ''),
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
    print(f"📊 CELEBRITY_DEPENDENT TRANSLATION PROGRESS")
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
