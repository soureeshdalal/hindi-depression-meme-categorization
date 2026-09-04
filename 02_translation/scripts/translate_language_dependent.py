#!/usr/bin/env python3
"""
Language-Dependent Meme Translation Pipeline
Translates memes with English wordplay/puns to Hindi/Hinglish equivalents
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
SOURCE_IMAGES_DIR = "categorized_memes/LANGUAGE_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "language_dependent_adaptations.csv"
CHECKPOINT_FILE = "language_dependent_progress.json"

TARGET_CATEGORY = "LANGUAGE_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# TRANSLATION PROMPT
# ============================================================================

LANGUAGE_DEPENDENT_PROMPT = """# GEMINI 2.0 FLASH EXP - LANGUAGE DEPENDENT MEME TRANSCREATION PROMPT

## MISSION
You are an expert Indian Stand-up Comedian and Cultural Transcreator. You are processing memes categorized as **"LANGUAGE_DEPENDENT"**.

This means the original English humor relies on puns, wordplay, slang, or idioms that **DO NOT** make sense if translated literally.

YOUR GOAL: **Don't just translate the words. Translate the LAUGH.**

You must rewrite the text entirely to make it funny for a Hindi-speaking Gen Z/Millennial audience using the *same visual*, while rendering the final text strictly in **Devanagari script**.

---

## YOUR TASK: THE 3-STEP TRANSCREATION PROTOCOL

### **STEP 1: STRATEGIC ADAPTATION (The Brainstorm)**
You cannot do a direct translation. You must choose one of these 3 strategies to save the meme:

**Strategy A: The Idiom Swap (Best for Metaphors)**
If English uses an idiom (e.g., "Cold feet"), find the Hindi equivalent (e.g., "Haath pair thande padna").
* *English:* "When life gives you lemons..."
* *Hindi:* "Jab kismat bajaye band..." (When luck plays the band)

**Strategy B: The Visual Anchor (Best for Puns)**
Look at the image. Forget the English text. Write a NEW caption in Hindi that fits that specific facial expression or object.
* *Image:* A dog looking suspicious.
* *English Pun:* "Paw-don me?"
* *Hindi Pivot:* "Tera kutta, Tommy. Mera kutta, Kutta?" (Pop culture reference pivot)

**Strategy C: The Emotional Pivot (Best for Relatable Anxiety)**
If the wordplay is impossible, drop the pun and write a funny, relatable line about the *feeling* in the image.

**Translation Guidelines:**
- **Script:** DEVANAGARI ONLY (देवनागरी). No Latin script in the final output text.
- **Language:** Natural "Hinglish" spoken by Indian youth (e.g., "Scene sort hai", "Vibe match nahi ho rahi").
- **Tone:** Informal, conversational, slightly dramatic (filmy).

**Output Format for Step 1:**
```
ORIGINAL TEXT: "[English text]"
VISUAL CONTEXT: "[What is the image actually showing?]"
CHOSEN STRATEGY: [Idiom Swap / Visual Anchor / Emotional Pivot]
DRAFT 1 (Literal - Do not use): [Literal translation showing why it fails]
DRAFT 2 (Adaptation): [Creative Hindi option]
DRAFT 3 (Adaptation): [Creative Hindi option]
SELECTED TEXT: [Best Hindi version in Devanagari]
RATIONALE: [Why this works for the Indian context]
```

---

### **STEP 2: PRE-RENDER VALIDATION**
Before generating the image, check against these fatal errors:

1. **The "Literal Fail" Check:** Did you translate a pun literally? (e.g., Translating "I'm feeling blue" to "Main neela hoon" is a FAILURE. It must be "Mera mood kharab hai").
2. **The Script Check:** Is the selected text in Devanagari? (e.g., "Scene Sort Hai" = FAIL. "सीन सॉर्ट है" = PASS).
3. **The Vibe Check:** Is it actually funny?

**Output Format for Step 2:**
```
VALIDATION:
  Strategy used: [Strategy Name]
  Is it funny in Hindi?: Yes
  Script check: Devanagari confirmed
  FINAL TEXT FOR IMAGE: [Insert Devanagari String Here]
```

---

### **STEP 3: IMAGE GENERATION (STRICT DEVANAGARI ENFORCEMENT)**
Generate the adapted meme image.

**CRITICAL RENDERING RULES:**
1. **ZERO LATIN SCRIPT:** The text on the image must be **100% Devanagari**. Do not render "Hinglish" in English letters.
   * *Wrong:* "Bhai kya kar raha hai"
   * *Right:* "भाई क्या कर रहा है"

2. **English Exceptions:** Only maintain English text for universally recognized brands or tech terms if absolutely necessary (e.g., "Netflix", "Error 404"), but prefer Devanagari for these too if possible (e.g., "नेटफ्लिक्स").

3. **Visual Fidelity:** The image (person, background, lighting) must be identical to the original. Only the text changes.

4. **Layout:** Match the original text positioning (top text, bottom text, speech bubble).

**Image Generation Prompt Template:**
```
Generate a meme image IDENTICAL to the source image, with these changes:

VISUALS:
  Recreate the exact scene: [Insert Visual Description]
  Maintain same art style, lighting, and composition.

TEXT REPLACEMENT:
  Remove ALL English text.
  Replace with this Hindi text: "[Insert Validated Devanagari Text]"
  Font: Use a bold, clear Devanagari font (like Noto Sans Devanagari).
  Color: Match original text color.
  Position: Match original text position.
  Background: Match original text background (if any).

STRICT CONSTRAINT: The text must be rendered in HINDI SCRIPT (Devanagari). Do not use Latin alphabets.
```

---

## COMPLETE OUTPUT FORMAT
```
=== MEME ID: [ID] ===

STEP 1: TRANSCREATION ANALYSIS
  Original: "..."
  Strategy: [Visual Anchor / Idiom Swap]
  Selected Hindi: [Devanagari Text]
  Rationale: "The original pun on 'toast' doesn't work in Hindi, so I pivoted to an idiom about 'burning out' which fits the image of burnt bread."

STEP 2: VALIDATION
  Script: Devanagari
  Funny: Yes

STEP 3: IMAGE GENERATION
  [Generated Image]

=== END MEME ===
```

CRITICAL: You MUST generate the translated meme image with the Hindi text in DEVANAGARI SCRIPT overlaid on the original image, preserving all visual elements exactly."""

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
    """Load LANGUAGE_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'adaptation_type', 
            'hindi_translation', 'hinglish_translation', 
            'rationale', 'confidence_score', 'status', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write completed memes
        for meme_id, data in progress['completed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'adaptation_type': data.get('adaptation_type', ''),
                'hindi_translation': data.get('hindi_translation', ''),
                'hinglish_translation': data.get('hinglish_translation', ''),
                'rationale': data.get('rationale', ''),
                'confidence_score': data.get('confidence_score', ''),
                'status': 'SUCCESS',
                'error': ''
            })
        
        # Write failed memes
        for meme_id, data in progress['failed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'adaptation_type': '',
                'hindi_translation': '',
                'hinglish_translation': '',
                'rationale': '',
                'confidence_score': '',
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

def parse_json_from_response(response_text):
    """Extract JSON from Gemini response"""
    import re
    
    # Try to find JSON block
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None

def translate_meme_with_gemini(model, meme_data, image_path):
    """Translate meme using Gemini API"""
    try:
        # Prepare prompt with meme context
        meme_specific_prompt = f"""{LANGUAGE_DEPENDENT_PROMPT}

---
## 🎯 MEME TO PROCESS:

**Meme ID:** {meme_data['meme_id']}
**Extracted English Text:**
{meme_data.get('extracted_text', 'N/A')}

**Visual Description:**
{meme_data.get('visual_description', 'N/A')}

**Humor Mechanism:**
{meme_data.get('humor_mechanism', 'N/A')}

**Existing Translation Suggestions:**
{meme_data.get('indian_equivalent_suggestions', 'N/A')}

---
BEGIN PROCESSING NOW. Provide JSON analysis first, then generate the translated image.
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
                "temperature": 0.7,  # Higher for creative adaptations
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
        
        # Parse JSON from response
        json_data = parse_json_from_response(response_text)
        if not json_data:
            print("⚠ Could not parse JSON from response")
            json_data = {
                "original_joke_analysis": "Could not parse",
                "adaptation_type": "Unknown",
                "hindi_translation_devanagari": "",
                "hinglish_translation": "",
                "cultural_rationale": "",
                "confidence_score": "0"
            }
        
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
            return True, json_data, image_data
        else:
            print("⚠ No image generated")
            return True, json_data, None  # Still success if we got analysis
        
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
    print("🚀 Starting LANGUAGE_DEPENDENT meme translation pipeline...")
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
                'adaptation_type': result.get('adaptation_type', ''),
                'hindi_translation': result.get('hindi_translation_devanagari', ''),
                'hinglish_translation': result.get('hinglish_translation', ''),
                'rationale': result.get('cultural_rationale', ''),
                'confidence_score': result.get('confidence_score', ''),
                'full_analysis': result
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
# MAIN
# ============================================================================

if __name__ == "__main__":
    process_memes()
