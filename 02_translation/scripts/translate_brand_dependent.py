#!/usr/bin/env python3
"""
BRAND_DEPENDENT Meme Translation Pipeline
Translates memes with Western brands to Hindi/Hinglish with Indian brand replacements
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
SOURCE_IMAGES_DIR = "categorized_memes/BRAND_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "brand_dependent_adaptations.csv"
CHECKPOINT_FILE = "brand_dependent_progress.json"

TARGET_CATEGORY = "BRAND_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# BRAND_DEPENDENT TRANSLATION PROMPT
# ============================================================================

BRAND_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - BRAND_DEPENDENT MEME TRANSLATION PROMPT

MISSION
You are a professional Hindi translator and cultural adaptation specialist. Your task is to translate BRAND_DEPENDENT mental health memes from English to Hinglish (Hindi in Devanagari script) while replacing Western brands/products with Indian equivalents when necessary.

CATEGORY OVERVIEW: BRAND_DEPENDENT
What makes these memes different:
- Contain recognizable Western brands, products, apps, or services
- Brand recognition may be critical to understanding the joke
- Require BOTH text translation AND visual editing (in most cases)
- Need cultural judgment: Replace brand or keep it?

YOUR TASK: THREE-STEP PROCESS

STEP 1: BRAND RECOGNITION DECISION
First, determine: Should the brand be replaced or kept?

Use this decision tree:
Is the brand internationally recognized in India?
├─ YES → Ask: Is brand recognition critical to the joke?
│   ├─ NO → Keep brand, translate text only
│   └─ YES → Ask: Is there a better Indian equivalent?
│       ├─ NO → Keep brand
│       └─ YES → Replace with Indian equivalent
│
└─ NO (Western-specific) → Replace with Indian equivalent

Examples of KEEP decisions:
- Netflix → Widely known in India, keep it
- McDonald's → Present in India, keep it
- iPhone → Globally recognized, keep it

Examples of REPLACE decisions:
- Animal Crossing → Niche game, unknown in India → Replace with Biryani (comfort food)
- MyFitnessPal → Western app, unknown → Replace with "Diet App" or notebook
- Al-Anon → Western recovery program → Replace with "Self-help book"

Decision Output Format:
```
BRAND DECISION:
  Original Brand: Animal Crossing: New Horizons
  Recognition in India: Low (niche gaming product)
  Critical to joke?: Yes (entire joke is about "comfort game")
  Replace?: YES
  Replacement: Biryani (comfort food universally understood)
  Rationale: Gaming culture not widespread; food comfort is universal
```

STEP 2: TRANSLATION & CULTURAL ADAPTATION
Once you've decided on replacement, translate the text.

Translation Guidelines:

A. If KEEPING brand:
Original: "Why won't Netflix heal my depression"
Hindi: "Netflix क्यों नहीं ठीक कर सकता मेरा depression"
(Keep "Netflix" in Latin, rest in Devanagari)

B. If REPLACING brand:
Original: "Why won't Animal Crossing heal my feelings"
Option 1: "Biryani क्यों नहीं ठीक कर सकती मेरी feelings"
Option 2: "तीखी Biryani खाने के बाद भी feelings ठीक क्यों नहीं हुईं"

C. Brand-specific Hinglish rules:
- Tech (iPhone, Windows): Keep in Latin script if globally known
- Apps (Instagram, WhatsApp): Keep as-is (widely used in India)
- Food chains (McDonald's): Keep if present in India, else replace
- Games (Xbox, PlayStation): Replace with Indian leisure activity unless very popular
- Streaming (Netflix, Spotify): Keep (known in urban India)

Validation Checklist:
Humor preserved: Does the joke still work?
Natural Hinglish: Does it sound conversational?
Brand appropriate: Is replacement culturally resonant?
Devanagari accurate: Spelling correct?

STEP 3: IMAGE GENERATION WITH BRAND REPLACEMENT

Three scenarios:

SCENARIO A: Text-Only Translation (No Object Replacement)
When: Brand is mentioned in text but not shown visually, OR brand is kept as-is
Action: Keep person, translate text to Hindi
- Keep all visual elements identical
- Only change: English text → Hindi text (Devanagari)
- Position text identically
- Match font style

SCENARIO B: Object Replacement Required
When: Western product/brand visible in image AND needs to be replaced
Action: Replace object + Translate text

Object Replacement Instructions:
1. Identify the object to replace (Nintendo Switch, Intel box, etc.)
2. Determine Indian replacement (Cricket bat, Biryani plate, Chai cup, etc.)
3. Replace object while maintaining:
   Same size/scale
   Same position in hand/frame
   Same lighting/shadows
   Same perspective/angle
   Natural interaction

Object Replacement Prompt Template:
```
Generate a meme image IDENTICAL to the provided image with this change:

KEEP EXACTLY AS-IS:
- Person/character: [describe]
- Pose: [describe]
- Background: [describe]
- Lighting: [describe]
- Facial expression: [describe]

REPLACE THIS OBJECT:
- Original object: [Nintendo Switch / Intel box / etc.]
- Location: [in person's hands / on table / etc.]
- New object: [Biryani plate / Cricket bat / etc.]
- Ensure: Same size, same position, natural interaction

TEXT CHANGES:
- Original English: "[original text]"
- Hindi translation: "[validated Hindi text]"
- Position: [top/bottom/center]
- Font style: [bold/normal]
- Font color: [white/black]

CRITICAL: Everything except the replaced object and text should look identical to original.
```

Example Object Replacements:
- Nintendo Switch → Plate of Biryani (Comfort/indulgence universal)
- Intel processor box → Cricket season ball (New vs. old metaphor)
- MyFitnessPal app icon → Notebook with "Kharche" (Tracking concept)
- Xbox controller → Ludo board (Gaming → Indian game)
- Starbucks cup → Cutting chai cup (Beverage comfort)

COMPLETE OUTPUT FORMAT

For each meme, provide:

=== MEME ID: [ID] ===

STEP 1: BRAND DECISION
  Original Brand: [brand name]
  Recognition in India: [High/Medium/Low]
  Critical to joke?: [Yes/No]
  Decision: [KEEP/REPLACE]
  Replacement: [Indian equivalent if replacing]
  Rationale: [explanation]

STEP 2: TRANSLATION
  Original: "[original text]"
  Alternative 1: "[Hindi option 1]"
  Alternative 2: "[Hindi option 2]"
  Alternative 3: "[Hindi option 3]"
  Selected: "[best Hindi version]"
  Rationale: [why this works]

STEP 3: VALIDATION
  Humor preserved: [Yes/No + explanation]
  Brand appropriate: [Yes/No + explanation]
  Natural Hinglish: [Yes/No]
  Visual works: [Yes/No]

STEP 4: IMAGE GENERATION
  [Generate the translated meme with object replacement]
  
  Visual Changes:
  - Replace: [original object] → [new object]
  - Keep: [what stays the same]
  - Text: [Hindi translation in same position/style]

=== END MEME ===

CRITICAL NOTES

When to Keep Western Brands:
- Globally recognized (Netflix, Instagram, iPhone)
- Present in India (McDonald's in major cities)
- No better Indian equivalent exists
- Brand is incidental, not central to joke

When to Replace:
- Western-specific product unknown in India
- Indian equivalent is more relatable
- Brand is central to joke and has good substitute
- Replacement enhances cultural resonance

REMEMBER
Your goal: Create a meme that feels like it was made FOR Indians, not translated FROM Western content. If keeping a Western brand makes it feel foreign, replace it. If replacement feels forced, keep the original.

Quality over perfection: Some memes may work better with partial adaptation (keep brand, translate text). Use judgment.

Cultural sensitivity: Mental health memes are personal. Maintain emotional authenticity even when changing brands.
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
    """Load BRAND_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'original_brand', 'brand_decision',
            'replacement_brand', 'hindi_translation', 'rationale',
            'status', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write completed memes
        for meme_id, data in progress['completed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'original_brand': data.get('original_brand', ''),
                'brand_decision': data.get('brand_decision', ''),
                'replacement_brand': data.get('replacement_brand', ''),
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
                'original_brand': '',
                'brand_decision': '',
                'replacement_brand': '',
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
        'original_brand': '',
        'brand_decision': '',
        'replacement_brand': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract brand decision
    brand_match = re.search(r'Original Brand:\s*(.+)', response_text)
    if brand_match:
        data['original_brand'] = brand_match.group(1).strip()
    
    decision_match = re.search(r'Decision:\s*(KEEP|REPLACE)', response_text, re.IGNORECASE)
    if decision_match:
        data['brand_decision'] = decision_match.group(1).upper()
    
    replacement_match = re.search(r'Replacement:\s*(.+)', response_text)
    if replacement_match:
        data['replacement_brand'] = replacement_match.group(1).strip()
    
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
        meme_specific_prompt = f"""{BRAND_DEPENDENT_PROMPT}

---
## 🎯 MEME TO PROCESS:

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
BEGIN PROCESSING NOW. Provide analysis first, then generate the translated image with brand replacement if needed.
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
    print("🚀 Starting BRAND_DEPENDENT meme translation pipeline...")
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
                'original_brand': result.get('original_brand', ''),
                'brand_decision': result.get('brand_decision', ''),
                'replacement_brand': result.get('replacement_brand', ''),
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
    print(f"📊 BRAND_DEPENDENT TRANSLATION PROGRESS")
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
