#!/usr/bin/env python3
"""
UNIVERSAL_HUMAN Meme Translation Pipeline
Smart translation that skips already-translated memes.
Use --only-ids <file> to retranslate only bad images (Pillow-unreadable).
"""

import argparse
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

TARGET_CATEGORY = "UNIVERSAL_HUMAN"

CSV_PATH = "gemini_categorization/analysis_output/gemini_analysis_results.csv"
JSON_PATH = "gemini_categorization/analysis_output/gemini_analysis_results.json"
SOURCE_IMAGES_DIR = f"categorized_memes/{TARGET_CATEGORY}/"
TRANSLATED_DIR = f"translated_categorized_memes/{TARGET_CATEGORY}/"
OUTPUT_CSV = f"{TARGET_CATEGORY.lower()}_adaptations.csv"
CHECKPOINT_FILE = f"{TARGET_CATEGORY.lower()}_progress.json"

CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# TRANSLATION PROMPT
# ============================================================================

TRANSLATION_PROMPT = """# GEMINI 3 PRO IMAGE - MENTAL HEALTH MEME TRANSLATION PROMPT

## MISSION
You are a professional Hindi translator and meme adaptation specialist. Your task is to translate English mental health memes into Hinglish (Hindi written in Devanagari script with English words mixed in naturally) while preserving humor, meaning, and visual appeal.

---

## INPUT DATA YOU WILL RECEIVE
For each meme, you will receive:
```json
{
"meme_id": "TE-1",
"original_image": [image file],
"extracted_text": "I don't know how much more I can take",
"visual_description": "A person sitting in corner looking distressed",
"humor_mechanism": "Relatable despair/venting",
"indian_equivalent_suggestions": "Translate to: 'Pata nahi main aur kitna jhel paunga'",
"cultural_notes": "Universal emotion, no cultural barriers"
}
```

---

##  YOUR TASK: COMPLETE MEME TRANSLATION + IMAGE GENERATION
You must perform **THREE STEPS** for each meme:

### **STEP 1: TRANSLATION & CULTURAL ADAPTATION**
Translate the English text to **Hinglish in Devanagari script** using:

**Translation Guidelines:**
- **Style:** Natural, conversational Hinglish (how urban Indian youth actually speak)
- **Script:** Devanagari (देवनागरी) 
- **English words:** Keep common English words that Indians use naturally (e.g., "depressed", "therapy", "anxiety" are often used in English even in Hindi conversations)
- **Cultural adaptation:** Moderate level
- Change Western names to Indian names (Sam → Sameer, Karen → Kavita)
- Adapt idioms to Indian equivalents
- Keep technical mental health terms in English if commonly used that way

**Hinglish Balance Examples:**

**Too Much Hindi (Avoid):**
"मुझे नहीं पता कि मैं और कितनी पीड़ा सहन कर सकता हूँ"

**Good Hinglish (Target):**
"पता नहीं मैं और कितना झेल पाऊँगा"

**Good Hinglish with English Words:**
"Therapy के बाद भी depression कम नहीं हो रहा"

**Translation Process:**
1. **Review existing suggestion** from `indian_equivalent_suggestions`
2. **Analyze if it's good Hinglish** (natural, not too formal)
3. **Check cultural adaptation** (names, idioms, context)
4. **Generate 2-3 alternative translations**
5. **Pick the best one** that:
- Sounds natural to Indian youth
- Preserves the humor mechanism
- Fits the image space (not too long)
- Maintains emotional tone

**Output Format for Step 1:**
```
ORIGINAL TEXT: "I don't know how much more I can take"
EXISTING SUGGESTION: "Pata nahi main aur kitna jhel paunga"
ALTERNATIVE 1: पता नहीं मैं और कितना झेल पाऊँगा
ALTERNATIVE 2: अब मुझसे और नहीं सहा जाता
ALTERNATIVE 3: कब तक यूँ ही चलता रहेगा
SELECTED TRANSLATION: पता नहीं मैं और कितना झेल पाऊँगा
RATIONALE: Most natural, preserves desperation tone, commonly used phrase
```

---

### **STEP 2: SELF-VALIDATION (Critical!)**
Before creating the image, validate your translation:

**Validation Checklist:**

**Humor Preserved?**
- Does the translated text still evoke the same emotional response?
- Is the self-deprecating/relatable humor intact?
- Would an Indian reading this find it funny/relatable for the same reasons?

**Natural Language?**
- Does it sound like how Indians actually talk?
- Is the Hinglish balance appropriate?
- Are there awkward direct translations that need fixing?

**Cultural Fit?**
- Have Western references been adapted?
- Are idioms culturally appropriate?
- Would an Indian with zero Western knowledge understand this?

**Technical Accuracy?**
- Is the Devanagari spelling correct?
- Are matras (diacritics) placed properly?
- Is punctuation appropriate?

**Output Format for Step 2:**
```
VALIDATION CHECK:
Humor preserved: Yes - desperation tone maintained
Natural Hinglish: Yes - commonly used phrase
Cultural fit: Yes - universal emotion
Devanagari accurate: Yes - checked spelling
Potential issue: Text may be slightly long for image space
Adjustment: None needed / [describe fix]
FINAL VALIDATED TRANSLATION: पता नहीं मैं और कितना झेल पाऊँगा
```

---

### **STEP 3: IMAGE GENERATION WITH HINDI TEXT**
Now create the translated meme image. You must:

**Image Generation Instructions:**
1. **Analyze original image layout:**
- Where is the text positioned? (top, bottom, center, overlay)
- What is the text style? (bold, italic, size, color, font)
- What is the background? (plain, image overlay, text box)
- Are there multiple text elements? (speech bubbles, captions, labels)

2. **Recreate the image identically with Hindi text:**
- **Keep ALL visual elements exactly the same** (person, pose, colors, lighting, background)
- **Remove English text completely**
- **Add Hindi text in the EXACT SAME position and style**
- **Match font style** (bold = bold, caps = caps, etc.)
- **Match text size** relative to image
- **Match text color** (white on black, black on white, etc.)
- **Preserve text formatting** (line breaks, alignment, spacing)

3. **Text rendering guidelines:**
- Use clear, readable Devanagari fonts (like Noto Sans Devanagari, Mangal, or similar)
- Ensure proper character spacing for Devanagari
- Maintain text hierarchy if multiple text elements exist
- If text is in a box/banner, preserve that box style
- If text has background (white box, black banner), keep it identical

4. **Quality requirements:**
- **Resolution:** Match original image resolution
- **Aspect ratio:** Preserve original aspect ratio
- **Compression:** High quality, no visible artifacts
- **Text clarity:** Devanagari text must be sharp and readable
- **Visual fidelity:** 95%+ match to original image (except text)

**Common Meme Text Layouts:**

**Layout Type 1: Top-Bottom Captions**
```
[White text at top]
[IMAGE]
[White text at bottom]
```
→ Recreate with Hindi text in same positions

**Layout Type 2: Overlay Text**
```
[IMAGE with text overlaid directly on it]
```
→ Position Hindi text in same location with same background

**Layout Type 3: Speech Bubbles**
```
[IMAGE with speech bubble containing text]
```
→ Preserve bubble, change text inside to Hindi

**Layout Type 4: Multi-Panel**
```
[Panel 1: Image + Text] [Panel 2: Image + Text]
```
→ Translate each panel's text independently

**Image Generation Prompt Template:**
```
Generate a meme image that is IDENTICAL to the provided image, with these specifications:

VISUAL ELEMENTS (Keep exactly as original):
- Subject: [describe person/character/scene from visual_description]
- Background: [describe background]
- Colors: [describe color scheme]
- Lighting: [describe lighting/mood]
- Composition: [describe framing/layout]

TEXT ELEMENTS (Replace with Hindi):
- Position: [top/bottom/center/overlay/speech bubble]
- Original English: "[original text]"
- Hindi translation: "[validated Hindi text]"
- Font style: [bold/normal/italic]
- Font size: [large/medium/small relative to image]
- Font color: [white/black/colored]
- Text background: [none/black banner/white box/transparent overlay]
- Text alignment: [center/left/right]

CRITICAL: The image must look EXACTLY like the original meme except the text is in Hindi (Devanagari script). All visual elements, colors, composition, and styling must be identical.
```

---

##  COMPLETE OUTPUT FORMAT
For each meme, provide:
```
=== MEME ID: TE-1 ===

STEP 1: TRANSLATION
Original: "I don't know how much more I can take"
Suggestion: "Pata nahi main aur kitna jhel paunga"
Alternative 1: पता नहीं मैं और कितना झेल पाऊँगा
Alternative 2: अब मुझसे और नहीं सहा जाता
Alternative 3: कब तक यूँ ही चलता रहेगा
Selected: पता नहीं मैं और कितना झेल पाऊँगा
Rationale: Most natural, preserves desperation tone

STEP 2: VALIDATION
Humor preserved: Yes
Natural Hinglish: Yes  
Cultural fit: Yes
Devanagari accurate: Yes
Final Translation: पता नहीं मैं और कितना झेल पाऊँगा

STEP 3: IMAGE GENERATION
[Generate the translated meme image according to specifications above]

=== END MEME TE-1 ===
```

---

## SPECIAL CASES & EDGE CASES

### **Case 1: Text is Too Long for Space**
If Hindi translation is longer than English (common!):

**Option A:** Abbreviate naturally
```
English: "I don't know how much more I can take"
Hindi: मैं और कितना झेल पाऊँगा (dropped "pata nahi" to fit)
```

**Option B:** Use line breaks
```
पता नहीं 
मैं और कितना झेल पाऊँगा
```

**Option C:** Reduce font size slightly (last resort)

### **Case 2: Multiple Text Elements**
Translate ALL text elements:
- Captions
- Speech bubbles  
- Labels
- Watermarks (keep as-is unless they're part of the joke)

### **Case 3: Text Embedded in Image (Hard to Remove)**
If text is artistically integrated:
- Try to recreate the artistic style with Hindi
- If impossible, flag for manual review
- Provide text-only translation as fallback

### **Case 4: English Words That Should Stay English**
Keep these in English even in Hinglish:
- Brand names (unless adapting): Instagram, Facebook
- Technical terms widely used in English: "therapy", "anxiety", "depression" (but write in Devanagari if context requires)
- Internet slang that's universally understood: "DM", "selfie"

**Example:**
```
Original: "My therapist when I say I'm fine"
Hinglish: मेरे therapist जब मैं कहता हूँ I'm fine
OR: मेरा थेरेपिस्ट जब मैं कहता हूँ मैं ठीक हूँ
(Both acceptable, pick based on context)
```

### **Case 5: Idioms and Metaphors**

**Direct translation if idiom exists in Hindi:**
```
"I'm losing my mind" → "मेरा दिमाग खराब हो रहा है" 
```

**Cultural adaptation if idiom doesn't translate:**
```
"I'm at the end of my rope" → "अब मुझसे नहीं होगा" (Literal translation would be confusing)
```

### **Case 6: Names and Cultural References**

**Moderate adaptation (as per your requirement):**

Names:
- Sam → समीर (Sameer)
- Karen → कविता (Kavita)  
- John → जॉन (John) or राहुल (Rahul)

Western contexts:
- "Thanksgiving dinner" → "family gathering" or "छुट्टी का खाना"
- "High school" → "school" or "कॉलेज"
- "Subway" → "Metro" if in Delhi/Mumbai context

**BUT don't over-adapt:**
- Keep mental health terms: anxiety, depression, OCD (in Devanagari if needed)
- Keep universally known brands: Netflix, Instagram
- Keep locations if they're just examples: "New York" can stay unless it's central to joke

---

## FONT & STYLING SPECIFICATIONS

**Devanagari Font Recommendations:**
- Noto Sans Devanagari (clean, modern, readable)
- Mangal (widely used, good for memes)
- Lohit Devanagari (clear, works well at small sizes)

**Styling Guidelines:**
1. **If original is BOLD → Hindi is BOLD**
2. **If original is ALL CAPS → Hindi is... normal (Devanagari doesn't have caps)**
3. **If original has shadow/outline → Add shadow/outline to Hindi**
4. **If original has color gradient → Match gradient on Hindi**

**Text Size Hierarchy:**
- Main caption: Largest
- Secondary text: Medium
- Fine print: Small
- Maintain these relationships in Hindi version

---

## FINAL QUALITY CHECKLIST
Before outputting the image, verify:
- Hindi text is in Devanagari script (देवनागरी)
- All English text has been removed/replaced
-  Text position matches original exactly
-Font style (bold/normal) matches original
-  Text size is proportional to image
-  Text color matches original
-  Background/overlay style preserved
-  Image resolution matches original
- Visual elements unchanged (same person, same pose, same colors)
- Humor/meaning preserved in translation
-  Hinglish sounds natural
-  No spelling errors in Devanagari
- Text is readable (not too small, not blurry)

---

## EXECUTION MODE
You will receive ONE meme at a time with all the data. **Your output should be:**
1. The translation analysis (Step 1 & 2 text)
2. The generated translated meme image (Step 3)

**Be concise in your analysis but thorough in your validation.**

---

## REMEMBER
**Your goal:** Create a meme that looks and feels identical to the original, but in Hinglish that resonates with Indian youth dealing with mental health issues. The translation should feel natural, the humor should be preserved, and the image should be indistinguishable from a professional Indian meme maker's work.

**Quality over speed:** Take time to get the translation right. A bad translation ruins a good meme.

**Cultural sensitivity:** Mental health memes are personal. Maintain the emotional tone and don't trivialize the struggles being expressed.

---

## YOU ARE READY
Process memes one at a time using the three-step process. Focus on creating high-quality, culturally resonant Hindi memes that maintain the authenticity and relatability of the originals.
"""

# ============================================================================
# SMART TRANSLATION LOGIC
# ============================================================================

def get_already_translated_meme_ids():
    """Get list of meme IDs that are already translated"""
    translated_ids = set()
    
    test_path = os.path.join(TRANSLATED_DIR, 'test')
    if os.path.exists(test_path):
        for filename in os.listdir(test_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                meme_id = os.path.splitext(filename)[0]
                translated_ids.add(meme_id)
    
    val_path = os.path.join(TRANSLATED_DIR, 'validation')
    if os.path.exists(val_path):
        for filename in os.listdir(val_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                meme_id = os.path.splitext(filename)[0]
                translated_ids.add(meme_id)
    
    return translated_ids

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
            "failed_memes": {},
            "skipped_already_translated": []
        }

def save_progress(progress):
    """Save progress to checkpoint file"""
    progress["last_checkpoint"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    print(f"✅ Checkpoint saved: {progress['total_processed']}/{progress['total_to_process']} completed")

def _row_from_json_item(item):
    """Convert one JSON result item to a flat dict like CSV DictReader."""
    suggestions = item.get("indian_equivalent_suggestions") or []
    if isinstance(suggestions, list):
        indian_equivalent_suggestions = "\n".join(str(s) for s in suggestions)
    else:
        indian_equivalent_suggestions = str(suggestions)
    return {
        "meme_id": str(item.get("meme_id", "")).strip(),
        "discovered_category": str(item.get("discovered_category", "")).strip(),
        "extracted_text": str(item.get("extracted_text", "")),
        "visual_description": str(item.get("visual_description", "")),
        "cultural_elements": str(item.get("cultural_elements", "")),
        "humor_mechanism": str(item.get("humor_mechanism", "")),
        "indian_equivalent_suggestions": indian_equivalent_suggestions,
        "adaptation_strategy": str(item.get("adaptation_strategy", "")),
    }


def _load_all_universal_memes_from_source(only_ids=None):
    """Load UNIVERSAL_HUMAN meme rows from CSV or JSON. If only_ids is set, include any meme in that set (for retranslation). Returns list of dicts."""
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row.get("discovered_category", "").strip() == TARGET_CATEGORY]
        if only_ids:
            rows = [r for r in rows if r.get("meme_id", "").strip() in only_ids]
        return rows
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results") or (data if isinstance(data, list) else [])
        if only_ids:
            # For retranslation: take any result with this meme_id (prefer UNIVERSAL_HUMAN)
            by_id = {}
            for item in results:
                mid = str(item.get("meme_id", "")).strip()
                if mid not in only_ids:
                    continue
                if mid not in by_id or str(item.get("discovered_category", "")).strip() == TARGET_CATEGORY:
                    by_id[mid] = item
            return [_row_from_json_item(by_id[mid]) for mid in only_ids if mid in by_id]
        return [_row_from_json_item(item) for item in results if str(item.get("discovered_category", "")).strip() == TARGET_CATEGORY]
    raise FileNotFoundError(f"Neither {CSV_PATH} nor {JSON_PATH} found.")


def load_memes_only_ids(only_ids):
    """Load meme rows from CSV/JSON for given IDs only (retranslation of bad images). Overwrites existing output."""
    print("📖 Loading memes (only-ids mode)...")
    only_ids = {x.strip() for x in only_ids if x.strip()}
    memes_data = _load_all_universal_memes_from_source(only_ids=only_ids)
    found_ids = {m.get("meme_id", "").strip() for m in memes_data}
    missing = only_ids - found_ids
    if missing:
        print(f"⚠️  {len(missing)} IDs not in source: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"⏳ Retranslating {len(memes_data)} memes (will overwrite existing files)")
    return memes_data


def load_memes_to_process(progress, already_translated_ids):
    """Load memes from CSV/JSON, skip already translated"""
    print("📖 Loading memes...")
    memes_data = _load_all_universal_memes_from_source()
    print(f"📊 Total {TARGET_CATEGORY} memes in source: {len(memes_data)}")
    
    memes_to_process = []
    skipped_translated = []
    
    for meme in memes_data:
        meme_id = meme.get('meme_id', '').strip()
        if not meme_id:
            continue
        
        if meme_id in already_translated_ids:
            skipped_translated.append(meme_id)
            continue
        
        if meme_id in progress["completed_memes"] or meme_id in progress["failed_memes"]:
            continue
        
        memes_to_process.append(meme)
    
    print(f"✅ Already translated (skipping): {len(skipped_translated)}")
    print(f"✅ Already processed in this run: {len(progress['completed_memes'])}")
    print(f"⏳ Remaining to process: {len(memes_to_process)}")
    
    progress['skipped_already_translated'] = skipped_translated
    progress['total_to_process'] = len(memes_data)
    
    return memes_to_process

def setup_output_directories():
    """Create output directory structure"""
    print("📁 Setting up output directory structure...")
    
    output_path = Path(TRANSLATED_DIR)
    (output_path / "test").mkdir(parents=True, exist_ok=True)
    (output_path / "validation").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Output directory ready: {output_path}")
    return output_path

def setup_gemini_api():
    """Initialize Gemini API"""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model

def translate_meme_with_gemini(model, meme_data, image_path):
    """Translate meme using Gemini API"""
    try:
        meme_specific_prompt = f"""{TRANSLATION_PROMPT}

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

**Indian Equivalent Suggestions:**
{meme_data.get('indian_equivalent_suggestions', 'N/A')}

**Adaptation Strategy:**
{meme_data.get('adaptation_strategy', 'N/A')}

---
BEGIN PROCESSING NOW. Generate the translated image with Devanagari script.
"""

        print(f"📡 Calling Gemini {GEMINI_MODEL} for {meme_data['meme_id']}...")
        
        image = Image.open(image_path)
        print(f"✓ Image loaded: {image.size}")
        
        print("⟳ Sending to Gemini API...")
        response = model.generate_content(
            [meme_specific_prompt, image],
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 4096,
            }
        )
        
        print("✓ Response received!")
        
        image_data = None
        if hasattr(response, 'parts'):
            for i, part in enumerate(response.parts):
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    print(f"✓ Found image in part {i}")
                    break
        
        if image_data:
            return True, {}, image_data
        else:
            print("⚠ No image generated")
            return True, {}, None
        
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
        
        if isinstance(image_data, bytes):
            with open(output_path, 'wb') as f:
                f.write(image_data)
            print(f"✅ Saved binary data")
            return True
        
        print(f"❌ Unknown image data type: {type(image_data)}")
        return False
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return False

def process_memes(only_ids_file=None):
    """Main processing loop. If only_ids_file is set, only process those meme IDs (retranslate bad images)."""
    print(f"🚀 Starting {TARGET_CATEGORY} meme translation pipeline...")
    print("=" * 80)
    
    progress = load_progress()
    category_output_path = setup_output_directories()

    if only_ids_file:
        only_ids = set(Path(only_ids_file).read_text(encoding="utf-8").strip().splitlines())
        only_ids.discard("")
        memes_to_process = load_memes_only_ids(only_ids)
        total_to_process = len(memes_to_process)
    else:
        print(f"📂 Loaded progress: {progress['total_processed']} memes already completed")
        already_translated_ids = get_already_translated_meme_ids()
        print(f"🔍 Found {len(already_translated_ids)} already translated memes in output folder")
        memes_to_process = load_memes_to_process(progress, already_translated_ids)
        total_to_process = progress['total_to_process']
    
    if len(memes_to_process) == 0:
        print("✅ Nothing to process (all done or no IDs in list).")
        return
    
    model = setup_gemini_api()
    
    processed_since_checkpoint = 0
    
    for idx, meme_data in enumerate(memes_to_process):
        meme_id = meme_data.get('meme_id', '').strip()
        
        print(f"\n{'='*80}")
        print(f"🎯 Processing: {meme_id} ({idx+1}/{len(memes_to_process)})")
        print(f"{'='*80}")
        
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
        
        success, result, image_data = translate_meme_with_gemini(model, meme_data, source_image_path)
        
        if success:
            output_filename = f"{meme_id}.jpg"
            output_path = category_output_path / output_subfolder / output_filename
            
            image_saved = False
            if image_data:
                image_saved = save_translated_image(image_data, output_path)
            
            progress['completed_memes'][meme_id] = {
                'status': 'completed',
                'output_path': str(output_path) if image_saved else 'NO_IMAGE',
                'timestamp': datetime.now().isoformat(),
                'original_text': meme_data.get('extracted_text', '')
            }
            progress['total_processed'] += 1
            processed_since_checkpoint += 1
            
            print(f"✅ Success! " + ("Image saved" if image_saved else "(no image)"))
            
            if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
                save_progress(progress)
                processed_since_checkpoint = 0
        else:
            error_msg = result
            
            if error_msg == "QUOTA_EXCEEDED":
                print(f"❌ API QUOTA EXCEEDED - Stopping")
                progress['failed_memes'][meme_id] = {
                    'status': 'failed',
                    'error': 'API quota exceeded',
                    'timestamp': datetime.now().isoformat()
                }
                save_progress(progress)
                return
            
            progress['failed_memes'][meme_id] = {
                'status': 'failed',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Error: {error_msg}")
        
        time.sleep(BATCH_DELAY)
    
    if not only_ids_file:
        save_progress(progress)
    
    print(f"\n{'='*80}")
    print(f"🎉 PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Processed: {len(memes_to_process)} memes")
    if not only_ids_file:
        print(f"✅ Total in progress: {progress['total_processed']}/{progress['total_to_process']}")
        print(f"⏭️  Skipped (already translated): {len(progress.get('skipped_already_translated', []))}")
    print(f"❌ Failed this run: {len(progress['failed_memes'])}")
    print(f"📁 Output directory: {category_output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNIVERSAL_HUMAN translation pipeline. Use --only-ids to retranslate only bad (Pillow-unreadable) images.")
    parser.add_argument("--only-ids", type=str, default=None, metavar="FILE", help="Path to file with meme IDs to process (one per line). From find_bad_universal_human_images.py.")
    args = parser.parse_args()
    process_memes(only_ids_file=args.only_ids)
