#!/usr/bin/env python3
"""
CULTURAL_REFERENCE_DEPENDENT Meme Translation Pipeline
Translates memes by replacing Western cultural references with Indian equivalents
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
SOURCE_IMAGES_DIR = "categorized_memes/CULTURAL_REFERENCE_DEPENDENT/"
OUTPUT_DIR = "translated_categorized_memes/"
OUTPUT_CSV = "cultural_reference_dependent_adaptations.csv"
CHECKPOINT_FILE = "cultural_reference_dependent_progress.json"

TARGET_CATEGORY = "CULTURAL_REFERENCE_DEPENDENT"
CHECKPOINT_INTERVAL = 5
BATCH_DELAY = 2

# ============================================================================
# CULTURAL_REFERENCE_DEPENDENT TRANSLATION PROMPT
# ============================================================================

CULTURAL_REFERENCE_DEPENDENT_PROMPT = """# GEMINI 3 PRO IMAGE - CULTURAL_REFERENCE_DEPENDENT MEME TRANSLATION PROMPT

## MISSION
Translate **CULTURAL_REFERENCE_DEPENDENT** mental health memes by replacing Western cultural references (holidays, history, institutions, phrases) with Indian equivalents using **Hinglish in DEVANAGARI SCRIPT (देवनागरी)**.

---

## CRITICAL REQUIREMENTS (READ THIS FIRST)

### **ABSOLUTE REQUIREMENT #1: DEVANAGARI SCRIPT ONLY**

**ALL Hindi text MUST be in Devanagari script (देवनागरी), NOT Romanized.**

**WRONG:** "Picture it, Mumbai, 1990"
**CORRECT:** "सोचो यह हुआ था, Mumbai, 1990"

---

### **ABSOLUTE REQUIREMENT #2: HINGLISH STYLE**

**CORRECT:**
- "सोचो यह हुआ था Mumbai में..."
- "Sharma ji का बेटा IIT में है"
- "FBI नहीं, Mohalla aunties देख रही हैं"

---

### **ABSOLUTE REQUIREMENT #3: FULLY EDITED MEME IMAGE**

Generate complete meme with Western references replaced by Indian equivalents.

---

## INPUT DATA

```json
{
  "meme_id": "TE-141",
  "extracted_text": "Picture it, Sicily, 1912...",
  "cultural_elements": "Golden Girls catchphrase; Sicily; US TV show",
  "indian_equivalent_suggestions": "Bollywood Dadi character; 'Suno yeh hua tha...'",
  "requires_cultural_reconceptualization": "Yes"
}
```

---

## CULTURAL REFERENCE REPLACEMENT GUIDE

### **WESTERN INSTITUTIONS → INDIAN EQUIVALENTS**

| Western | Indian Replacement | Devanagari |
|---------|-------------------|-----------|
| Harvard | IIT (Indian Institute of Technology) | "IIT" |
| NASA | ISRO or "NASA" (known in India) | "NASA" or "ISRO" |
| FBI | CBI, Jio surveillance, Mohalla aunties | "CBI" or "Jio" |

---

### **WESTERN CULTURAL PHRASES → INDIAN EQUIVALENTS**

| Western | Context | Indian Replacement | Devanagari |
|---------|---------|-------------------|-----------|
| "Picture it, Sicily, 1912" | Golden Girls storytelling | "सोचो यह हुआ था [place], [year]" | सोचो यह हुआ था... |
| "Sound of Silence" | Simon & Garfunkel | "Dost dost na raha" (Bollywood) | दोस्त दोस्त न रहा |
| "Let people enjoy things" | Western discourse | "जो मर्ज़ी करो, हमें क्या" | जो मर्ज़ी करो... |
| "Roaring Twenties" | 1920s prosperity | "धमाकेदार शुरुआत" or Bollywood analogy | धमाकेदार शुरुआत |

---

### **WESTERN POP CULTURE → INDIAN EQUIVALENTS**

| Western | Indian | Devanagari Example |
|---------|--------|-------------------|
| Harry Potter | Cricket, Bigg Boss, Bollywood | "Cricket" or "Bigg Boss" |
| Anime girls | Instagram Reels, Cricket highlights | "Instagram Reels" |
| Hyphy music | Punjabi wedding music, Bhojpuri DJ | "Punjabi गाने" |
| Wine culture | Desi daru, Chai | "देसी दारू" or "चाय" |

---

### **WESTERN STEREOTYPES → INDIAN EQUIVALENTS**

| Western | Indian | Devanagari |
|---------|--------|-----------|
| "Kids in China making iPhones" | "Sharma ji ka beta" (overachiever trope) | "Sharma जी का बेटा coding कर रहा है" |
| FBI surveillance | Jio tracking, Nosy neighbors | "Jio देख रहा है" or "Aunties देख रही हैं" |

---

## THREE-STEP PROCESS

### **STEP 1: IDENTIFY & REPLACE CULTURAL REFERENCE**

**Analysis:**
```
Western Reference: "Picture it, Sicily, 1912" (Golden Girls)
Type: TV show catchphrase
Role: Nostalgic storytelling opener
Recognition in India: Very Low (niche US TV)

Indian Replacement: Bollywood Dadi storytelling
Phrase: "सोचो यह हुआ था..." (Think, this happened...)
Why: Bollywood grandmothers tell stories this way
Recognition: High (universal Indian storytelling)
```

---

### **STEP 2: TRANSLATION IN DEVANAGARI**

**Original:** "Picture it, Sicily, 1912"

**Devanagari Options:**
1. "सोचो यह हुआ था, Mumbai, 1990"
2. "सुनो यह कहानी, गाँव में, बहुत पहले"
3. "याद है वो दिन, जब..."

**Selected:** "सोचो यह हुआ था, Mumbai, 1990"
**Rationale:** Direct parallel to original structure, Mumbai familiar

---

### **STEP 3: VALIDATION**

```
DEVANAGARI: सोचो यह हुआ था... (देवनागरी)
Cultural replacement: Golden Girls → Dadi storytelling
Humor preserved: Nostalgic deflection tactic
Hinglish natural: सोचो + Mumbai mix
Recognition: High (everyone knows Dadi stories)
```

---

## RECONCEPTUALIZATION EXAMPLES

### **Example 1: Golden Girls → Bollywood Dadi**

**Original:**
- Text: "Picture it, Sicily, 1912"
- Visual: Old woman telling story

**Indian:**
- Text: "सोचो यह हुआ था, Mumbai, 1990"
- Visual: Indian grandmother (Dadi) in saree
- Context: Bollywood-style storytelling

---

### **Example 2: FBI Agent → Jio/Aunties**

**Original:**
- Text: "My FBI agent watching me google..."
- Visual: Surveillance imagery

**Indian:**
- Text: "Jio मुझे देख रहा है जब मैं Google करता हूँ..." OR
         "Mohalla aunties देख रही हैं..."
- Visual: Keep surveillance theme or nosy neighbors
- Context: Jio surveillance jokes OR Indian neighbor culture

---

### **Example 3: Harvard → IIT**

**Original:**
- Text: "Harvard wants to know your location"

**Indian:**
- Text: "IIT तुम्हें ढूंढ रहा है"
- Visual: Keep as-is (concept translates)
- Context: IIT = Indian Harvard

---

### **Example 4: Sharma Ji Ka Beta**

**Original:**
- Text: "Kids your age making iPhones in China"

**Indian:**
- Text: "Sharma जी का बेटा तुम्हारी उम्र में coding कर रहा था"
- Visual: Keep parent scolding child
- Context: Universal Indian parental comparison trope

---

## VALIDATION CHECKLIST

```
DEVANAGARI: All Hindi in देवनागरी (not Romanized)
Cultural reference: Western → Indian equivalent
Recognition: 70%+ Indians understand replacement
Humor: Preserved (same emotional impact)
Hinglish: Natural mix
Context: Makes sense in Indian setting
Matras: Properly rendered (ा, ि, ी, ु, ू)
```

---

## IMAGE GENERATION

**Scenario A: Text Replacement Only**
```
Keep visual → Replace text in Devanagari
Example: "Harvard" → "IIT तुम्हें ढूंढ रहा है"
```

**Scenario B: Character Replacement**
```
Replace Western character with Indian
Example: Golden Girls → Bollywood Dadi in saree
```

**Scenario C: Complete Reconceptualization**
```
Rebuild entire cultural context
Example: Roaring Twenties → Bollywood movie structure metaphor
```

---

## FONT SPECIFICATIONS

**Devanagari Fonts:**
- Noto Sans Devanagari (primary)
- Mangal (alternative)

**Rendering:**
- Proper matras (ा, ि, ी, ु, ू, े, ै, ो, ौ)
- Sharp, readable text
- NO Romanization

---

## OUTPUT FORMAT

```
=== MEME ID: TE-141 ===

STEP 1: CULTURAL REFERENCE REPLACEMENT
Original: "Picture it, Sicily, 1912" (Golden Girls)
Indian: Bollywood Dadi storytelling → "सोचो यह हुआ था..."
Recognition: High (universal storytelling)

STEP 2: DEVANAGARI TRANSLATION
Original: "Picture it, Sicily, 1912"
Hindi: "सोचो यह हुआ था, Mumbai, 1990"
Rationale: Direct parallel, familiar location

STEP 3: VALIDATION
Devanagari: सोचो यह हुआ था...
Cultural: Golden Girls → Dadi 
Humor: Deflection tactic preserved 
Recognition: High 

IMAGE GENERATION:
[Generate Dadi character with Devanagari text]

TEXT VERIFICATION:
Devanagari confirmed (सोचो...)
NO Romanization

=== END TE-141 ===
```

---

## QUICK REFERENCE - ALL 9 MEMES

| Meme | Western Reference | Indian Replacement | Devanagari Key Phrase |
|------|------------------|-------------------|---------------------|
| TE-141 | Picture it, Sicily | सोचो यह हुआ था | सोचो यह हुआ था... |
| TE-147 | FBI agent | Jio/Mohalla aunties | Jio देख रहा है |
| TE-160 | Anime girls | Instagram Reels | Instagram Reels |
| TE-232 | Sound of Silence | Dost dost na raha | दोस्त दोस्त न रहा |
| TE-411 | Harvard | IIT | IIT तुम्हें ढूंढ रहा है |
| TE-506 | Roaring Twenties | धमाकेदार शुरुआत | धमाकेदार शुरुआत |
| TE-533 | China iPhones | Sharma ji ka beta | Sharma जी का बेटा |
| TE-70 | Hyphy music | Punjabi wedding songs | Punjabi गाने |
| TE-74 | Harry Potter | Cricket/Bigg Boss | Cricket या Bigg Boss |

---

## DEVANAGARI ENFORCEMENT

**BEFORE generating:**
1. Is translation in Devanagari? (सोचो  vs "Socho" )
2. Can see Hindi characters (स, ो, च)?
3. NO Romanization anywhere?

**AFTER generating:**
1. Look at image - text in Devanagari?
2. See Hindi script (not Latin)?
3. Verify: सोचो यह हुआ था not "Socho yeh hua tha"

---

## FINAL CHECKLIST

- [ ] Hindi in Devanagari (देवनागरी) ← **CRITICAL**
- [ ] NO Romanization (no "socho", "ki", "ka")
- [ ] Western reference replaced (Harvard→IIT, FBI→Jio)
- [ ] Indian replacement recognizable (70%+ know it)
- [ ] Humor preserved
- [ ] Hinglish natural
- [ ] Matras correct (ा, ि, ी, ु, ू)
- [ ] Cultural context makes sense

---

## REMEMBER

**Goal:** Replace Western cultural touchpoints with Indian equivalents while maintaining humor and using Devanagari script.

**Priority order:**
1. Devanagari script (non-negotiable)
2. Indian cultural equivalent (universally known)
3. Preserved humor (same emotional impact)

---

## EXECUTION

Process 9 CULTURAL_REFERENCE_DEPENDENT memes:
1. **Identify** Western cultural reference
2. **Replace** with Indian equivalent
3. **Translate** in Devanagari
4. **Validate** script + cultural fit + humor

**Every meme must use देवनागरी, not Romanization.**

Good luck! """

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
    """Load CULTURAL_REFERENCE_DEPENDENT memes from CSV"""
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
            'meme_id', 'original_text', 'western_reference', 'indian_replacement',
            'hindi_translation', 'rationale', 'status', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write completed memes
        for meme_id, data in progress['completed_memes'].items():
            writer.writerow({
                'meme_id': meme_id,
                'original_text': data.get('original_text', ''),
                'western_reference': data.get('western_reference', ''),
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
                'western_reference': '',
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
        'western_reference': '',
        'indian_replacement': '',
        'hindi_translation': '',
        'rationale': ''
    }
    
    # Extract western reference
    west_match = re.search(r'Western Reference:\s*(.+)', response_text)
    if west_match:
        data['western_reference'] = west_match.group(1).strip()
    
    # Extract Indian replacement
    indian_match = re.search(r'Indian Replacement:\s*(.+)', response_text)
    if indian_match:
        data['indian_replacement'] = indian_match.group(1).strip()
    
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
        meme_specific_prompt = f"""{CULTURAL_REFERENCE_DEPENDENT_PROMPT}

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
BEGIN PROCESSING NOW. Provide cultural reference analysis first, then generate the translated image with Indian cultural references in Devanagari script.
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
    print("🚀 Starting CULTURAL_REFERENCE_DEPENDENT meme translation pipeline...")
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
                'western_reference': result.get('western_reference', ''),
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
    print(f"📊 CULTURAL_REFERENCE_DEPENDENT TRANSLATION PROGRESS")
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
