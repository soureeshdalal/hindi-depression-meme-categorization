#!/usr/bin/env python3
"""
Category Consolidation Analysis
Merge 184 granular categories into 5-10 practical research categories
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_categories():
    """Analyze the 184 categories and group them into broader themes"""
    
    # Load the results
    results_path = Path("analysis_output/gemini_analysis_results.json")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Group categories by common patterns
    category_groups = defaultdict(list)
    
    for result in results:
        category = result.get('discovered_category', '')
        dependency = result.get('cultural_dependency_score', 0)
        difficulty = result.get('adaptation_difficulty', '')
        
        # Analyze category name patterns
        if 'Universal' in category:
            if any(word in category for word in ['Emotional', 'Emotion', 'Sentiment', 'Despair', 'Angst']):
                category_groups['Universal_Emotional'].append((category, dependency, difficulty))
            elif any(word in category for word in ['Biological', 'Physiological', 'Somatic', 'Body', 'Sleep']):
                category_groups['Universal_Biological'].append((category, dependency, difficulty))
            elif any(word in category for word in ['Behavioral', 'Behavior', 'Social', 'Workplace', 'Situational']):
                category_groups['Universal_Behavioral'].append((category, dependency, difficulty))
            elif any(word in category for word in ['Visual', 'Metaphor', 'Abstract']):
                category_groups['Universal_Visual_Metaphor'].append((category, dependency, difficulty))
            else:
                category_groups['Universal_Other'].append((category, dependency, difficulty))
        
        elif any(word in category for word in ['Celebrity', 'Pop_Culture', 'Persona', 'Icon']):
            category_groups['Celebrity_Dependent'].append((category, dependency, difficulty))
        
        elif any(word in category for word in ['Niche', 'Subculture', 'Slang', 'Acronym', 'Fandom']):
            category_groups['Subculture_Specific'].append((category, dependency, difficulty))
        
        elif any(word in category for word in ['Linguistic', 'Slang', 'Idiom', 'Language']):
            category_groups['Language_Dependent'].append((category, dependency, difficulty))
        
        elif any(word in category for word in ['Brand', 'Product', 'Consumer']):
            category_groups['Brand_Product'].append((category, dependency, difficulty))
        
        elif any(word in category for word in ['Media', 'Context', 'Format', 'Template']):
            category_groups['Format_Dependent'].append((category, dependency, difficulty))
        
        else:
            category_groups['Miscellaneous'].append((category, dependency, difficulty))
    
    return category_groups

def create_consolidated_taxonomy():
    """Create the final 5-10 category taxonomy"""
    
    category_groups = analyze_categories()
    
    print("=== CATEGORY CONSOLIDATION ANALYSIS ===\n")
    
    for group_name, categories in category_groups.items():
        print(f"{group_name}: {len(categories)} categories")
        avg_dependency = sum(dep for _, dep, _ in categories) / len(categories) if categories else 0
        difficulties = [diff for _, _, diff in categories]
        easy_count = difficulties.count('Easy')
        
        print(f"  Average Dependency: {avg_dependency:.1f}%")
        print(f"  Easy Adaptations: {easy_count}/{len(categories)} ({easy_count/len(categories)*100:.1f}%)")
        print(f"  Sample categories: {[cat[:30] + '...' if len(cat) > 30 else cat for cat, _, _ in categories[:3]]}")
        print()
    
    # Proposed consolidated taxonomy
    consolidated_taxonomy = {
        "1. UNIVERSAL_HUMAN": {
            "description": "Universal emotions, biology, and behaviors - no cultural barriers",
            "includes": ["Universal_Emotional", "Universal_Biological", "Universal_Behavioral"],
            "adaptation": "Text translation only",
            "examples": ["Depression, anxiety, sleep issues, workplace stress"]
        },
        
        "2. UNIVERSAL_VISUAL": {
            "description": "Universal concepts expressed through visual metaphors",
            "includes": ["Universal_Visual_Metaphor", "Universal_Other"],
            "adaptation": "Text translation + optional visual tweaks",
            "examples": ["Weather metaphors, abstract representations"]
        },
        
        "3. CELEBRITY_POPCULTURE": {
            "description": "Requires knowledge of Western celebrities or pop culture",
            "includes": ["Celebrity_Dependent"],
            "adaptation": "Face swap + text translation",
            "examples": ["Movie stars, musicians, TV characters"]
        },
        
        "4. LANGUAGE_SLANG": {
            "description": "Depends on English slang, idioms, or wordplay",
            "includes": ["Language_Dependent", "Subculture_Specific"],
            "adaptation": "Cultural linguistic mapping",
            "examples": ["Internet slang, English idioms, acronyms"]
        },
        
        "5. BRAND_PRODUCT": {
            "description": "Features Western brands or products",
            "includes": ["Brand_Product"],
            "adaptation": "Brand replacement + text translation",
            "examples": ["Food brands, tech products, retail chains"]
        },
        
        "6. FORMAT_CONTEXT": {
            "description": "Requires understanding specific meme formats or contexts",
            "includes": ["Format_Dependent", "Miscellaneous"],
            "adaptation": "Format explanation or replacement",
            "examples": ["Specific meme templates, UI screenshots"]
        }
    }
    
    print("\n=== PROPOSED CONSOLIDATED TAXONOMY (6 CATEGORIES) ===\n")
    
    for key, info in consolidated_taxonomy.items():
        print(f"{key}")
        print(f"  Description: {info['description']}")
        print(f"  Adaptation: {info['adaptation']}")
        print(f"  Examples: {info['examples']}")
        print()
    
    return consolidated_taxonomy

if __name__ == "__main__":
    taxonomy = create_consolidated_taxonomy()