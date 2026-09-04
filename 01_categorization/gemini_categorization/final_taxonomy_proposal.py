#!/usr/bin/env python3
"""
Final 5-Category Taxonomy for Mental Health Meme Cultural Adaptation
Based on analysis of 198 memes with 184 granular categories
"""

def create_final_taxonomy():
    """Create the final 5-category taxonomy with statistics"""
    
    # Based on the consolidation analysis
    final_taxonomy = {
        "1. UNIVERSAL": {
            "description": "Universal human experiences - emotions, biology, behaviors",
            "subcategories": [
                "Universal_Emotional (31 categories)",
                "Universal_Biological (14 categories)", 
                "Universal_Behavioral (11 categories)",
                "Universal_Visual_Metaphor (23 categories)",
                "Universal_Other (52 categories)"
            ],
            "total_memes": 131,  # 31+14+11+23+52
            "percentage": "66.2%",
            "avg_dependency": "13.4%",
            "easy_adaptations": "96.2%",
            "adaptation_strategy": "Text translation only",
            "examples": [
                "Depression, anxiety, sleep problems",
                "Body image, eating disorders", 
                "Workplace stress, social anxiety",
                "Weather metaphors, abstract visuals"
            ],
            "cost": "Lowest - Hindi translation only"
        },
        
        "2. CELEBRITY_POPCULTURE": {
            "description": "Requires Western celebrity or pop culture knowledge",
            "subcategories": ["Celebrity_Dependent (16 categories)"],
            "total_memes": 16,
            "percentage": "8.1%", 
            "avg_dependency": "75.6%",
            "easy_adaptations": "0%",
            "adaptation_strategy": "Face swap + text translation",
            "examples": [
                "Movie stars (Leonardo DiCaprio, Ryan Gosling)",
                "Musicians (Harry Styles, Billie Eilish)",
                "TV characters, anime characters"
            ],
            "cost": "High - AI face swap + translation"
        },
        
        "3. SUBCULTURE_SLANG": {
            "description": "Internet slang, niche terminology, English wordplay",
            "subcategories": [
                "Subculture_Specific (15 categories)",
                "Language_Dependent (4 categories)"
            ],
            "total_memes": 19,
            "percentage": "9.6%",
            "avg_dependency": "72.1%", 
            "easy_adaptations": "10.5%",
            "adaptation_strategy": "Cultural linguistic mapping",
            "examples": [
                "Internet slang (weebs, yeet, stan)",
                "Medical acronyms (ED, BPD)",
                "English idioms and wordplay"
            ],
            "cost": "Medium - Cultural consultant + translation"
        },
        
        "4. BRAND_FORMAT": {
            "description": "Western brands, products, or specific meme formats",
            "subcategories": [
                "Brand_Product (3 categories)",
                "Format_Dependent (6 categories)"
            ],
            "total_memes": 9,
            "percentage": "4.5%",
            "avg_dependency": "50.0%",
            "easy_adaptations": "33.3%", 
            "adaptation_strategy": "Brand replacement or format adaptation",
            "examples": [
                "Food brands (Nesquik, energy drinks)",
                "UI screenshots, specific meme templates",
                "Western product references"
            ],
            "cost": "Medium - Visual editing + translation"
        },
        
        "5. COMPLEX_HYBRID": {
            "description": "Multiple barriers or unclear categorization",
            "subcategories": ["Miscellaneous (23 categories)"],
            "total_memes": 23,
            "percentage": "11.6%",
            "avg_dependency": "43.5%",
            "easy_adaptations": "30.4%",
            "adaptation_strategy": "Case-by-case analysis",
            "examples": [
                "Multiple cultural elements",
                "Unclear or unique barriers",
                "Requires human review"
            ],
            "cost": "Variable - Manual analysis required"
        }
    }
    
    return final_taxonomy

def print_taxonomy_summary():
    """Print the final taxonomy with key insights"""
    
    taxonomy = create_final_taxonomy()
    
    print("=== FINAL 5-CATEGORY TAXONOMY FOR MEME CULTURAL ADAPTATION ===\n")
    
    total_memes = 198
    
    for category, info in taxonomy.items():
        print(f"{category}")
        print(f"  📊 Coverage: {info['total_memes']} memes ({info['percentage']})")
        print(f"  🎯 Cultural Dependency: {info['avg_dependency']} average")
        print(f"  ⚡ Easy Adaptations: {info['easy_adaptations']}")
        print(f"  🔧 Strategy: {info['adaptation_strategy']}")
        print(f"  💰 Cost: {info['cost']}")
        print(f"  📝 Examples: {', '.join(info['examples'][:2])}")
        print()
    
    print("=== KEY RESEARCH INSIGHTS ===")
    print(f"✅ 66.2% of memes are UNIVERSAL (text translation only)")
    print(f"⚠️  Only 8.1% require celebrity face swaps (expensive)")
    print(f"📚 9.6% need cultural linguistic mapping")
    print(f"🔧 4.5% need brand/format changes")
    print(f"❓ 11.6% need case-by-case analysis")
    print()
    print("=== COST-EFFECTIVENESS RANKING ===")
    print("1. UNIVERSAL (66.2%) - Cheapest, highest ROI")
    print("2. SUBCULTURE_SLANG (9.6%) - Medium cost, cultural consultant")
    print("3. BRAND_FORMAT (4.5%) - Medium cost, visual editing")
    print("4. CELEBRITY_POPCULTURE (8.1%) - Expensive, AI face swap")
    print("5. COMPLEX_HYBRID (11.6%) - Variable cost, manual review")

if __name__ == "__main__":
    print_taxonomy_summary()