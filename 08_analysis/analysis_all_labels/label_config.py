"""Shared label metadata for all-labels deep analysis."""

LABELS7 = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problem",
    "Sleeping Disorder",
]

SHORT = {
    "Feeling Down": "FD",
    "Lack of Interest": "LOI",
    "Self-Harm": "SH",
    "Eating Disorder": "ED",
    "Low Self-Esteem": "LSE",
    "Concentration Problem": "CP",
    "Sleeping Disorder": "SD",
}

MODELS = ["gpt4o_mini", "mistral_large3", "mistral_small_2503", "llama4", "gemini"]
MODEL_LABELS = {
    "gpt4o_mini": "GPT-4o-mini",
    "mistral_large3": "Mistral-L3",
    "mistral_small_2503": "Mistral-S",
    "llama4": "Llama-4",
    "gemini": "Gemini",
}

EXPERIMENTS = [
    ("deva_image_only", "Deva Img-Only"),
    ("eng_image_only", "Eng Img-Only"),
    ("deva_expl_only", "Deva Expl-Only"),
    ("eng_expl_only", "Eng Expl-Only"),
    ("deva_img_deva_expl", "Deva Img+Deva Expl"),
    ("eng_img_eng_expl", "Eng Img+Eng Expl"),
    ("deva_img_eng_expl", "Deva Img+Eng Expl"),
]

# Explicit mention patterns (English / Devanagari-ish) per label
LABEL_PATTERNS = {
    "Feeling Down": {
        "eng": [
            r"feeling down", r"\bsadness\b", r"\bdepressed\b", r"\bsad\b",
            r"emotional distress", r"hopelessness", r"despair",
        ],
        "deva": [
            r"feeling down", r"फीलिंग डाउन", r"उदास", r"निराश", r"अवसाद", r"दुख",
        ],
    },
    "Lack of Interest": {
        "eng": [
            r"lack of interest", r"anhedonia", r"loss of pleasure", r"loss of interest",
            r"no motivation", r"nothing enjoyable",
        ],
        "deva": [
            r"lack of interest", r"लैक ऑफ", r"लेस ऑफ", r"रुचि", r"अनहेडोनिया", r"उत्साह",
        ],
    },
    "Self-Harm": {
        "eng": [
            r"self-harm", r"self harm", r"suicidal", r"suicide", r"self-injur",
        ],
        "deva": [
            r"self-harm", r"self harm", r"सेल्फ-हार्म", r"आत्महत्या", r"suicidal",
        ],
    },
    "Eating Disorder": {
        "eng": [
            r"eating disorder", r"binge", r"anorexia", r"bulimia", r"appetite",
            r"disordered eating", r"food restriction",
        ],
        "deva": [
            r"eating disorder", r"ईटिंग", r"भूख", r"खाने", r"appetite",
        ],
    },
    "Low Self-Esteem": {
        "eng": [
            r"low self-esteem", r"self-esteem", r"worthless", r"self-worth",
            r"self-criticism", r"inadequacy",
        ],
        "deva": [
            r"low self-esteem", r"self-esteem", r"आत्म-सम्मान", r"आत्मसम्मान", r"self esteem",
        ],
    },
    "Concentration Problem": {
        "eng": [
            r"concentration problem", r"concentration", r"focus", r"rumination",
            r"overthinking", r"over-analyzing", r"mental fixation",
        ],
        "deva": [
            r"concentration", r"एकाग्रता", r"focus", r"overthinking", r"विचार",
        ],
    },
    "Sleeping Disorder": {
        "eng": [
            r"sleeping disorder", r"sleep disorder", r"insomnia", r"\bsleep\b",
            r"sleepless", r"can't sleep", r"fatigue", r"exhaustion",
        ],
        "deva": [
            r"sleeping disorder", r"sleep", r"नींद", r"insomnia", r"थकान", r"स्लीप",
        ],
    },
}

# When gold is X, these co-occurring labels in explanations count as "substitute"
SUBSTITUTES = {
    "Feeling Down": ["Low Self-Esteem", "Lack of Interest", "Concentration Problem"],
    "Lack of Interest": ["Feeling Down", "Low Self-Esteem", "Sleeping Disorder"],
    "Self-Harm": ["Feeling Down", "Low Self-Esteem"],
    "Eating Disorder": ["Feeling Down", "Low Self-Esteem", "Self-Harm"],
    "Low Self-Esteem": ["Feeling Down", "Lack of Interest", "Concentration Problem"],
    "Concentration Problem": ["Feeling Down", "Sleeping Disorder", "Low Self-Esteem"],
    "Sleeping Disorder": ["Feeling Down", "Lack of Interest", "Concentration Problem"],
}

RECOMMENDATIONS = {
    "Feeling Down": [
        "Distinguish transient sadness from clinical low mood in explanation prompts; avoid collapsing all negative affect into Feeling Down.",
        "Report Feeling Down co-labeled memes separately; sadness cues often satisfy models before subtler labels are tagged.",
        "Use explanation text that names mood decline explicitly rather than generic distress vocabulary.",
    ],
    "Lack of Interest": [
        "Separate PHQ-9 item 1 (Feeling Down) from item 2 (LOI) with an explicit anhedonia check in explanation prompts.",
        "Report LOI-only subset separately; image-only baselines underperform when anhedonia is not visually explicit.",
        "Use Eng explanation + Deva image when Hindi OCR reads as generic sadness.",
    ],
    "Self-Harm": [
        "Add a dark-humor vs genuine self-harm disambiguation step; memes with death jokes often trigger false Self-Harm positives.",
        "Audit Hindi explanations for over-assignment of Self-Harm on existential or humorous content.",
        "Require explicit suicidal/self-injury language before assigning Self-Harm in explanation generation.",
    ],
    "Eating Disorder": [
        "Prompt explanations to look for appetite, restriction, binge, or body-image cues rather than generic stress eating.",
        "ED is rare in the test set; treat low support as a driver of unstable F1 across conditions.",
        "Cross-check image text for food/body references when explanation-only runs miss ED.",
    ],
    "Low Self-Esteem": [
        "Separate self-deprecation humor from clinical low self-worth in prompts; many memes use self-mockery without LSE gold.",
        "Co-label confusion with Feeling Down is common; require explicit worthlessness/inadequacy language.",
        "Explanation-only runs help when memes verbalize self-criticism rather than show it visually.",
    ],
    "Concentration Problem": [
        "Differentiate rumination/overthinking (CP) from sadness (Feeling Down) and fatigue (Sleeping Disorder) in prompts.",
        "Memes about procrastination or brain fog often map to CP only when text mentions focus or mental overload.",
        "Multimodal runs benefit when explanation names unproductive fixation explicitly.",
    ],
    "Sleeping Disorder": [
        "Distinguish tiredness/zoom-meme humor from clinical sleep disturbance in explanation prompts.",
        "Insomnia wording in Devanagari OCR is a strong signal; image-only runs miss when fatigue is implied visually only.",
        "Avoid mapping all low-energy memes to Sleeping Disorder when gold is LOI or Feeling Down.",
    ],
}

GLOBAL_SUMMARIES = {
    "Feeling Down": (
        "Feeling Down is the most frequent label and generally achieves the highest F1. "
        "Failures often involve co-label memes where models tag a secondary symptom but miss FD, "
        "or substitute LSE/CP when sadness is implied indirectly."
    ),
    "Lack of Interest": (
        "LOI scores lowest overall. Image-only inputs rarely surface anhedonia; explanation pipelines "
        "often substitute Feeling Down. Classifiers sometimes ignore explicit LOI mentions in explanation text."
    ),
    "Self-Harm": (
        "Self-Harm F1 is mid-range but volatile. Dark humor and existential memes cause false positives; "
        "subtle self-harm gold labels are missed when explanations emphasize sadness instead."
    ),
    "Eating Disorder": (
        "Eating Disorder has low support (~90 gold memes) and moderate F1. Models confuse ED with "
        "Feeling Down or LSE when food imagery is absent and explanations describe generic distress."
    ),
    "Low Self-Esteem": (
        "Low Self-Esteem sits in the mid F1 band. Self-deprecating meme formats frequently pull predictions "
        "toward LSE even when gold is another label, and vice versa when gold is LSE but text reads as sadness."
    ),
    "Concentration Problem": (
        "Concentration Problem F1 is moderate. Rumination and overthinking meme templates help when named "
        "in explanations; failures cluster on memes where worry reads as Feeling Down or sleep fatigue."
    ),
    "Sleeping Disorder": (
        "Sleeping Disorder achieves relatively strong F1 when insomnia or exhaustion is explicit in text. "
        "Failures occur on memes where tiredness is comedic or co-labeled with LOI/FD."
    ),
}
