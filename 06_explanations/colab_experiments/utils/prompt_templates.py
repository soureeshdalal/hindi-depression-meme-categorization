"""
Prompt templates for Colab multimodal experiments.
"""

SYMPTOM_NAMES = [
    "Feeling Down",
    "Lack of Interest",
    "Self-Harm",
    "Eating Disorder",
    "Low Self-Esteem",
    "Concentration Problems",
    "Sleeping Disorder",
]

INSTRUCTBLIP_PROMPT = """You are classifying a mental-health meme into 7 PHQ-9 symptom indicators.

Indicators in order:
1) Feeling Down
2) Lack of Interest
3) Self-Harm
4) Eating Disorder
5) Low Self-Esteem
6) Concentration Problems
7) Sleeping Disorder

Rules:
- Mark 1 if clearly present, else 0.
- Be conservative; avoid guessing.
- Return exactly this format and nothing else:
PREDICTION: [b1,b2,b3,b4,b5,b6,b7]
"""

EXPLANATION_AUGMENTED_PROMPT = """You are classifying a mental-health meme into 7 PHQ-9 symptom indicators.

Use both the image and this supporting explanation:
EXPLANATION:
{explanation}

Indicators in order:
1) Feeling Down
2) Lack of Interest
3) Self-Harm
4) Eating Disorder
5) Low Self-Esteem
6) Concentration Problems
7) Sleeping Disorder

Rules:
- Mark 1 if clearly present, else 0.
- If explanation conflicts with image, prioritize direct visual/text evidence from the meme.
- Return exactly this format and nothing else:
PREDICTION: [b1,b2,b3,b4,b5,b6,b7]
"""


def format_instructblip_prompt() -> str:
    return INSTRUCTBLIP_PROMPT


def format_explanation_prompt(explanation_text: str) -> str:
    explanation_text = (explanation_text or "").strip()
    if not explanation_text:
        explanation_text = "[NO_EXPLANATION_AVAILABLE]"
    return EXPLANATION_AUGMENTED_PROMPT.format(explanation=explanation_text)
