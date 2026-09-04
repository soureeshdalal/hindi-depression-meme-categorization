2. Overall Performance

Best Macro-F1 is about 0.32

Average Macro-F1 is about 0.19

This is low, but expected for zero-shot clinical multi-label classification

Top performers:

MentalBERT: 0.321

MentalBART: 0.321

BLIP-2: 0.293

3. What the Models Actually Did
3.1 Most models predicted everything as positive

Many models predicted at least one symptom for almost every meme:

MentalBART: 99.9 percent of samples

MentalBERT: 95.5 percent

Effect

Recall becomes very high

Precision drops to the base rate of each symptom

F1 looks decent but is misleading

Example:
If a symptom appears in 15 percent of the data, predicting it for all samples gives:

Precision 15 percent

Recall 100 percent

F1 about 0.26

This explains many of the scores.

3.2 Some models used fixed symptom patterns

Several models always predicted the same symptoms and ignored the input.

Examples:

BART-Base predicted four specific symptoms and never the others

BART-Large predicted only Feeling Down and Sleeping Disorder

MuRIL predicted three specific symptoms only

LLaVA models mostly predicted Feeling Down and Low Self-Esteem

These models are not reading the meme content. They are using a fixed output pattern.

3.3 MentalBERT and MentalBART produced identical outputs

Same Macro-F1 score

Same per symptom results

Same behavior of predicting almost everything

This shows the scores are driven by dataset base rates, not by learning.

4. Best Model per Symptom

Feeling Down: LLaVA-1.5 at 0.617
This symptom is common, so high recall helps

Lack of Interest: MentalBERT at 0.258
This comes from predicting everything

Self-Harm: BLIP-2 at 0.338
Only model with a real balance of precision and recall

Eating Disorder: IndicBERTv2-Sam at 0.270
Only text model showing real signal

Low Self-Esteem: MentalBERT at 0.331
Again due to predicting everything

Concentration Problem: BLIP-2 at 0.317
Shows balanced detection

Sleeping Disorder: LLaVA-1.5 at 0.295
Some real detection ability

5. Did Any Model Actually Learn the Task

Only two models show real signal:

BLIP-2

Balanced precision and recall

Detects Self-Harm and Concentration Problems with some accuracy

Uses image and text together effectively

IndicBERTv2-Sam

Shows some signal for Eating Disorder

Likely picking up food or body related language from OCR text

All other models either predict everything, predict nothing, or use fixed patterns. Their scores do not reflect real understanding.

6. Why Multimodal Models Did Not Outperform Text Models

Even though multimodal models see the image, they still do not win.

Reason 1: Language mismatch

Memes are in Hindi using Devanagari script

Most models were trained mainly on English

Reading Hindi text in images is harder for them

Reason 2: Memes require cultural context

Memes rely on humor, references, and implied meaning

The depressive signal is often indirect

Zero-shot models cannot infer this well

Reason 3: Clinical categories are subtle

PHQ-9 categories are clinically defined

Mapping meme content to these categories requires domain knowledge

Zero-shot models lack this calibration

Result: the image input often adds noise instead of useful signal.

BLIP-2 is the only model that partially overcomes this.

7. Summary of Model Behavior

Always predicts almost everything:

MentalBART, MentalBERT, MentalRoBERTa, BART-Base
These rely on base rates

Fixed symptom pattern:

BART-Large, MuRIL, LLaVA-1.5, LLaVA-NeXT, PALO
These ignore the input

Almost never predicts:

BLIP
Misses most cases

Some real signal:

BLIP-2, IndicBERTv2-Sam

Partial signal on dominant symptom:

IDEFICS mainly on Feeling Down