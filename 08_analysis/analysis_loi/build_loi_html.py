#!/usr/bin/env python3
"""Regenerate LOI analysis HTML with model failure diagnostics."""

import html
import json
import re
import base64
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = Path(__file__).resolve().parent / "data.json"
LOI = "Lack of Interest"
LABELS7 = [
    "Feeling Down", "Lack of Interest", "Self-Harm", "Eating Disorder",
    "Low Self-Esteem", "Concentration Problem", "Sleeping Disorder",
]

LOI_F1 = [
    ("Deva Img-Only", {"GPT-4o-mini": 0.2591, "Mistral-L3": 0.3117, "Mistral-S": 0.3452, "Llama-4": 0.3750}),
    ("Eng Img-Only", {"GPT-4o-mini": 0.3351, "Mistral-L3": 0.4236, "Mistral-S": 0.4574, "Llama-4": 0.4234}),
    ("Deva Expl-Only", {"GPT-4o-mini": 0.4348, "Mistral-L3": 0.4053, "Mistral-S": 0.3636, "Llama-4": 0.3549, "Gemini": 0.4432}),
    ("Eng Expl-Only", {"GPT-4o-mini": 0.4238, "Mistral-L3": 0.4206, "Mistral-S": 0.3646, "Llama-4": 0.4156}),
    ("Deva Img+Deva Expl", {"GPT-4o-mini": 0.4249, "Mistral-L3": 0.4066, "Mistral-S": 0.3636, "Llama-4": 0.3655, "Gemini": 0.4565}),
    ("Eng Img+Eng Expl", {"GPT-4o-mini": 0.4249, "Mistral-L3": 0.4223, "Mistral-S": 0.3696, "Llama-4": 0.4221}),
    ("Deva Img+Eng Expl", {"GPT-4o-mini": 0.4317, "Mistral-L3": 0.4219, "Mistral-S": 0.3686, "Llama-4": 0.4235}),
]

CASE_TITLES = {
    "pure_loi_miss": "Case A - Pure LOI, both explanations miss",
    "loi_fd_miss": "Case B - LOI + Feeling Down, sadness substitute",
    "loi_captured_eng": "Case C - English captures LOI; Hindi still vague",
    "loi_captured_deva": "Case D - Both languages surface LOI; models succeed",
}


def replace_em_dash(text):
    """Replace em dashes with comma-based alternatives for display."""
    if not text:
        return text
    text = re.sub(r"\s—\s", ", ", str(text))
    return text.replace("—", ", ")


def esc(s):
    return html.escape(replace_em_dash(str(s or "")))


def highlight_loi(text):
    if not text:
        return '<em class="muted">(missing)</em>'
    text = replace_em_dash(text)
    patterns = [
        r"lack of interest", r"anhedonia", r"loss of pleasure", r"loss of interest",
        r"Lack of Interest", r"लैक ऑफ", r"लेस ऑफ", r"रुचि", r"अनहेडोनिया", r"उत्साह",
    ]
    out = esc(text)
    for p in patterns:
        out = re.sub(f"({p})", r"<mark>\1</mark>", out, flags=re.I)
    return out.replace("\n", "<br>")


def img_data_uri(rel_path):
    p = ROOT / rel_path
    if not p.exists():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def english_image_path(sample_id):
    for rel in (
        f"dataset/test/{sample_id}.jpg",
        f"colab_experiments/data/test_images/{sample_id}.jpg",
    ):
        if (ROOT / rel).exists():
            return rel
    return None


def image_pair_html(ex):
    eng_rel = english_image_path(ex["sample_id"])
    deva_rel = ex["image_rel"]
    eng_uri = img_data_uri(eng_rel) if eng_rel else None
    deva_uri = img_data_uri(deva_rel)

    eng_img = (
        f"<img src='{eng_uri}' alt='{esc(ex['sample_id'])} English'>"
        if eng_uri
        else "<p class='muted'>English image not found</p>"
    )
    deva_img = (
        f"<img src='{deva_uri}' alt='{esc(ex['sample_id'])} Devanagari'>"
        if deva_uri
        else "<p class='muted'>Devanagari image not found</p>"
    )
    return (
        "<div class='img-pair'>"
        f"<figure><figcaption class='eng'>Original (English)</figcaption>{eng_img}</figure>"
        f"<figure><figcaption class='deva'>Translated (Devanagari)</figcaption>{deva_img}</figure>"
        "</div>"
    )


def cond_group(key):
    c = key.split("/", 1)[1]
    if "image_only" in c:
        return "image_only"
    if "expl_only" in c:
        return "expl_only"
    return "multimodal"


def fmt_run(key):
    model, cond = key.split("/", 1)
    names = {
        "gpt4o_mini": "GPT-4o-mini",
        "mistral_large3": "Mistral-L3",
        "mistral_small_2503": "Mistral-S",
        "llama4": "Llama-4",
        "gemini": "Gemini",
    }
    cond_n = cond.replace("_", " ").replace("deva", "Deva").replace("eng", "Eng").title()
    return f"{names.get(model, model)} ({cond_n})"


def wrong_labels(pred):
    return [l for l in LABELS7 if l != LOI and pred.get(l, 0) == 1]


def build_analysis(ex):
    preds = ex.get("predictions", {})
    loi_map = ex.get("loi_predictions", {})
    n = len(loi_map) or 1
    hits = [k for k, v in loi_map.items() if v == 1]
    misses = [k for k, v in loi_map.items() if v == 0]

    by_group = {"image_only": [], "expl_only": [], "multimodal": []}
    for k, v in loi_map.items():
        by_group[cond_group(k)].append((k, v))

    wrong_counter = Counter()
    for k in misses:
        if k in preds:
            for l in wrong_labels(preds[k]["pred"]):
                wrong_counter[l] += 1

    eng, deva = ex["eng_class"], ex["deva_class"]
    gold = ex["gold_labels"]
    loi_only = gold == [LOI]

    sections = []

    # --- Explanation ---
    expl_points = []
    if eng == "explicit_loi":
        expl_points.append("English explanation names Lack of Interest / anhedonia.")
    elif eng == "substitutes_fd":
        expl_points.append("English explanation substitutes LOI with Feeling Down, Low Self-Esteem, or withdrawal.")
    else:
        expl_points.append("English explanation never signals anhedonia; routes to sleep, concentration, or fatigue.")

    if deva in ("explicit_loi", "explicit_loi_label"):
        expl_points.append("Hindi explanation names Lack of Interest.")
    elif deva == "substitutes_fd":
        expl_points.append("Hindi explanation emphasizes sadness/isolation, not loss of pleasure.")
    else:
        expl_points.append("Hindi explanation omits LOI cues entirely.")

    sections.append(
        "<h4>Explanation coverage</h4><ul>"
        + "".join(f"<li>{p}</li>" for p in expl_points)
        + "</ul>"
    )

    # --- Model failure summary ---
    fail_pct = 100 * len(misses) / n
    hit_runs = ", ".join(fmt_run(k) for k in hits) if hits else "none"
    sections.append(
        f"<h4>Where models fail on LOI</h4>"
        f"<p><strong>{len(misses)}/{n} runs predict LOI=0</strong> "
        f"({fail_pct:.0f}% failure on this meme). "
        f"Runs that <span class='ok-text'>correctly predict LOI=1</span>: {esc(hit_runs)}.</p>"
    )

    group_lines = []
    for g, label in [
        ("image_only", "Image-only"),
        ("expl_only", "Explanation-only"),
        ("multimodal", "Image + explanation"),
    ]:
        items = by_group[g]
        if not items:
            continue
        ok = sum(1 for _, v in items if v == 1)
        tot = len(items)
        group_lines.append(f"<li><strong>{label}:</strong> {ok}/{tot} correct</li>")
    sections.append("<ul class='compact'>" + "".join(group_lines) + "</ul>")

    if misses and wrong_counter:
        wrong_str = ", ".join(f"{l} ({c}/{len(misses)} misses)" for l, c in wrong_counter.most_common(3))
        sections.append(
            f"<p>On failed runs, models most often fire <strong>wrong labels</strong> instead: {esc(wrong_str)}.</p>"
        )

    # --- Per-run detail for misses ---
    if misses:
        detail = []
        for k in misses[:6]:
            wrong = wrong_labels(preds[k]["pred"]) if k in preds else []
            wrong_s = ", ".join(wrong) if wrong else "no positive label"
            detail.append(f"<li>{esc(fmt_run(k))}: predicted <span class='bad-text'>{esc(wrong_s)}</span></li>")
        if len(misses) > 6:
            detail.append(f"<li class='muted'>…and {len(misses)-6} more failed runs</li>")
        sections.append("<p><strong>Failed run details:</strong></p><ul class='compact'>" + "".join(detail) + "</ul>")

    # --- Possible reasons ---
    reasons = []

    if loi_only:
        reasons.append(
            "Gold is LOI-only with no Feeling Down co-label, so models cannot use sadness as a proxy; "
            "this is the hardest subset for LOI."
        )

    if eng != "explicit_loi" and deva not in ("explicit_loi", "explicit_loi_label"):
        reasons.append(
            "Explanation pipeline failure: neither language surfaces anhedonia, so explanation-only "
            "and multimodal runs inherit the wrong clinical framing."
        )
    elif eng == "explicit_loi" and sum(1 for k in misses if "eng" in k) > sum(1 for k in hits if "eng" in k):
        reasons.append(
            "Explanation-classifier gap: English text mentions LOI but the classifier still outputs LOI=0; "
            "the symptom is stated in prose but not transferred to the binary head."
        )
    elif deva in ("explicit_loi", "explicit_loi_label") and sum(1 for k in misses if "deva_expl" in k) > 0:
        reasons.append(
            "Hindi explanation names LOI but some Devanagari explanation-only runs still miss, "
            "possibly due to transliterated-label noise or co-occurring symptom confusion."
        )

    if by_group["image_only"] and all(v == 0 for _, v in by_group["image_only"]):
        reasons.append(
            "Image-only failure: anhedonia is not visually explicit (meme reads as sad, tired, or humorous); "
            "Devanagari OCR does not contain classic “no motivation / nothing enjoyable” wording."
        )

    if "Feeling Down" in gold and LOI in gold:
        reasons.append(
            "Co-labeling with Feeling Down: salient sadness cues in image/text dominate; "
            "models satisfy FD and stop without tagging the subtler anhedonia dimension."
        )

    if wrong_counter.get("Concentration Problem", 0) >= 2:
        reasons.append(
            "Rumination/overthinking framing in meme text pulls models toward Concentration Problem "
            "instead of LOI (PHQ-9 item 6 vs item 2 conflation)."
        )
    if wrong_counter.get("Sleeping Disorder", 0) >= 2:
        reasons.append(
            "Fatigue / waking / Zoom tropes trigger Sleeping Disorder predictions rather than "
            "loss of interest in activities."
        )
    if wrong_counter.get("Self-Harm", 0) >= 2:
        reasons.append(
            "Dark humor or death references mislead models toward Self-Harm despite gold LOI-only label."
        )
    if wrong_counter.get("Low Self-Esteem", 0) >= len(misses) // 2:
        reasons.append(
            "Self-deprecation and “life choice” framing maps to Low Self-Esteem in model outputs, "
            "masking the annotator’s anhedonia reading."
        )

    if ex["sample_id"] == "TE-516":
        reasons.append(
            "Hindi explanation incorrectly assigns Self-Harm, adding label noise that steers classifiers "
            "away from LOI entirely."
        )
    if ex["sample_id"] == "TE-114" and "gemini/deva_img_deva_expl" in misses:
        reasons.append(
            "Gemini multimodal miss despite strong explanations elsewhere, possibly due to image+text fusion "
            "override when multiple symptoms co-occur."
        )
    if ex["sample_id"] == "TE-15" and hits and misses:
        reasons.append(
            "Partial success: English explanation with explicit numbness/anhedonia helps Eng-Expl runs; "
            "Mistral Deva-Expl-Only still maps Hindi text to Feeling Down + Concentration Problem."
        )

    if not reasons:
        reasons.append(
            "Models largely succeed when explanations explicitly name social withdrawal or loss of interest; "
            "failures cluster on image-only and substitute-symptom explanations."
        )

    sections.append(
        "<h4>Possible reasons for failure</h4><ul>"
        + "".join(f"<li>{r}</li>" for r in reasons)
        + "</ul>"
    )

    return "\n".join(sections)


def main():
    data = json.load(open(DATA))
    st = data["stats"]
    letters = "EFGH"

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Lack of Interest Analysis</title><style>",
        "body{font-family:Segoe UI,system-ui,sans-serif;background:#f8f9fb;color:#1e293b;line-height:1.55;margin:0;padding:24px;max-width:1100px;margin-inline:auto}",
        "h1{color:#5b21b6} h2{border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-top:36px}",
        "h3{margin-top:0} h4{margin:14px 0 6px;font-size:14px;color:#334155}",
        ".card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px;margin:14px 0}",
        "table{width:100%;border-collapse:collapse;font-size:14px} th,td{border:1px solid #e2e8f0;padding:6px 8px;text-align:center}",
        "th{background:#f1f5f9} td:first-child,th:first-child{text-align:left}",
        ".low{background:#fef2f2;color:#b91c1c;font-weight:600}.mid{background:#fffbeb;color:#b45309}",
        "mark{background:#fef08a}.muted{color:#94a3b8;font-size:13px}",
        ".badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;margin:2px}",
        ".badge-gold{background:#ede9fe;color:#5b21b6}.badge-bad{background:#fee2e2;color:#b91c1c}.badge-ok{background:#dcfce7;color:#15803d}",
        ".bad-text{color:#b91c1c;font-weight:600}.ok-text{color:#15803d;font-weight:600}",
        ".case-grid{display:grid;grid-template-columns:420px 1fr;gap:16px;align-items:start}",
        ".img-pair{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}",
        ".img-pair figure{margin:0}",
        ".img-pair figcaption{font-size:11px;font-weight:600;color:#475569;text-align:center;margin-bottom:6px}",
        ".img-pair img{width:100%;border-radius:8px;border:1px solid #e2e8f0;display:block}",
        ".img-pair figcaption.deva{color:#c2410c}",
        ".img-pair figcaption.eng{color:#1d4ed8}",
        ".expl-box{background:#f8fafc;border-left:4px solid #94a3b8;padding:10px 12px;margin:10px 0;font-size:14px;max-height:240px;overflow:auto}",
        ".expl-box.eng{border-left-color:#2563eb}.expl-box.deva{border-left-color:#ea580c}",
        ".analysis-box{background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;padding:12px 14px;margin-top:10px;font-size:14px}",
        ".analysis-box ul{margin:6px 0;padding-left:18px}.analysis-box li{margin-bottom:5px}",
        "ul.compact li{margin-bottom:3px}",
        ".stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}",
        ".stat{text-align:center;padding:14px;background:#f1f5f9;border-radius:8px}.stat .num{font-size:28px;font-weight:700;color:#5b21b6}",
        ".finding li{margin-bottom:6px}",
        "@media(max-width:800px){.case-grid{grid-template-columns:1fr}}",
        "</style></head><body>",
        "<h1>Lack of Interest (Symptom #2) - Deep Analysis</h1>",
        "<p style='color:#64748b'>Self-contained report with embedded images. Each case shows the "
        "<strong>original English</strong> and <strong>translated Devanagari</strong> meme side by side, plus "
        "explanation coverage, model failure breakdown, and hypothesized failure reasons.</p>",
    ]

    parts.append("<h2>1. Why LOI scores lowest</h2><div class='card'><table>")
    parts.append("<tr><th>Condition</th><th>GPT-4o-mini</th><th>Mistral-L3</th><th>Mistral-S</th><th>Llama-4</th><th>Gemini</th></tr>")
    for cond, scores in LOI_F1:
        row = f"<tr><td>{esc(cond)}</td>"
        for m in ["GPT-4o-mini", "Mistral-L3", "Mistral-S", "Llama-4"]:
            v = scores[m]
            cls = "low" if v < 0.4 else ("mid" if v < 0.5 else "")
            row += f"<td class='{cls}'>{v:.4f}</td>"
        row += f"<td>{scores['Gemini']:.4f}</td>" if "Gemini" in scores else "<td>-</td>"
        row += "</tr>"
        parts.append(row)
    parts.append("</table>")
    parts.append(
        "<p><strong>Global failure pattern:</strong> Deva Img-Only F1 &lt; 0.38 for all models; image-only is the "
        "weakest input. Explanation-only lifts F1 modestly (~0.36-0.44) but still far below Feeling Down or "
        "Sleeping Disorder. Failures are driven by (1) explanation symptom substitution, (2) image-only inability "
        "to read anhedonia, and (3) classifiers ignoring explicit LOI mentions in explanation text.</p>"
    )
    parts.append("</div>")

    parts.append("<h2>2. Explanation coverage (97 gold LOI memes)</h2><div class='card'><div class='stat-grid'>")
    parts.append(f"<div class='stat'><div class='num'>{st['eng']['explicit_loi']}/97</div>English mention LOI</div>")
    parts.append(f"<div class='stat'><div class='num'>{st['eng']['substitutes_fd']}/97</div>English substitute sadness</div>")
    parts.append(f"<div class='stat'><div class='num'>{st['deva']['explicit_loi']+st['deva']['explicit_loi_label']}/97</div>Hindi name LOI</div>")
    parts.append(f"<div class='stat'><div class='num'>{st['both_miss']}/97</div>Both miss/substitute</div>")
    parts.append("</div></div>")

    parts.append("<h2>3. Side-by-side case studies</h2>")
    for i, ex in enumerate(data["examples"]):
        title = CASE_TITLES.get(ex["tag"], f"Case {letters[i - len(CASE_TITLES)]} - Persistent miss")
        title += f" ({ex['sample_id']})"
        gold_badges = "".join(f"<span class='badge badge-gold'>{esc(l)}</span>" for l in ex["gold_labels"])
        pred_bits = []
        for key, v in sorted(ex.get("loi_predictions", {}).items()):
            cls = "badge-ok" if v == 1 else "badge-bad"
            pred_bits.append(f"<span class='badge {cls}'>{key.replace('/', ' ')}: LOI={v}</span>")

        parts.append(f"<div class='card'><h3>{esc(title)}</h3>")
        parts.append(f"<p>{esc(ex['category'])} | Gold: {gold_badges}</p>")
        parts.append("<div class='case-grid'>")
        parts.append(f"<div>{image_pair_html(ex)}<p class='muted'><b>Original OCR:</b> {esc(ex['ocr_text'][:180])}...</p></div>")
        parts.append("<div>")
        parts.append("<p><b>English explanation</b> <span class='muted'>(mistral_large3/eng)</span></p>")
        parts.append(f"<div class='expl-box eng'>{highlight_loi(ex['eng_explanation'][:1000])}</div>")
        parts.append("<p><b>Hindi explanation</b> <span class='muted'>(gpt4o_mini/deva)</span></p>")
        parts.append(f"<div class='expl-box deva'>{highlight_loi(ex['deva_explanation'][:1000])}</div>")
        parts.append(f"<p><b>LOI predictions:</b> {''.join(pred_bits)}</p>")
        parts.append(f"<div class='analysis-box'>{build_analysis(ex)}</div>")
        parts.append("</div></div></div>")

    parts.append("<h2>4. Recommendations</h2><div class='card'><ul class='finding'>")
    parts.append("<li>Separate PHQ-9 item 1 (Feeling Down) from item 2 (LOI) in explanation prompts with an explicit anhedonia check.</li>")
    parts.append("<li>Report LOI-only subset separately (~20 memes); image-only baselines will underperform by design.</li>")
    parts.append("<li>Audit Hindi explanations for wrong symptom injection (e.g. Self-Harm on existential humor).</li>")
    parts.append("<li>Use Eng explanation + Deva image for LOI: English prose disambiguates when Hindi image reads as generic sadness.</li>")
    parts.append("</ul></div></body></html>")

    content = "".join(parts)
    out_dir = Path(__file__).resolve().parent
    for name in ("lack_of_interest_analysis.html", "lack_of_interest_analysis_standalone.html"):
        (out_dir / name).write_text(content, encoding="utf-8")
        print(f"Wrote {out_dir / name} ({(out_dir / name).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
