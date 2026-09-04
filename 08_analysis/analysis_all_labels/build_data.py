#!/usr/bin/env python3
"""Build data.json for all-labels deep analysis."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_config import (
    EXPERIMENTS,
    GLOBAL_SUMMARIES,
    LABEL_PATTERNS,
    LABELS7,
    MODEL_LABELS,
    MODELS,
    RECOMMENDATIONS,
    SHORT,
    SUBSTITUTES,
)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
RESULTS_BASE = ROOT / "08_analysis" / "colab_test_results"
ENG_EXPL = RESULTS_BASE / "explanations" / "mistral_large3" / "eng" / "explanations.json"
DEVA_EXPL = RESULTS_BASE / "explanations" / "gpt4o_mini" / "deva" / "explanations.json"
TEST_JSON = ROOT / "test.json"
IMG_BASE = ROOT / "translated_categorized_memes"


def f1_binary(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    if tp + fp + fn == 0:
        return 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)


def load_results(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else list(data.values())


def load_explanations(path):
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["sample_id"]: r.get("explanation", "") for r in rows}


def build_category_lookup():
    lookup = {}
    if not IMG_BASE.exists():
        return lookup
    for cat_dir in IMG_BASE.iterdir():
        if not cat_dir.is_dir():
            continue
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for img in test_dir.glob("*.jpg"):
                lookup[img.stem] = cat_dir.name
    return lookup


def image_rel(sample_id):
    for cat_dir in IMG_BASE.iterdir():
        if not cat_dir.is_dir():
            continue
        p = cat_dir / "test" / f"{sample_id}.jpg"
        if p.exists():
            return str(p.relative_to(ROOT))
    return None


def matches_label(text, label, lang):
    if not text:
        return False
    pats = LABEL_PATTERNS.get(label, {}).get(lang, [])
    return any(re.search(p, text, flags=re.I) for p in pats)


def classify_explanation(text, target_label, lang):
    if not text or len(str(text).strip()) < 15:
        return "missing"
    if matches_label(text, target_label, lang):
        return "explicit"
    for sub in SUBSTITUTES.get(target_label, []):
        if matches_label(text, sub, lang):
            return "substitute"
    return "no_signal"


def load_all_predictions():
    """run_key -> sample_id -> {pred, gt}"""
    all_preds = {}
    sample_ids = None
    for model in MODELS:
        for exp_key, _ in EXPERIMENTS:
            path = RESULTS_BASE / model / exp_key / "results.json"
            records = load_results(path)
            if not records:
                continue
            run_key = f"{model}/{exp_key}"
            run_map = {}
            for r in records:
                sid = r["sample_id"]
                run_map[sid] = {
                    "pred": r["predictions"],
                    "gt": r.get("ground_truth", r.get("meme_depressive_categories", [])),
                }
            all_preds[run_key] = run_map
            ids = set(run_map)
            sample_ids = ids if sample_ids is None else sample_ids & ids
    return all_preds, sorted(sample_ids or [])


def compute_f1_table(all_preds, label):
    rows = []
    label_idx = LABELS7.index(label)
    for exp_key, exp_label in EXPERIMENTS:
        scores = {}
        for model in MODELS:
            run_key = f"{model}/{exp_key}"
            if run_key not in all_preds:
                continue
            y_true, y_pred = [], []
            for sid, rec in all_preds[run_key].items():
                gt = rec["gt"]
                pred = rec["pred"]
                y_true.append(1 if label in gt else 0)
                y_pred.append(pred.get(label, 0))
            if y_true:
                scores[MODEL_LABELS[model]] = f1_binary(y_true, y_pred)
        if scores:
            rows.append({"condition": exp_label, "scores": scores})
    return rows


def explanation_stats(gold_ids, eng_map, deva_map, label):
    eng_counts = Counter()
    deva_counts = Counter()
    both_miss = 0
    for sid in gold_ids:
        ec = classify_explanation(eng_map.get(sid, ""), label, "eng")
        dc = classify_explanation(deva_map.get(sid, ""), label, "deva")
        eng_counts[ec] += 1
        deva_counts[dc] += 1
        if ec != "explicit" and dc != "explicit":
            both_miss += 1
    return {
        "eng": dict(eng_counts),
        "deva": dict(deva_counts),
        "both_miss": both_miss,
        "gold_count": len(gold_ids),
    }


def score_candidate(sid, label, gold, eng_class, deva_class, label_preds, fail_rate):
    """Higher score = better case study candidate for a given slot type."""
    pure = gold == [label]
    co = len(gold) > 1 and label in gold
    both_miss = eng_class != "explicit" and deva_class != "explicit"
    eng_hit = eng_class == "explicit"
    deva_hit = deva_class == "explicit"
    hits = sum(1 for v in label_preds.values() if v == 1)
    total = len(label_preds) or 1

    return {
        "sid": sid,
        "gold": gold,
        "eng_class": eng_class,
        "deva_class": deva_class,
        "label_preds": label_preds,
        "fail_rate": fail_rate,
        "hits": hits,
        "total": total,
        "pure": pure,
        "co": co,
        "both_miss": both_miss,
        "eng_hit": eng_hit,
        "deva_hit": deva_hit,
    }


def pick_cases(candidates, used_ids):
    """Pick 3 diverse cases per label."""
    chosen = []
    used = set(used_ids)

    def pick(pool, key_fn, reverse=True):
        pool = [c for c in pool if c["sid"] not in used]
        if not pool:
            return None
        best = sorted(pool, key=key_fn, reverse=reverse)[0]
        used.add(best["sid"])
        chosen.append(best)
        return best

    # Case 1: persistent miss, explanation miss preferred
    pick(
        [c for c in candidates if c["fail_rate"] >= 0.55],
        lambda c: (c["both_miss"], c["fail_rate"], c["pure"]),
    ) or pick(candidates, lambda c: c["fail_rate"])

    # Case 2: co-label or explanation-classifier gap
    pick(
        [c for c in candidates if c["co"] and c["fail_rate"] >= 0.35],
        lambda c: (c["eng_hit"] and c["fail_rate"], c["fail_rate"]),
    ) or pick(
        [c for c in candidates if c["eng_hit"] and c["fail_rate"] >= 0.4],
        lambda c: c["fail_rate"],
    ) or pick(
        [c for c in candidates if c["co"]],
        lambda c: c["fail_rate"],
    )

    # Case 3: partial success or clear success when explanations hit
    pick(
        [c for c in candidates if c["eng_hit"] and c["deva_hit"] and 0.2 <= c["fail_rate"] <= 0.6],
        lambda c: (-c["fail_rate"], c["hits"]),
    ) or pick(
        [c for c in candidates if c["fail_rate"] <= 0.35],
        lambda c: (-c["fail_rate"], c["hits"]),
        reverse=False,
    ) or pick(candidates, lambda c: -c["fail_rate"], reverse=False)

    # Fill if needed
    while len(chosen) < 3:
        rest = [c for c in candidates if c["sid"] not in used]
        if not rest:
            break
        pick(rest, lambda c: c["fail_rate"])

    return chosen[:3], used


def case_title(label, case_num, c):
    short = SHORT[label]
    sid = c["sid"]
    if c["both_miss"] and c["fail_rate"] >= 0.6:
        return f"Case {case_num} - Persistent miss, explanations miss ({short}, {sid})"
    if c["co"] and c["fail_rate"] >= 0.4:
        others = [l for l in c["gold"] if l != label]
        other = others[0] if others else "co-label"
        return f"Case {case_num} - Co-label with {other}, partial confusion ({short}, {sid})"
    if c["eng_hit"] and c["fail_rate"] >= 0.35:
        return f"Case {case_num} - English names {short}; models still split ({sid})"
    if c["fail_rate"] <= 0.35:
        return f"Case {case_num} - Models largely succeed ({short}, {sid})"
    return f"Case {case_num} - Mixed failure pattern ({short}, {sid})"


def build_example(c, label, all_preds, test_lookup, eng_map, deva_map, cat_lookup):
    sid = c["sid"]
    predictions = {}
    for run_key, run_map in all_preds.items():
        if sid in run_map:
            predictions[run_key] = run_map[sid]

    return {
        "sample_id": sid,
        "gold_labels": c["gold"],
        "category": cat_lookup.get(sid, "UNKNOWN"),
        "image_rel": image_rel(sid),
        "ocr_text": test_lookup.get(sid, {}).get("ocr_text", ""),
        "eng_explanation": eng_map.get(sid, ""),
        "deva_explanation": deva_map.get(sid, ""),
        "eng_class": c["eng_class"],
        "deva_class": c["deva_class"],
        "label_predictions": c["label_preds"],
        "predictions": predictions,
        "fail_rate": round(c["fail_rate"], 3),
        "hits": c["hits"],
        "total_runs": c["total"],
    }


def main():
    all_preds, sample_ids = load_all_predictions()
    test_rows = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    test_lookup = {r["sample_id"]: r for r in test_rows}
    eng_map = load_explanations(ENG_EXPL)
    deva_map = load_explanations(DEVA_EXPL)
    cat_lookup = build_category_lookup()

    print(f"Loaded {len(all_preds)} model runs, {len(sample_ids)} shared samples")

    used_ids = set()
    labels_out = []

    for label in LABELS7:
        gold_ids = [
            sid for sid in sample_ids
            if label in all_preds[next(iter(all_preds))][sid]["gt"]
        ]
        stats = explanation_stats(gold_ids, eng_map, deva_map, label)
        f1_table = compute_f1_table(all_preds, label)

        candidates = []
        for sid in gold_ids:
            gold = all_preds[next(iter(all_preds))][sid]["gt"]
            eng_class = classify_explanation(eng_map.get(sid, ""), label, "eng")
            deva_class = classify_explanation(deva_map.get(sid, ""), label, "deva")
            label_preds = {}
            for run_key, run_map in all_preds.items():
                if sid in run_map:
                    label_preds[run_key] = run_map[sid]["pred"].get(label, 0)
            misses = sum(1 for v in label_preds.values() if v == 0)
            total = len(label_preds) or 1
            fail_rate = misses / total
            candidates.append(
                score_candidate(sid, label, gold, eng_class, deva_class, label_preds, fail_rate)
            )

        picked, used_ids = pick_cases(candidates, used_ids)
        examples = []
        for i, c in enumerate(picked, 1):
            ex = build_example(c, label, all_preds, test_lookup, eng_map, deva_map, cat_lookup)
            ex["title"] = case_title(label, chr(64 + i), c)
            examples.append(ex)

        labels_out.append({
            "name": label,
            "short": SHORT[label],
            "gold_count": stats["gold_count"],
            "f1_table": f1_table,
            "stats": stats,
            "global_summary": GLOBAL_SUMMARIES.get(label, ""),
            "recommendations": RECOMMENDATIONS.get(label, []),
            "examples": examples,
        })
        print(f"{label}: {stats['gold_count']} gold, {len(examples)} cases")

    data = {
        "run_count": len(all_preds),
        "sample_count": len(sample_ids),
        "labels": labels_out,
    }
    out_path = OUT_DIR / "data.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
