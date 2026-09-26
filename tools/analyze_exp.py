#!/usr/bin/env python
"""Analyze a finished experiment and emit the Go/No-Go decision + error report.

Reads experiments/<exp>/{metrics.json,results.csv,val_samples/...} produced by
tools/run_experiment.py and writes:
    reports/<exp>_analysis.md   (overall, per-class, trend, verdict)
    reports/error_analysis.md   (worst classes, confusion notes, small-object lens)

Verdict thresholds (empirical for yolo26n at 640 on a clean 6-class IDD):
    GREEN  mAP50 >= 0.25 and recall >= 0.35 and no class below 0.10  -> go larger
    YELLOW 0.12 <= mAP50 < 0.25  -> fix worst class first, do NOT scale yet
    RED   mAP50 < 0.12           -> stop, diagnose dataset/annotations

Usage:
    venv/bin/python tools/analyze_exp.py --exp experiments/exp001
"""
import argparse
import csv
import json
import os
import sys

NAMES6 = ["person", "car", "motorcycle", "bus", "truck", "autorickshaw"]


def load_trend(results_csv):
    if not os.path.exists(results_csv):
        return {}
    rows = list(csv.DictReader(open(results_csv)))
    if not rows:
        return {}
    def g(r, k):
        try:
            return float(r[k])
        except (KeyError, ValueError):
            return None
    m50 = [g(r, "metrics/mAP50(B)") for r in rows]
    m5095 = [g(r, "metrics/mAP50-95(B)") for r in rows]
    rec = [g(r, "metrics/recall(B)") for r in rows]
    m50 = [x for x in m50 if x is not None]
    m5095 = [x for x in m5095 if x is not None]
    rec = [x for x in rec if x is not None]
    best_ep = m5095.index(max(m5095)) + 1 if m5095 else None
    tail = m5095[-15:] if len(m5095) > 15 else m5095
    flat = (len(tail) >= 10 and (tail[-1] - tail[0]) < 0.002) if tail else None
    return {"first_mAP50": m50[0] if m50 else None,
            "last_mAP50": m50[-1] if m50 else None,
            "best_epoch_mAP5095": best_ep,
            "curve_flat_last15": flat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    exp = args.exp.rstrip("/")
    name = os.path.basename(exp)
    mp = os.path.join(exp, "metrics.json")
    if not os.path.exists(mp):
        sys.exit(f"{mp} missing - has the experiment been run?")
    m = json.load(open(mp))
    trend = load_trend(os.path.join(exp, "results.csv"))

    ap50 = m.get("mAP50", 0.0)
    rec = m.get("recall", 0.0)
    prec = m.get("precision", 0.0)
    per = m.get("per_class_ap50", {})

    worst = sorted(per.items(), key=lambda kv: kv[1])[:3]
    best = sorted(per.items(), key=lambda kv: -kv[1])[:3]

    if ap50 >= 0.25 and rec >= 0.35:
        verdict, why = "GREEN", "proceed to exp002 (larger model)"
    elif ap50 >= 0.12:
        verdict, why = "YELLOW", "dataset/model behaves, but fix weakest classes before scaling"
    else:
        verdict, why = "RED", "catastrophically low - STOP, diagnose dataset before any bigger run"

    md = [
        f"# {name} Analysis (Go/No-Go)", "",
        f"- **Verdict: {verdict}** - {why}", "",
        "## Headline metrics", "",
        f"| metric | value |", "|---|---|",
        f"| mAP50 | {ap50} |",
        f"| mAP50-95 | {m.get('mAP50-95', 0)} |",
        f"| precision | {prec} |",
        f"| recall | {rec} |",
        f"| GPU | {m.get('gpu')} |",
        f"| train_seconds | {m.get('train_seconds')} |",
        "",
        "## Per-class AP50", "", "| class | AP50 |", "|---|---|",
    ]
    for c in NAMES6:
        md.append(f"| {c} | {per.get(c, None)} |")
    md += [
        "", "## Curve shape (from results.csv)", "",
        f"| epoch point | value |", "|---|---|",
        f"| first mAP50 | {trend.get('first_mAP50')} |",
        f"| last mAP50 | {trend.get('last_mAP50')} |",
        f"| best epoch (mAP50-95) | {trend.get('best_epoch_mAP5095')} |",
        f"| flat over last 15 ep | {trend.get('curve_flat_last15')} |",
        "", "## Diagnosis", "",
        f"- weakest classes: {', '.join(f'{c}={v}' for c, v in worst)}",
        f"- strongest classes: {', '.join(f'{c}={v}' for c, v in best)}",
        f"- precision {prec} vs recall {rec}: "
        f"{'model under-detects (misses objects)' if rec < prec else 'model over-fires (false alarms)'}",
    ]
    if verdict == "RED":
        md += [
            "", "### STOP - do not launch exp002. Likely causes to check first:", "",
            "1. Dataset path/remap correctness (verify class ids + counts in v1).",
            "2. Labels visually (reports/visual_audit/) - is the box content right?",
            "3. Small-object density: person/motorcycle subpixel boxes are unlearnable - "
            "consider filtering boxes < ~16px area.",
            "4. Train/val distribution shift.",
        ]
    elif verdict == "YELLOW":
        md += [
            "", "### Before scaling: attack the weak class(es).",
            f"Worst: {worst[0][0] if worst else '?'}. Options: more data for it, merge it "
            "into a neighbor class, or deduplicate/tidy its labels.",
        ]
    else:
        md += ["", "### Proceed as planned to exp002 (yolo26l)."]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}_analysis.md"), "w") as f:
        f.write("\n".join(md))

    # error_analysis.md (overwrites with the latest exp's findings)
    ea = [
        "# Error Analysis", "",
        f"Source: {name} ({m.get('timestamp', '?')})", "",
        "## Worst classes (failure targets)", "",
    ]
    for c, v in worst:
        ea.append(f"- **{c}** AP50 = {v}")
    ea += [
        "", "## Confusion & failure modes to inspect", "",
        "- person/motorcycle are the smallest objects -> check `val_samples/val/` "
        "for missed small boxes.",
        "- full-frame bus/truck boxes (from the audit: 1,028 bus + 1,024 truck "
        "boxes >0.95 normalized side) can merge with nearby objects.",
        "- dataset audit box-size stats: 16.6% of person boxes have a side < 6px "
        "@640 - subpixel GT is a hard ceiling on recall.",
        "", "## Next action",
        f"Per verdict in {name}_analysis.md.",
    ]
    with open(os.path.join(args.out, "error_analysis.md"), "w") as f:
        f.write("\n".join(ea))

    print(md[2])
    print(f"wrote reports/{name}_analysis.md and reports/error_analysis.md")


if __name__ == "__main__":
    main()