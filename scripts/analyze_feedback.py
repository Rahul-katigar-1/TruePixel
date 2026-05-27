"""
scripts/analyze_feedback.py

Offline analysis of the dev-feedback CSV produced by the dashboard's
Real/Fake buttons. Reads logs/dev_feedback.csv and reports:

  1. Confusion matrix (system verdict vs developer's ground-truth label)
  2. Per-signal separation analysis — for each numeric signal (composite,
     face_confidence, lbp_variance, liveness_score, signal_quality, etc.)
     it computes the range of values seen for "real" vs "fake" labels
     and suggests the threshold that best separates them.
  3. Misclassifications worth investigating — checks where the system
     disagreed with the developer's label.

Run from project root:
    python scripts/analyze_feedback.py

No CSV yet? Click some Real/Fake buttons in the dashboard first.
"""

import csv
import os
import sys
import statistics
from collections import defaultdict

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Signals we want to analyse — must match column names in dev_feedback.csv
NUMERIC_SIGNALS = [
    "composite_score",
    "face_confidence",
    "blink_rate_per_min",
    "ear_variance",
    "iris_drift",
    "head_drift",
    "mouth_var",
    "liveness_score",
    "signal_quality",
    "heart_rate_bpm",
    "lbp_variance",
    "background_correlation",
]


def load_rows(csv_path):
    if not os.path.exists(csv_path):
        print(f"No feedback CSV found at {csv_path}.")
        print("Run main.py with DEV_FEEDBACK_BUTTON=True, click some Real/Fake "
              "buttons after checks, then re-run this script.")
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_split_threshold(real_vals, fake_vals):
    """
    Find the threshold T that best separates 'real' from 'fake' by F1 score.
    Returns (best_T, best_F1, real_above_T_count, fake_below_T_count).
    Convention: signal HIGH = real (works for lbp, face_conf, liveness).
    If real and fake means are reversed (signal LOW = real), the caller
    interprets accordingly.
    """
    if not real_vals or not fake_vals:
        return None, 0.0, 0, 0

    all_vals = sorted(set(real_vals + fake_vals))
    if len(all_vals) < 2:
        return None, 0.0, 0, 0

    best_f1 = -1.0
    best_t  = None
    best_real_correct = 0
    best_fake_correct = 0

    # Try thresholds at the midpoints between adjacent unique values.
    for i in range(len(all_vals) - 1):
        t = (all_vals[i] + all_vals[i + 1]) / 2.0

        # Hypothesis 1: real >= T (signal HIGH = real)
        tp = sum(1 for v in real_vals if v >= t)
        fn = len(real_vals) - tp
        fp = sum(1 for v in fake_vals if v >= t)
        tn = len(fake_vals) - fp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
            best_real_correct = tp
            best_fake_correct = tn

    return best_t, best_f1, best_real_correct, best_fake_correct


def analyze(rows):
    if not rows:
        return

    print(f"\nLoaded {len(rows)} labelled checks from feedback CSV.\n")

    # ── Confusion matrix: system verdict vs developer label ────────────────
    print("=" * 64)
    print("CONFUSION MATRIX  (rows = developer label, cols = system verdict)")
    print("=" * 64)
    counts = defaultdict(int)
    for r in rows:
        gt = r["ground_truth"].strip().lower()
        sv = r["system_verdict"].strip().upper()
        counts[(gt, sv)] += 1
    real_pass = counts[("real", "PASS")]
    real_fail = counts[("real", "FAIL")]
    fake_pass = counts[("fake", "PASS")]
    fake_fail = counts[("fake", "FAIL")]
    n_real = real_pass + real_fail
    n_fake = fake_pass + fake_fail

    print(f"                  system PASS    system FAIL    total")
    print(f"  real (truth):   {real_pass:>10}    {real_fail:>11}    {n_real:>5}")
    print(f"  fake (truth):   {fake_pass:>10}    {fake_fail:>11}    {n_fake:>5}")
    print()
    if n_real > 0:
        tpr = real_pass / n_real
        fnr = real_fail / n_real
        print(f"  Real-user accept rate (TPR):  {tpr*100:.1f}%  ({real_pass}/{n_real} passed)")
        print(f"  Real-user reject rate (FNR):  {fnr*100:.1f}%  ← real users wrongly failed")
    if n_fake > 0:
        fpr = fake_pass / n_fake
        tnr = fake_fail / n_fake
        print(f"  Fake-attack accept rate (FPR):{fpr*100:.1f}%  ← attacks that got through")
        print(f"  Fake-attack reject rate (TNR):{tnr*100:.1f}%  ({fake_fail}/{n_fake} caught)")

    # ── Per-signal separation analysis ─────────────────────────────────────
    print()
    print("=" * 64)
    print("PER-SIGNAL SEPARATION  (suggests threshold that best splits real/fake)")
    print("=" * 64)
    print(f"{'signal':<26} {'real range':<22} {'fake range':<22} {'best T':>8}  F1")
    print("-" * 90)

    for sig in NUMERIC_SIGNALS:
        real_vals = [safe_float(r[sig]) for r in rows
                     if r["ground_truth"].lower() == "real" and r.get(sig)]
        real_vals = [v for v in real_vals if v is not None]
        fake_vals = [safe_float(r[sig]) for r in rows
                     if r["ground_truth"].lower() == "fake" and r.get(sig)]
        fake_vals = [v for v in fake_vals if v is not None]

        if not real_vals or not fake_vals:
            continue

        real_range = f"[{min(real_vals):.3f}, {max(real_vals):.3f}]"
        fake_range = f"[{min(fake_vals):.3f}, {max(fake_vals):.3f}]"

        best_t, best_f1, _, _ = best_split_threshold(real_vals, fake_vals)
        t_str = f"{best_t:.3f}" if best_t is not None else "—"
        f1_str = f"{best_f1:.2f}" if best_t is not None else "—"

        real_mean = statistics.fmean(real_vals)
        fake_mean = statistics.fmean(fake_vals)
        direction = "↑ real" if real_mean > fake_mean else "↓ real"

        print(f"{sig:<26} {real_range:<22} {fake_range:<22} {t_str:>8}  {f1_str}  {direction}")

    # ── Misclassifications worth investigating ─────────────────────────────
    print()
    print("=" * 64)
    print("MISCLASSIFICATIONS  (system disagreed with developer label)")
    print("=" * 64)
    mistakes = [r for r in rows
                if (r["ground_truth"].lower() == "real" and r["system_verdict"] == "FAIL")
                or (r["ground_truth"].lower() == "fake" and r["system_verdict"] == "PASS")]
    if not mistakes:
        print("  None — system agreed with developer on every labelled check.")
    else:
        for r in mistakes:
            print(f"  check #{r['check_number']}: developer={r['ground_truth']}  "
                  f"system={r['system_verdict']}  "
                  f"composite={r['composite_score']}  "
                  f"face={r['face_confidence']}  "
                  f"lbp={r['lbp_variance']}  "
                  f"reason={r['alert_reason'][:60] if r['alert_reason'] else '—'}")

    print()
    print("To tune thresholds: look at the 'best T' column above and update the")
    print("corresponding config.py constant if the F1 score is meaningfully better")
    print("than the current threshold's F1 would be.")
    print()


def main():
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config.DEV_FEEDBACK_CSV_PATH,
    )
    rows = load_rows(csv_path)
    analyze(rows)


if __name__ == "__main__":
    main()
