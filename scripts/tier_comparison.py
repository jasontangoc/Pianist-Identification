"""Compare the MIDI and archival tiers on equal terms.

The headline numbers are not comparable as they stand: one classifies 60 performers
(chance 1.67%) and the other 6 pianists (chance 16.7%). Neither raw accuracy nor lift
survives that gap, since with 6 classes the maximum lift is 6x against 60x, so the
archival tier is capped below the other's observed lift before a note is transcribed.

The fix is to re-run the MIDI tier subsampled to the archival tier's shape, same class
count and same recordings per class, repeatedly, and compare the distributions. Then
the only systematic difference left is that one reads notes from a MIDI file and the
other recovers them from audio.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import features as F

OUT = ROOT / "results"
N_SUBSAMPLES = 200


def load_tier1():
    """Rebuild Tier 1's matrix from the cached per-performance features."""
    cached = json.loads((OUT / "asap_features.json").read_text())
    import csv
    import io
    import re
    import urllib.request

    RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
    with urllib.request.urlopen(
        urllib.request.Request(RAW + "metadata.csv", headers={"User-Agent": "PianistIdentification/0.1"}),
        timeout=90,
    ) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))
    work_of = {r["midi_performance"]: r["folder"] for r in rows}

    def pid(p):
        m = re.match(r"^([A-Za-z_]+?)\d+M?$", p.split("/")[-1].replace(".mid", ""))
        return m.group(1) if m else None

    X, y, works = [], [], []
    for rel, f in cached.items():
        if "error" in f or f.get("n_notes", 0) < 200:
            continue
        v = F.vector(f)
        if not np.all(np.isfinite(v)):
            continue
        p = pid(rel)
        if p is None or rel not in work_of:
            continue
        X.append(v)
        y.append(p)
        works.append(work_of[rel])
    return np.array(X), np.array(y), np.array(works)


def normalise(X, y, works):
    """Leave-one-out per-work z-scoring, then Tier-1's restriction."""
    Xn = np.zeros_like(X)
    for w in np.unique(works):
        idx = np.where(works == w)[0]
        if len(idx) < 3:
            continue
        block = X[idx]
        for k, i in enumerate(idx):
            others = np.delete(block, k, axis=0)
            mu, sd = others.mean(axis=0), others.std(axis=0)
            sd[sd < 1e-9] = 1.0
            Xn[i] = (X[i] - mu) / sd
    Xn = np.clip(Xn, -6, 6)
    wc = Counter(works)
    keep = np.array([wc[w] >= 3 for w in works])
    return Xn[keep], y[keep], works[keep]


def accuracy(X, y, seeds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    out = []
    for s in range(seeds):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=s)
        a = []
        for tr, te in skf.split(X, y):
            lr = LogisticRegression(max_iter=4000).fit(X[tr], y[tr])
            a.append(np.mean(lr.predict(X[te]) == y[te]))
        out.append(np.mean(a))
    return float(np.mean(out))


def main() -> int:
    t2_path = OUT / "tier3_result.json"
    if not t2_path.exists():
        print("Tier 2 has not finished yet; run scripts/tier3_pipeline.py first.")
        return 1
    t2 = json.loads(t2_path.read_text())
    n_classes = int(t2["n_pianists"])
    per_class = max(3, round(t2["n_recordings"] / n_classes))
    print(f"Tier 2 shape: {n_classes} pianists, {t2['n_recordings']} recordings "
          f"(~{per_class} each), chance {100*t2['chance']:.1f}%")

    X, y, works = load_tier1()
    Xn, y, works = normalise(X, y, works)
    counts = Counter(y)
    eligible = [p for p, c in counts.items() if c >= per_class]
    print(f"Tier 1 pool: {len(y)} performances, {len(counts)} performers, "
          f"{len(eligible)} with >={per_class} performances\n")
    if len(eligible) < n_classes:
        print("Not enough Tier 1 performers to match Tier 2's shape.")
        return 1

    rng = np.random.default_rng(0)
    accs = []
    for _ in range(N_SUBSAMPLES):
        chosen = rng.choice(eligible, size=n_classes, replace=False)
        idx = []
        for p in chosen:
            pool = np.where(y == p)[0]
            idx.extend(rng.choice(pool, size=per_class, replace=False))
        idx = np.array(idx)
        accs.append(accuracy(Xn[idx], y[idx], seeds=2))
    accs = np.array(accs)

    t2_acc = t2["features"][0]
    chance = 1.0 / n_classes
    pct = float(np.mean(accs < t2_acc) * 100)

    print("=" * 74)
    print(f"MATCHED COMPARISON  ({n_classes} classes, {per_class} recordings each)")
    print("=" * 74)
    print(f"  chance                           {100*chance:>7.2f}%")
    print(f"  Tier 1, note-exact MIDI          {100*accs.mean():>7.2f}%  "
          f"+/- {100*accs.std():.2f}   (n={N_SUBSAMPLES} subsamples)")
    print(f"  Tier 2, transcribed audio        {100*t2_acc:>7.2f}%")
    print(f"  retained                         {100*t2_acc/max(accs.mean(),1e-9):>7.1f}% "
          f"of Tier 1 accuracy")
    print(f"  Tier 2 sits at the {pct:.0f}th percentile of the Tier 1 distribution")
    print()
    above = (t2_acc - chance) / max(accs.mean() - chance, 1e-9)
    print(f"  above-chance signal retained     {100*above:>7.1f}%")
    print("  (accuracy minus chance, which is what shrinks when notes get noisy)")

    (OUT / "tier_comparison.json").write_text(json.dumps({
        "n_classes": n_classes, "per_class": per_class, "chance": chance,
        "tier1_matched_mean": float(accs.mean()), "tier1_matched_sd": float(accs.std()),
        "tier2": t2_acc, "retained_fraction": float(t2_acc / max(accs.mean(), 1e-9)),
        "above_chance_retained": float(above), "percentile": pct,
    }, indent=2), encoding="utf-8")
    print("\n  -> results/tier_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
