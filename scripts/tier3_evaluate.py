"""Re-evaluate the archival tier correctly, and compare it to Tier 1 on equal terms.

Three corrections to the first pass, each of which changed how the result reads.

Balanced accuracy rather than raw accuracy. The classes are imbalanced, so a model that
always predicts the biggest class scores 22.8% while 1/n is 16.7%. Calling 16.7%
"chance" inflated every lift figure and made the shuffled-label control (24.06%) look
like a leak when it was majority-class guessing. Macro-averaged recall has chance at
exactly 1/n whatever the class sizes.

Matched class counts. Tier 1 has 60 performers against 12 here, and with 12 classes the
maximum possible lift is 12x, so Tier 1 is re-run on random subsets matched to this
tier's shape and the distribution of those runs is the comparison.

Source items held out, not random tracks. This is the correction that matters most.
Every recording comes from an Archive.org item, one LP transfer digitised once, and
items nest perfectly inside pianists: of 44 items, not one contains two pianists. Under
random k-fold a test track almost always shares its transfer with the training set, so
a model can score by recognising the transfer's artefacts. No arm of the first two
passes could have detected it, controls included.

The fix is StratifiedGroupKFold on the item. Pianists whose tracks all come from one
item are dropped, since for them the question is unanswerable with this data. The gap
between the random-fold and item-held-out figures is itself the measurement: how much
of the apparent signal was the recording.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import features as F
from pianist_id.evaluate import (
    independent_groups,
    leave_one_out,
    loo_normalise,
    multi_item_mask,
    permute_by_item,
)

OUT = ROOT / "results"
NOTES = OUT / "tier3_notes"
CANDIDATES = OUT / "candidate_recordings.csv"
RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
UA = {"User-Agent": "PianistIdentification/0.1"}
N_SUBSAMPLES = 150
N_PERM = 200


def key_of(identifier: str, filename: str) -> str:
    return hashlib.sha1(f"{identifier}/{filename}".encode()).hexdigest()[:16]


def catalogue() -> dict[str, dict]:
    """key -> row for the recordings the pipeline currently selects.

    Scope comes from the pipeline's own select(), not from globbing the cache, which
    outlives the selection rules: tightening MAX_DURATION leaves ineligible
    transcriptions on disk that a glob would readmit. Nor from tier3_index.json, which
    the pipeline rewrites and whose shape has changed between runs.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tier3_pipeline", ROOT / "scripts" / "tier3_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    chosen, _ = mod.select()
    return {key_of(r["identifier"], r["filename"]): r for r in chosen}


def tier3():
    cat = catalogue()
    feats, meta = [], []
    for p in sorted(NOTES.glob("*.json")):
        row = cat.get(p.stem)
        if row is None:
            continue
        notes = json.loads(p.read_text())
        if len(notes) < 200:
            continue
        ns = [F.Note(n["onset_time"], n["midi_note"], n["velocity"],
                     n["offset_time"] - n["onset_time"]) for n in notes]
        ns.sort(key=lambda n: (n.onset, n.pitch))
        f = F.extract(ns)
        if "error" in f:
            continue
        v = F.vector(f)
        if not np.all(np.isfinite(v)):
            continue
        feats.append(v)
        meta.append((row["pianist"], row["work"], row["identifier"],
                     float(row["duration_s"]), len(notes)))
    X = np.array(feats, dtype=float)
    y = np.array([m[0] for m in meta])
    w = np.array([m[1] for m in meta])
    it = np.array([m[2] for m in meta])
    # transcription-quality proxies, for the artefact probe: raw note count and
    # notes recovered per minute. Neither says anything about interpretation.
    aux = np.array([[m[4], m[4] / (m[3] / 60.0)] for m in meta], dtype=float)
    return X, y, w, it, aux


def tier1():
    cached = json.loads((OUT / "asap_features.json").read_text())
    with urllib.request.urlopen(urllib.request.Request(RAW + "metadata.csv", headers=UA),
                                timeout=90) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))
    work_of = {r["midi_performance"]: r["folder"] for r in rows}

    def pid(p):
        m = re.match(r"^([A-Za-z_]+?)\d+M?$", p.split("/")[-1].replace(".mid", ""))
        return m.group(1) if m else None

    X, y, w = [], [], []
    for rel, f in cached.items():
        if "error" in f or f.get("n_notes", 0) < 200 or rel not in work_of:
            continue
        v = F.vector(f)
        p = pid(rel)
        if p and np.all(np.isfinite(v)):
            X.append(v)
            y.append(p)
            w.append(work_of[rel])
    X, y, w = np.array(X), np.array(y), np.array(w)
    # No item hold-out needed here: each ASAP performance is its own session.
    Xn, mask = loo_normalise(X, y, w)
    return Xn, y[mask], w[mask]


def main() -> int:
    from sklearn.preprocessing import OneHotEncoder

    X2, y2, w2, i2, aux2 = tier3()
    # Drop single-item pianists before normalising, so per-work z-scoring is computed
    # over exactly the recordings that will be classified. Merge non-independent items
    # first: a pianist whose items all collapse into one group has no transfer that can
    # be held out.
    g2 = independent_groups(y2, w2, i2)
    keep, dropped = multi_item_mask(y2, g2)
    X2, y2, w2, i2, g2, aux2 = (X2[keep], y2[keep], w2[keep], i2[keep],
                                g2[keep], aux2[keep])
    # The artefact proxies ride along as extra columns so they get the same per-work
    # z-scoring. Globally z-scored they would mostly encode which work (a nine-minute
    # ballade has far more notes than a two-minute waltz), and since pianists differ in
    # repertoire the artefact arm would report repertoire as artefact. What is wanted is
    # note yield unusual for this work.
    n_feat = X2.shape[1]
    both, mask = loo_normalise(np.hstack([X2, aux2]), y2, w2)
    X2n, aux2 = both[:, :n_feat], both[:, n_feat:]
    y2, w2, i2, g2 = y2[mask], w2[mask], i2[mask], g2[mask]

    n2 = len(set(y2))
    chance2 = 1.0 / n2
    print("=" * 78)
    print(f"TIER 2 - transcribed archival audio, {n2} pianists, {len(y2)} recordings, "
          f"{len(set(i2))} source items -> {len(set(g2))} independent groups")
    print("=" * 78)
    print(f"  dropped (no second independent transfer): "
          f"{', '.join(dropped) or 'none'}")
    print("  class sizes:", dict(Counter(y2).most_common()))
    print(f"  {'pianist':<32}{'recordings':>11}{'items':>7}{'groups':>8}")
    for p in sorted(set(y2)):
        sel_p = y2 == p
        print(f"  {p:<32}{int(sel_p.sum()):>11}"
              f"{len(set(i2[sel_p])):>7}{len(set(g2[sel_p])):>8}")
    print()

    P2 = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
        w2.reshape(-1, 1))
    rng = np.random.default_rng(0)

    rows = [
        ("chance", (chance2, 0.0)),
        ("shuffled (group-level)", leave_one_out(X2n, permute_by_item(y2, g2, rng), g2)),
        ("piece identity only", leave_one_out(np.zeros((len(y2), 1)), y2, g2, extra=P2)),
        ("transcription artefacts", leave_one_out(aux2, y2, g2)),
        ("features, one track out", leave_one_out(X2n, y2)),
        ("features, one GROUP out", leave_one_out(X2n, y2, g2)),
        ("piece + features, group out", leave_one_out(X2n, y2, g2, extra=P2)),
    ]
    print(f"  {'arm':<30}{'balanced acc':>14}")
    for name, (m, _s) in rows:
        print(f"  {name:<30}{100*m:>13.2f}%")

    res = dict(rows)
    rand = res["features, one track out"][0]
    held = res["features, one GROUP out"][0]
    print()
    print(f"  one-track-out minus one-item-out: {100*(rand-held):+.2f} points")
    print("  (the same estimator either way; the only difference is whether a test")
    print("   track may share its transfer with the training set, so the gap is the")
    print("   share of the apparent signal that was the recording, not the pianist)")

    # permutation test on the honest arm
    null = []
    for _ in range(N_PERM):
        yp = permute_by_item(y2, g2, rng)
        null.append(leave_one_out(X2n, yp, g2)[0])
    null = np.array(null)
    p_val = float((np.sum(null >= held) + 1) / (N_PERM + 1))
    print()
    print(f"  permutation null (item-level, n={N_PERM}): "
          f"{100*null.mean():.2f}% +/- {100*null.std():.2f}")
    print(f"  observed {100*held:.2f}%   ->   p = {p_val:.4f}")

    # ---- Tier 1, matched -------------------------------------------------
    X1, y1, w1 = tier1()
    counts = Counter(y1)
    want = max(3, round(len(y2) / n2))
    per_class = want
    while per_class > 3 and sum(1 for c in counts.values() if c >= per_class) < n2:
        per_class -= 1
    eligible = [p for p, c in counts.items() if c >= per_class]
    n_match = min(n2, len(eligible))
    print()
    print("=" * 78)
    print(f"TIER 1 matched to Tier 2's shape: {n_match} performers, "
          f"{per_class} recordings each")
    print("=" * 78)
    if per_class != want or n_match != n2:
        print(f"  note: Tier 2 averages {want} recordings/pianist across {n2} pianists; "
              f"Tier 1 supports {n_match} performers at {per_class} each")
    print(f"  pool: {len(y1)} performances, {len(counts)} performers, "
          f"{len(eligible)} eligible")

    accs = []
    for _ in range(N_SUBSAMPLES):
        chosen = rng.choice(eligible, size=n_match, replace=False)
        idx = []
        for p in chosen:
            pool = np.where(y1 == p)[0]
            idx.extend(rng.choice(pool, size=per_class, replace=False))
        idx = np.array(idx)
        accs.append(leave_one_out(X1[idx], y1[idx])[0])
    accs = np.array(accs)

    chance_m = 1.0 / n_match
    pct = float(np.mean(accs < held) * 100)
    sig1 = accs.mean() - chance_m
    sig2 = held - chance2
    print(f"  Tier 1 matched              {100*accs.mean():>7.2f}% +/- "
          f"{100*accs.std():.2f}  ({N_SUBSAMPLES} subsamples)")
    print(f"  Tier 2 item held out        {100*held:>7.2f}%")
    print(f"  chance                      {100*chance_m:>7.2f}%")
    print()
    print(f"  above-chance signal retained: {100*sig2/max(sig1, 1e-9):.0f}%")
    print(f"  Tier 2 sits at the {pct:.0f}th percentile of matched Tier 1 runs")
    print()
    print("  Caveat kept from the previous pass: only a handful of Tier 1 performers")
    print("  go this deep, so the subsamples overlap heavily and the sd understates")
    print("  the true spread.")

    (OUT / "tier_comparison.json").write_text(json.dumps({
        "tier3": {k: list(v) for k, v in res.items()},
        "tier3_n_pianists": n2, "tier3_n_recordings": int(len(y2)),
        "tier3_n_items": int(len(set(i2))),
        "tier3_n_groups": int(len(set(g2))),
        "tier3_dropped_single_item": dropped,
        "random_minus_grouped": float(rand - held),
        "permutation_p": p_val,
        "permutation_null_mean": float(null.mean()),
        "permutation_null_sd": float(null.std()),
        "tier1_matched_mean": float(accs.mean()), "tier1_matched_sd": float(accs.std()),
        "per_class": per_class, "n_matched_classes": n_match,
        "chance": chance2,
        "signal_retained": float(sig2 / max(sig1, 1e-9)), "percentile": pct,
    }, indent=2), encoding="utf-8")
    print("\n  -> results/tier_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
