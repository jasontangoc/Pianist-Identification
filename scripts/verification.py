"""Verification: were these two performances played by the same pianist?

Identification -- given one performance, which of N pianists played it -- is the hardest
form of the question and is punished for having more pianists: 7.90% balanced accuracy
on 23 performers, with only 6 of them ever named correctly. The pairwise form is the
standard framing for speaker and authorship problems and suits this data better, for
reasons about the estimator rather than about getting a bigger number. ROC AUC over
pairs does not depend on the pool size, so scores are comparable across pools. Each
performance appears in every pair it belongs to, as evidence both for and against. And
verification fits one decision boundary where identification fits N.

It is not more independent data: 516 same-person pairs come from 357 performances by 102
performers, so the independent information is still bounded by the performers, which is
why the null below is computed at session level.

This question was asked on MIDI before any classification was run, and found a pianist
reliably closer to themselves (85% of same-performer pairs closer, Cohen's d +1.49).
That earlier pass pooled every pair without holding out recording sessions, which is the
flaw that inflated the archival tier, so its number cannot stand as measured.

Three controls. A pair must come from two different sessions, since two performances from
one competition session share a hall, a piano, a tuning and an afternoon; this costs 9%
of the same-person pairs. Both classes are restricted to different works, because
performers almost never repeat a work, so left alone "same piece" is a near-perfect cue
for "different people" and the test would partly measure piece identity; the headline is
cross-piece by construction. And labels are permuted across sessions, never across
performances, so one session holding one performer survives the shuffle.

Uncertainty comes from leaving out one performer at a time. A bootstrap drawing a
performer twice would pair their own performances as a different-person pair at distance
zero.

    python scripts/verification.py
    python scripts/verification.py --perm 2000
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from pianist_id import features as F
from pianist_id.evaluate import loo_normalise, permute_by_group

import tier2_modern_pipeline as P

OUT = ROOT / "results"
MIDI = ROOT / "data" / "asap" / "midi"
NOTES = OUT / "tier2_modern_notes"
CACHE = OUT / "verification_features.npz"
ASAP_RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
UA = {"User-Agent": "PianistIdentification/0.1"}

# Looser minimums than identification. A performer needs two performances in two
# different sessions, since one whose performances all sit in one session contributes
# only same-session pairs, which the control discards anyway. Nothing needs three of
# anything: the three-per-work rule exists so a per-work standard deviation can be
# taken, and pooled scaling removes that need.
MIN_PERFORMANCES_PER_WORK = 2
MIN_SESSIONS_PER_PERFORMER = 2
MIN_PERFORMANCES_PER_PERFORMER = 2
N_PERM = 500


def select(segs):
    """The verification thresholds, applied to a fixed point."""
    pool = list(segs)
    while True:
        before = len(pool)
        wc = Counter(s["work"] for s in pool)
        pool = [s for s in pool if wc[s["work"]] >= MIN_PERFORMANCES_PER_WORK]
        pc = Counter(s["performer"] for s in pool)
        pool = [s for s in pool
                if pc[s["performer"]] >= MIN_PERFORMANCES_PER_PERFORMER]
        sess = defaultdict(set)
        for s in pool:
            sess[s["performer"]].add(s["session"])
        pool = [s for s in pool
                if len(sess[s["performer"]]) >= MIN_SESSIONS_PER_PERFORMER]
        if len(pool) == before:
            break
    if not pool:
        raise SystemExit("selection is empty")
    wc = Counter(s["work"] for s in pool)
    pc = Counter(s["performer"] for s in pool)
    sess = defaultdict(set)
    for s in pool:
        sess[s["performer"]].add(s["session"])
    assert min(wc.values()) >= MIN_PERFORMANCES_PER_WORK
    assert min(pc.values()) >= MIN_PERFORMANCES_PER_PERFORMER
    assert min(len(v) for v in sess.values()) >= MIN_SESSIONS_PER_PERFORMER
    pool.sort(key=lambda s: (s["session"], s["start"]))
    return pool


def local_midi(rel: str) -> Path:
    p = MIDI / rel.replace("/", "__")
    if not p.exists():
        MIDI.mkdir(parents=True, exist_ok=True)
        try:
            data = urllib.request.urlopen(urllib.request.Request(
                ASAP_RAW + urllib.parse.quote(rel), headers=UA), timeout=60).read()
            p.write_bytes(data)
        except Exception:  # noqa: BLE001
            return p
    return p


def load_features(pool, use_cache=True):
    """Ground-truth MIDI features, cached against the exact set of rows."""
    keys = sorted(s["key"] for s in pool)
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["keys"]) == keys:
            print(f"  features from cache ({len(keys)} rows)")
            return z["X"], [tuple(m) for m in z["meta"].tolist()]
        print("  cache does not match the current selection; re-extracting")
    X, meta = [], []
    for n, s in enumerate(pool, 1):
        if n % 50 == 0:
            print(f"    {n}/{len(pool)}")
        mp = local_midi(s["midi"])
        if not mp.exists():
            continue
        try:
            f = F.extract(F.load_midi(mp))
        except Exception:  # noqa: BLE001
            continue
        if "error" in f:
            continue
        v = F.vector(f)
        if not np.all(np.isfinite(v)):
            continue
        X.append(v)
        meta.append((s["performer"], s["work"], s["session"], s["key"]))
    X = np.array(X)
    np.savez(CACHE, X=X, meta=np.array(meta, dtype=object),
             keys=np.array(sorted(m[3] for m in meta), dtype=object))
    return X, meta


def pair_index(works, sessions, cross_session=True, cross_work=True,
               same_work_only=False):
    """Upper-triangle pair indices surviving the requested controls."""
    n = len(works)
    ii, jj = np.triu_indices(n, k=1)
    keep = np.ones(len(ii), dtype=bool)
    if cross_session:
        keep &= sessions[ii] != sessions[jj]
    if same_work_only:
        keep &= works[ii] == works[jj]
    elif cross_work:
        keep &= works[ii] != works[jj]
    return ii[keep], jj[keep]


def auc_and_eer(scores, positive):
    """ROC AUC and equal-error rate. `scores`: higher means more same-person."""
    from sklearn.metrics import roc_auc_score, roc_curve
    if positive.sum() == 0 or (~positive).sum() == 0:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(positive, scores))
    fpr, tpr, _ = roc_curve(positive, scores)
    fnr = 1 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return auc, float((fpr[k] + fnr[k]) / 2)


def score_pairs(Xn, ii, jj):
    """Negative Euclidean distance, so that higher means more alike."""
    return -np.linalg.norm(Xn[ii] - Xn[jj], axis=1)


def evaluate(Xn, y, works, sessions, ii, jj):
    same = y[ii] == y[jj]
    return auc_and_eer(score_pairs(Xn, ii, jj), same), int(same.sum()), int((~same).sum())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    n_perm = int(argv[argv.index("--perm") + 1]) if "--perm" in argv else N_PERM

    pool = select(P.segments())
    print(f"pool: {len(pool)} performances, "
          f"{len({s['performer'] for s in pool})} performers, "
          f"{len({s['work'] for s in pool})} works, "
          f"{len({s['session'] for s in pool})} sessions")
    X, meta = load_features(pool, use_cache="--no-cache" not in argv)
    if len(meta) < 50:
        print(f"only {len(meta)} performances with usable features")
        return 1

    y = np.array([m[0] for m in meta])
    w = np.array([m[1] for m in meta])
    g = np.array([m[2] for m in meta])

    Xn, mask = loo_normalise(X, y, w, min_per_work=MIN_PERFORMANCES_PER_WORK,
                             min_per_label=MIN_PERFORMANCES_PER_PERFORMER,
                             scale="pooled")
    y, w, g = y[mask], w[mask], g[mask]
    dropped = int((~mask).sum())

    print("=" * 78)
    print(f"VERIFICATION - same pianist or not. {len(y)} performances, "
          f"{len(set(y))} performers, {len(set(w))} works, {len(set(g))} sessions")
    print("=" * 78)
    if dropped:
        print(f"  {dropped} rows dropped by the normalisation mask")

    # The same-work arms explain the headline rather than compete with it. The earlier
    # MIDI pass scored AUC 0.849 comparing performances of the same work, so if the
    # cross-work headline lands far below that, the gap is either the session control or
    # cross-piece generalisation, and only a same-work arm separates the two. Performers
    # almost never repeat a work, so those arms rest on a few dozen pairs.
    arms = {}
    for name, xs, xw, sw in (
            ("cross-session + cross-work  (headline)", True, True, False),
            ("cross-work only, sessions allowed", False, True, False),
            ("cross-session only, same work allowed", True, False, False),
            ("no controls, every pair pooled", False, False, False),
            ("SAME work only, cross-session", True, False, True),
            ("SAME work only, no session control", False, False, True)):
        ii, jj = pair_index(w, g, cross_session=xs, cross_work=xw,
                            same_work_only=sw)
        (auc, eer), ns, nd = evaluate(Xn, y, w, g, ii, jj)
        arms[name] = {"auc": auc, "eer": eer, "n_same": ns, "n_diff": nd}
        print(f"\n  {name}")
        print(f"    AUC {auc:.3f}   equal-error rate {100*eer:.1f}%   "
              f"({ns} same-person pairs, {nd} different)")

    ii, jj = pair_index(w, g, True, True)
    head = arms["cross-session + cross-work  (headline)"]

    # ---- shuffled control ----------------------------------------------
    rng = np.random.default_rng(0)
    ysh = permute_by_group(y, g, rng)
    (auc_sh, _), _, _ = evaluate(Xn, ysh, w, g, ii, jj)
    print(f"\n  CONTROLS")
    print(f"    shuffled labels (session-level), single draw   AUC {auc_sh:.3f}")

    # Identification found 6 of 23 performers carrying the whole score, so a good
    # average built from a handful of distinctive players is a different finding from a
    # good average across everyone.
    per = {}
    for lab in sorted(set(y)):
        sel = (y[ii] == lab) | (y[jj] == lab)
        if not sel.any():
            continue
        same = y[ii][sel] == y[jj][sel]
        if same.sum() == 0 or (~same).sum() == 0:
            continue
        a, _ = auc_and_eer(score_pairs(Xn, ii[sel], jj[sel]), same)
        per[lab] = a
    vals = np.array(list(per.values()))
    print(f"\n  WHO IS VERIFIABLE  (per-performer AUC, {len(per)} performers)")
    print(f"    median {np.median(vals):.3f}   "
          f"quartiles {np.percentile(vals,25):.3f} / {np.percentile(vals,75):.3f}   "
          f"range {vals.min():.3f} to {vals.max():.3f}")
    print(f"    above 0.5: {int((vals > 0.5).sum())} of {len(vals)}   "
          f"above 0.7: {int((vals > 0.7).sum())}")

    # ---- jackknife over performers -------------------------------------
    jack = []
    for lab in sorted(set(y)):
        keep = y != lab
        sel = keep[ii] & keep[jj]
        a, _ = auc_and_eer(score_pairs(Xn, ii[sel], jj[sel]),
                           y[ii][sel] == y[jj][sel])
        jack.append(a)
    jack = np.array(jack)
    print(f"\n  SENSITIVITY  (leave one performer out, {len(jack)} refits)")
    print(f"    AUC range {jack.min():.3f} to {jack.max():.3f}, "
          f"sd {jack.std():.4f}")

    # ---- session-level permutation null --------------------------------
    nrng = np.random.default_rng(1)
    null = np.empty(n_perm)
    for k in range(n_perm):
        yp = permute_by_group(y, g, nrng)
        null[k], _ = auc_and_eer(score_pairs(Xn, ii, jj), yp[ii] == yp[jj])
    p = float((np.sum(null >= head["auc"]) + 1) / (n_perm + 1))
    print(f"\n  PERMUTATION NULL  (session-level, n={n_perm})")
    print(f"    null AUC {null.mean():.3f} +/- {null.std():.3f}   "
          f"observed {head['auc']:.3f}   ->  p = {p:.4f}")

    (OUT / "verification_result.json").write_text(json.dumps({
        "n_performances": int(len(y)), "n_performers": len(set(y)),
        "n_works": int(len(set(w))), "n_sessions": int(len(set(g))),
        "arms": arms, "shuffled_auc": auc_sh,
        "per_performer_auc": per,
        "jackknife": {"min": float(jack.min()), "max": float(jack.max()),
                      "sd": float(jack.std())},
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
        "n_perm": n_perm, "p": p,
    }, indent=2), encoding="utf-8")
    print("\n  -> results/verification_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
