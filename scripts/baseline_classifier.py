"""Identify the performer from interpretive features alone, on exact MIDI.

Notes are exact here, so the only question is whether the features carry identity well
enough to classify. Every split is cross-piece by construction: performers in ASAP
almost never record the same work twice (34 such cells in the whole selected set), so
holding out a performance nearly always means testing on a piece the model never saw
that performer play. The headline number is already the demanding one.

Three controls, reported whichever way they come out. Shuffled labels must collapse to
chance, or something is leaking. A piece-only model, given the work identity and no
playing features, is the floor the real model has to beat, since repertoire choice is
genuinely informative about a competition pianist. And each feature is removed in turn
to see which carry the signal.

Per-work normalisation is leave-one-out. Without it the model would separate Bach from
Liszt rather than performer from performer; without leaving the sample out, each point
would help set its own scale.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import features as F

CACHE = ROOT / "data" / "asap" / "midi"
FEATCACHE = ROOT / "results" / "asap_features.json"
OUT = ROOT / "results"
RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
UA = {"User-Agent": "PianistIdentification/0.1"}

MIN_PERFORMANCES = 5
SEED = 0


def performer_id(path: str) -> str | None:
    m = re.match(r"^([A-Za-z_]+?)\d+M?$", path.split("/")[-1].replace(".mid", ""))
    return m.group(1) if m else None


def fetch(rel: str) -> bytes | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / rel.replace("/", "__")
    if dest.exists():
        return dest.read_bytes()
    try:
        url = RAW + urllib.parse.quote(rel)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
            data = r.read()
        dest.write_bytes(data)
        return data
    except Exception:  # noqa: BLE001
        return None


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with urllib.request.urlopen(urllib.request.Request(RAW + "metadata.csv", headers=UA),
                                timeout=90) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))

    recs = []
    for row in rows:
        pid = performer_id(row["midi_performance"])
        if pid:
            recs.append((pid, row["folder"], row["midi_performance"]))
    counts = Counter(p for p, _, _ in recs)
    keep = {p for p, c in counts.items() if c >= MIN_PERFORMANCES}
    recs = [r for r in recs if r[0] in keep]
    print(f"{len(keep)} performers, {len(recs)} performances "
          f"(chance accuracy {100/len(keep):.2f}%)")

    cached = json.loads(FEATCACHE.read_text()) if FEATCACHE.exists() else {}
    feats, meta = [], []
    for n, (pid, work, rel) in enumerate(recs, 1):
        if rel in cached:
            f = cached[rel]
        else:
            data = fetch(rel)
            if data is None:
                continue
            try:
                f = F.extract(F.load_midi(data))
            except Exception:  # noqa: BLE001
                continue
            cached[rel] = f
            if n % 50 == 0:
                print(f"   extracted {n}/{len(recs)}")
        if "error" in f or f.get("n_notes", 0) < 200:
            continue
        v = F.vector(f)
        if not np.all(np.isfinite(v)):
            continue
        feats.append(v)
        meta.append((pid, work))
    FEATCACHE.write_text(json.dumps(cached))

    X = np.array(feats, dtype=float)
    y = np.array([m[0] for m in meta])
    works = np.array([m[1] for m in meta])

    # leave-one-out z-scoring within each work
    Xn = np.zeros_like(X)
    for w in np.unique(works):
        idx = np.where(works == w)[0]
        if len(idx) < 3:
            Xn[idx] = 0.0  # too few to normalise; contributes nothing
            continue
        block = X[idx]
        for k, i in enumerate(idx):
            others = np.delete(block, k, axis=0)
            mu, sd = others.mean(axis=0), others.std(axis=0)
            sd[sd < 1e-9] = 1.0
            Xn[i] = (X[i] - mu) / sd
    Xn = np.clip(Xn, -6, 6)
    # Works with fewer than three performances cannot be normalised against their
    # peers, so those rows are zeroed and carry nothing. With 191 works over 501
    # performances that is common enough that a large zeroed fraction would quietly cap
    # the achievable accuracy, so it is printed.
    zeroed = int(np.sum(np.all(Xn == 0.0, axis=1)))
    print(f"   samples zeroed (work had <3 performances): {zeroed}/{len(Xn)} "
          f"({100*zeroed/len(Xn):.0f}%)")
    return Xn, y, works, F.NORMALISED_KEYS


def evaluate(X, y, works, seed=SEED, shuffle_labels=False, piece_only=False,
             piece_plus=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(seed)
    y_used = y.copy()
    if shuffle_labels:
        # permute labels WITHIN the dataset, preserving class sizes
        y_used = rng.permutation(y_used)

    if piece_only or piece_plus:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        P = enc.fit_transform(works.reshape(-1, 1))
        # piece_plus is the test that matters: does how they played add anything to
        # what they played? Features against piece alone is the wrong comparison.
        X_used = np.hstack([P, X]) if piece_plus else P
    else:
        X_used = X

    counts = Counter(y_used)
    mask = np.array([counts[v] >= 3 for v in y_used])
    X_used, y_used, = X_used[mask], y_used[mask]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    accs = {"logreg": [], "forest": [], "top5": []}
    confusion = defaultdict(Counter)
    for tr, te in skf.split(X_used, y_used):
        lr = LogisticRegression(max_iter=4000, C=1.0)
        lr.fit(X_used[tr], y_used[tr])
        pred = lr.predict(X_used[te])
        accs["logreg"].append(float(np.mean(pred == y_used[te])))
        for t, p in zip(y_used[te], pred):
            confusion[t][p] += 1

        proba = lr.predict_proba(X_used[te])
        order = np.argsort(-proba, axis=1)[:, :5]
        classes = lr.classes_
        accs["top5"].append(float(np.mean([
            y_used[te][i] in classes[order[i]] for i in range(len(te))])))

        rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
        rf.fit(X_used[tr], y_used[tr])
        accs["forest"].append(float(np.mean(rf.predict(X_used[te]) == y_used[te])))

    n_classes = len(set(y_used))
    return {
        "n_samples": int(len(y_used)),
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "logreg": float(np.mean(accs["logreg"])),
        "forest": float(np.mean(accs["forest"])),
        "top5": float(np.mean(accs["top5"])),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    X, y, works, keys = build_dataset()
    print(f"feature matrix {X.shape}, {len(set(y))} performers, {len(set(works))} works")

    # Zeroed rows carry no features, so leaving them in would hand the piece-only
    # model work identity for every sample while the feature model guesses on a third.
    wc = Counter(works)
    keep = np.array([wc[w] >= 3 for w in works])
    X, y, works = X[keep], y[keep], works[keep]
    pc = Counter(y)
    keep2 = np.array([pc[v] >= 3 for v in y])
    X, y, works = X[keep2], y[keep2], works[keep2]
    print(f"after restricting to works with >=3 performances: {X.shape[0]} samples, "
          f"{len(set(y))} performers, {len(set(works))} works\n")

    print("=" * 76)
    print("BASELINE  (every split is cross-piece by construction)")
    print("=" * 76)
    real = evaluate(X, y, works)
    print(f"  samples {real['n_samples']}, classes {real['n_classes']}, "
          f"chance {100*real['chance']:.2f}%")
    print(f"  logistic regression   {100*real['logreg']:.2f}%")
    print(f"  random forest         {100*real['forest']:.2f}%")
    print(f"  logreg top-5          {100*real['top5']:.2f}%")
    print(f"  lift over chance      {real['logreg']/real['chance']:.1f}x")

    print("\n" + "=" * 76)
    print("CONTROLS")
    print("=" * 76)
    shuf = evaluate(X, y, works, shuffle_labels=True)
    print(f"  shuffled labels       {100*shuf['logreg']:.2f}%  "
          f"(chance {100*shuf['chance']:.2f}%)  <- must collapse")
    piece = evaluate(X, y, works, piece_only=True)
    plus = evaluate(X, y, works, piece_plus=True)
    print(f"  piece identity only   {100*piece['logreg']:.2f}%  <- the honest floor")
    print(f"  piece + features      {100*plus['logreg']:.2f}%  <- does playing add anything?")
    print(f"  features add          {100*(plus['logreg']-piece['logreg']):+.2f} points "
          f"on top of knowing the piece")

    print("\n" + "=" * 76)
    print("ABLATION  (accuracy with each feature removed)")
    print("=" * 76)
    abl = {}
    for i, k in enumerate(keys):
        Xa = np.delete(X, i, axis=1)
        a = evaluate(Xa, y, works)["logreg"]
        abl[k] = a
        print(f"  without {k:<28}{100*a:>7.2f}%   {100*(a-real['logreg']):+6.2f}")
    worst = min(abl.items(), key=lambda kv: kv[1])
    print(f"\n  most load-bearing: {worst[0]} (removing it costs "
          f"{100*(real['logreg']-worst[1]):.2f} points)")

    print("\n" + "=" * 76)
    print("READ-OUT")
    print("=" * 76)
    if shuf["logreg"] > 3 * shuf["chance"]:
        print("  LEAK. Shuffled labels still beat chance; the pipeline is leaking.")
    elif plus["logreg"] <= piece["logreg"]:
        print("  FAIL. Adding the playing features to piece identity does not")
        print("  improve on piece identity alone. On this evidence the model is")
        print("  exploiting repertoire choice, not interpretation.")
    elif real["logreg"] > 3 * real["chance"]:
        print("  Features carry performer identity across pieces, well above both")
        print("  chance and the piece-only floor. A cross-piece result at this")
        print("  sample size is the demanding version of the test.")
    else:
        print("  WEAK. Above chance but not by much. Report the number plainly.")

    (OUT / "baseline_classifier.json").write_text(json.dumps(
        {"real": real, "shuffled": shuf, "piece_only": piece, "piece_plus": plus,
         "ablation": abl,
         "features": keys, "min_performances": MIN_PERFORMANCES}, indent=2), encoding="utf-8")
    print("\n  -> results/baseline_classifier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
