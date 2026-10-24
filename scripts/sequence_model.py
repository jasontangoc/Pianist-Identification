"""Does a sequence model recover rubato signal that the summaries discard?

The baseline left a specific question. Summary statistics of timing variability
(`ioi_cv`, `ioi_log_sd`, `ioi_ratio_p90_p10`) carried no measurable identity while
per-chord melodic quantities did, which suggests a whole-piece summary throws away
where the stretching happens. If so, a model that sees the curve itself should beat
8.94%. If not, timing variability genuinely carries less identity than melodic voicing,
which is the more interesting answer. Either one gets reported.

The same performers, performances and splits as the baseline, so only the
representation changes. Three arms: logistic regression on 11 numbers, a 1D CNN over
the 4-channel chord series, and the CNN probabilities combined with the summaries.

Controls: shuffled labels must collapse, and windows from one performance never cross a
train/test boundary, since they are near-duplicates.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import features as F
from pianist_id import sequences as S

CACHE = ROOT / "data" / "asap" / "midi"
OUT = ROOT / "results"
RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
UA = {"User-Agent": "PianistIdentification/0.1"}
MIN_PERFORMANCES = 5
WIN, STRIDE = 64, 24
SEEDS = 3


def performer_id(path: str) -> str | None:
    m = re.match(r"^([A-Za-z_]+?)\d+M?$", path.split("/")[-1].replace(".mid", ""))
    return m.group(1) if m else None


def fetch(rel: str) -> bytes | None:
    dest = CACHE / rel.replace("/", "__")
    return dest.read_bytes() if dest.exists() else None


def build():
    with urllib.request.urlopen(urllib.request.Request(RAW + "metadata.csv", headers=UA),
                                timeout=90) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))
    recs = []
    for row in rows:
        pid = performer_id(row["midi_performance"])
        if pid:
            recs.append((pid, row["folder"], row["midi_performance"]))
    counts = Counter(p for p, _, _ in recs)
    recs = [r for r in recs if counts[r[0]] >= MIN_PERFORMANCES]

    feats, seqs, meta = [], [], []
    for n, (pid, work, rel) in enumerate(recs, 1):
        data = fetch(rel)
        if data is None:
            continue
        try:
            notes = F.load_midi(data)
            f = F.extract(notes)
            series = S.chord_series(notes)
        except Exception:  # noqa: BLE001
            continue
        if "error" in f or f.get("n_notes", 0) < 200 or not series:
            continue
        v = F.vector(f)
        w = S.windows(series, WIN, STRIDE)
        if not np.all(np.isfinite(v)) or len(w) == 0:
            continue
        feats.append(v)
        seqs.append(w)
        meta.append((pid, work))
        if n % 100 == 0:
            print(f"   {n}/{len(recs)}")

    X = np.array(feats, dtype=float)
    y = np.array([m[0] for m in meta])
    works = np.array([m[1] for m in meta])

    # same leave-one-out per-work normalisation as the baseline
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

    # identical restriction to the baseline, so the numbers are comparable
    wc = Counter(works)
    keep = np.array([wc[w] >= 3 for w in works])
    Xn, y, works = Xn[keep], y[keep], works[keep]
    seqs = [s for s, k in zip(seqs, keep) if k]
    pc = Counter(y)
    keep2 = np.array([pc[v] >= 3 for v in y])
    Xn, y, works = Xn[keep2], y[keep2], works[keep2]
    seqs = [s for s, k in zip(seqs, keep2) if k]
    return Xn, seqs, y, works


class CNN:
    """Small 1D CNN with global pooling.

    Global pooling is what makes cross-piece transfer possible: the model can only use
    what a short span looks like, never where in the piece it sits.
    """

    def __init__(self, n_classes: int, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        C = len(S.CHANNELS)
        self.net = nn.Sequential(
            nn.Conv1d(C, 48, 7, padding=3), nn.BatchNorm1d(48), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, n_classes),
        )

    def fit(self, Xw, yw, epochs=30, lr=2e-3, bs=128):
        torch, nn = self.torch, self.torch.nn
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xw); yt = torch.tensor(yw)
        n = len(Xt)
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 4:
                    continue
                opt.zero_grad()
                loss = lossf(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict_proba_windows(self, Xw):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(Xw), 512):
                out.append(torch.softmax(self.net(torch.tensor(Xw[i:i + 512])), dim=1).numpy())
        return np.vstack(out) if out else np.zeros((0, 1))


def run(Xn, seqs, y, works, shuffle=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    classes = sorted(set(y))
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_i = np.array([cls_idx[v] for v in y])

    res = {"summary": [], "sequence": [], "both": []}
    for seed in range(SEEDS):
        y_used = y_i.copy()
        if shuffle:
            y_used = np.random.default_rng(seed).permutation(y_used)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xn, y_used):
            # windows inherit their performance's split assignment
            Xw_tr = np.concatenate([seqs[i] for i in tr])
            yw_tr = np.concatenate([np.full(len(seqs[i]), y_used[i]) for i in tr])

            lr = LogisticRegression(max_iter=4000).fit(Xn[tr], y_used[tr])
            p_sum = lr.predict_proba(Xn[te])
            res["summary"].append(np.mean(lr.classes_[p_sum.argmax(1)] == y_used[te]))

            cnn = CNN(len(classes), seed=seed).fit(Xw_tr, yw_tr)
            p_seq = []
            for i in te:
                pw = cnn.predict_proba_windows(seqs[i])
                p_seq.append(pw.mean(axis=0) if len(pw) else np.zeros(len(classes)))
            p_seq = np.array(p_seq)
            res["sequence"].append(np.mean(p_seq.argmax(1) == y_used[te]))

            # combine: map the logreg probabilities onto the full class axis
            full = np.zeros_like(p_seq)
            for j, c in enumerate(lr.classes_):
                full[:, c] = p_sum[:, j]
            res["both"].append(np.mean((full + p_seq).argmax(1) == y_used[te]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}, len(classes)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("building dataset (reusing cached MIDI)...")
    Xn, seqs, y, works = build()
    nwin = sum(len(s) for s in seqs)
    print(f"{len(y)} performances, {len(set(y))} performers, {nwin} windows "
          f"({WIN} chords, stride {STRIDE})\n")

    real, n_classes = run(Xn, seqs, y, works)
    chance = 1.0 / n_classes
    print("=" * 70)
    print(f"RESULTS  {n_classes} performers, chance {100*chance:.2f}%, {SEEDS} seeds x 3 folds")
    print("=" * 70)
    for k in ("summary", "sequence", "both"):
        m, s = real[k]
        print(f"  {k:<12}{100*m:>7.2f}%  +/- {100*s:.2f}")
    print()
    shuf, _ = run(Xn, seqs, y, works, shuffle=True)
    print(f"  shuffled labels, sequence model: {100*shuf['sequence'][0]:.2f}% "
          f"(chance {100*chance:.2f}%)  <- must collapse")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    diff = real["sequence"][0] - real["summary"][0]
    both = real["both"][0] - real["summary"][0]
    if shuf["sequence"][0] > 3 * chance:
        print("  LEAK in the sequence pipeline; windows are crossing the split.")
    elif diff > 2 * real["sequence"][1]:
        print(f"  Sequence model BEATS summaries by {100*diff:+.2f} points.")
        print("  The tempo curve carries identity that whole-piece statistics")
        print("  discard, which is what the summary features predicted.")
    elif both > 2 * real["both"][1]:
        print(f"  Sequence alone does not beat summaries ({100*diff:+.2f}), but")
        print(f"  combining them gains {100*both:+.2f} points, so the curve carries")
        print("  something the summaries miss even if it is weaker on its own.")
    else:
        print(f"  Sequence model does NOT beat summaries ({100*diff:+.2f} points).")
        print("  On this evidence timing variability genuinely carries less")
        print("  performer identity than melodic voicing does -- the extra")
        print("  capacity found nothing the 11 summary numbers had missed.")
        print("  Report this plainly; it is the more surprising result.")

    (OUT / "sequence_model.json").write_text(json.dumps(
        {"real": real, "shuffled": shuf, "n_classes": n_classes, "chance": chance,
         "n_windows": nwin, "window": WIN, "stride": STRIDE}, indent=2), encoding="utf-8")
    print("\n  -> results/sequence_model.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
