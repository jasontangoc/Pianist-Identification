"""The kill test, on note-exact MIDI: is a pianist closer to themselves?

The audio version of this could not answer it, because pairwise alignment was not a
stable enough instrument. ASAP's performance MIDI has none of the confounds that
defeated it: notes are exact (Disklavier capture, no transcription), the time axis is
true (no transfer speed), and the performer is labelled.

Distances are computed only within a work, so the music is held constant and the
comparison is between interpretations rather than between pieces. Pairs are then pooled
across works and split into same-performer and different-performer. The project needs
within < between.

Feature space is z-scored per work first. Without that, whichever feature has the
largest raw spread in a given piece dominates the distance, and the result is about
units rather than playing.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pianist_id import features as F

CACHE = ROOT / "data" / "asap" / "midi"
OUT = ROOT / "results"
RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
UA = {"User-Agent": "PianistIdentification/0.1"}
MIN_PER_WORK = 4  # need enough performances of a work for between-pairs to mean anything


def performer_id(path: str) -> str | None:
    stem = path.split("/")[-1].replace(".mid", "")
    m = re.match(r"^([A-Za-z_]+?)\d+M?$", stem)
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
    except Exception as exc:  # noqa: BLE001
        print(f"   ! {rel}: {exc}")
        return None


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("reading ASAP metadata...")
    with urllib.request.urlopen(urllib.request.Request(RAW + "metadata.csv", headers=UA),
                                timeout=90) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", errors="replace"))))

    # works with at least one performer who has two takes, and enough performances
    # for the between-group to mean something
    by_work: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        pid = performer_id(row["midi_performance"])
        if pid:
            by_work[row["folder"]].append((pid, row["midi_performance"]))

    targets = {}
    for work, entries in by_work.items():
        counts = defaultdict(int)
        for pid, _ in entries:
            counts[pid] += 1
        if len(entries) >= MIN_PER_WORK and any(c > 1 for c in counts.values()):
            targets[work] = entries
    print(f"works with a repeat performer and >={MIN_PER_WORK} performances: {len(targets)}")

    feats: dict[str, dict] = {}
    work_members: dict[str, list[tuple[str, str]]] = {}
    for work, entries in sorted(targets.items()):
        kept = []
        for pid, rel in entries:
            data = fetch(rel)
            if data is None:
                continue
            try:
                f = F.extract(F.load_midi(data))
            except Exception as exc:  # noqa: BLE001
                print(f"   ! extract {rel}: {exc}")
                continue
            if "error" in f or f.get("n_notes", 0) < 200:
                continue
            feats[rel] = f
            kept.append((pid, rel))
        if len(kept) >= MIN_PER_WORK and len({p for p, _ in kept}) < len(kept):
            work_members[work] = kept
        print(f"   {work[:52]:<53} {len(kept):>3} performances")

    print(f"\nusable works: {len(work_members)}")

    within, between = [], []
    detail = []
    for work, members in work_members.items():
        mat = np.array([F.vector(feats[rel]) for _, rel in members], dtype=float)
        ok = ~np.any(~np.isfinite(mat), axis=0)
        mat = mat[:, ok]
        if mat.shape[1] < 4 or mat.shape[0] < MIN_PER_WORK:
            continue
        mu, sd = mat.mean(axis=0), mat.std(axis=0)
        sd[sd < 1e-12] = 1.0
        z = (mat - mu) / sd
        for i, j in itertools.combinations(range(len(members)), 2):
            d = float(np.linalg.norm(z[i] - z[j]) / np.sqrt(z.shape[1]))
            same = members[i][0] == members[j][0]
            (within if same else between).append(d)
            detail.append({"work": work, "a": members[i][1], "b": members[j][1],
                           "same_performer": same, "distance": d})

    print()
    print("=" * 78)
    print("RESULT  normalised feature distance, computed within each work")
    print("=" * 78)
    for label, vals in (("within  (same performer)", within), ("between (different)", between)):
        if vals:
            a = np.array(vals)
            print(f"  {label:<26} n={len(a):<5} mean {a.mean():.4f}  "
                  f"median {np.median(a):.4f}  sd {a.std():.4f}")
    if not within or not between:
        print("\n  INCONCLUSIVE: not enough pairs.")
        return 1

    w, b = np.array(within), np.array(between)
    # Probability a random within-pair is closer than a random between-pair: 0.5 is
    # chance, 1.0 is perfect separation. Reported instead of a p-value because
    # performances appear in many pairs, so a significance test would overstate it.
    auc = float(np.mean([np.mean(wi < b) + 0.5 * np.mean(wi == b) for wi in w]))
    effect = (b.mean() - w.mean()) / np.sqrt((b.var() + w.var()) / 2)

    print()
    print(f"  within mean {w.mean():.4f}  vs  between mean {b.mean():.4f}")
    print(f"  separation (Cohen's d)             {effect:+.3f}")
    print(f"  P(within pair closer than between) {auc:.3f}   (0.5 = chance)")
    print()
    if auc >= 0.75 and effect > 0.5:
        print("  PASS. A performer is reliably closer to themselves than to")
        print("  others on these features. The premise holds, so a classifier")
        print("  on them is worth building.")
    elif auc >= 0.6:
        print("  WEAK PASS. The effect is in the right direction but small.")
        print("  These features carry some identity; a classifier on them alone")
        print("  will be limited. Expect modest accuracy and report it honestly.")
    else:
        print("  FAIL. Same-performer pairs are not closer than different-performer")
        print("  pairs. On these features, identification is not supported.")

    (OUT / "within_between_midi.json").write_text(
        json.dumps({"within": within, "between": between, "auc": auc,
                    "cohens_d": effect, "pairs": detail}, indent=1), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.linspace(0, max(b.max(), w.max()), 40)
    ax.hist(b, bins=bins, alpha=.6, label=f"between performers (n={len(b)})", color="C0",
            density=True)
    ax.hist(w, bins=bins, alpha=.75, label=f"same performer (n={len(w)})", color="C3",
            density=True)
    ax.axvline(w.mean(), color="C3", ls="--", lw=1.4)
    ax.axvline(b.mean(), color="C0", ls="--", lw=1.4)
    ax.set_xlabel("normalised feature distance (within-work, z-scored)")
    ax.set_ylabel("density")
    ax.set_title(f"Is a pianist closer to themselves?  AUC {auc:.3f}, d {effect:+.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "within_between_midi.png", dpi=150)
    print(f"\n  -> results/within_between_midi.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
