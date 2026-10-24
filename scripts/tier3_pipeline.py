"""Tier 3: run the validated pipeline on real archival audio.

Tier 1 established, on note-exact MIDI, that these features identify a performer across
pieces at 9.02% against 1.67% chance. This tier asks what survives when the notes have
to be recovered from a recording, which is the claim the project actually makes, since
it is about concert pianists rather than competition entrants. The gap between the
tiers is the result, not this tier's accuracy on its own.

Four design constraints, all inherited rather than chosen here.

Post-1950 only, because shellac loses 20-40% of notes, differently in each transfer
(F1 0.12 between two digitisations of one performance). Enforced only where a date is
known: 53% of candidate rows have none, and the year field is an upload date as often
as a recording date (three Michelangeli rows read 2026; he died in 1995). Screening on
audio bandwidth instead failed, since shellac surface noise is broadband and
high-frequency presence ranks a 1926 transfer as wider-band than a 1963 tape, so era
artefacts are controlled after transcription rather than before.

Works with three or more pianists, because per-work normalisation needs peers.

One work per track, because a "complete op. 25" upload is twelve etudes in one file and
is not the same music as a single etude under the same opus. Test 5 of
validate_features.py measures the damage: pooling four movements of one sonata leaves
ioi_ratio_p90_p10 88% out, while pooling six performances of the same piece moves
nothing. Since only some pianists have whole-set uploads, granularity would leak
identity through the timing features the project is about.

Ten minutes maximum, because transcription runs at ~1.1x realtime on CPU, and 600 s
also happens to separate the longest single works here from the whole-set uploads.

Resumable: audio, transcriptions and features all cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import features as F

CANDIDATES = ROOT / "results" / "candidate_recordings.csv"
AUDIO = ROOT / "data" / "audio" / "tier2"
NOTES = ROOT / "results" / "tier3_notes"
OUT = ROOT / "results"
UA = {"User-Agent": "PianistIdentification/0.1"}

MAX_DURATION = 600     # seconds; the run is resumable, so this spans two sessions
MIN_DURATION = 60      # below this there is too little to characterise
MIN_PIANISTS_PER_WORK = 3
MIN_RECORDINGS_PER_PIANIST = 4
CAP_PER_PIANIST = 14
MODEL_SR = 16000

# A recording counts as the same music as its work-mates only if its length is within
# this factor of the median for that work. Two performances of one piece rarely differ
# by more than ~40% in tempo, so 1.8x keeps every genuine reading and excludes a
# whole-set upload.
DURATION_TOLERANCE = 1.8
EARLIEST_YEAR = 1950   # applied only where the year is known and plausible


def key_of(row: dict) -> str:
    return hashlib.sha1(f"{row['identifier']}/{row['filename']}".encode()).hexdigest()[:16]


def dated_before(row: dict, year: int) -> bool:
    """Is this row known to predate `year`? Unknown dates return False.

    A year counts as a recording date only if it is plausible as one: values above 1999
    are upload dates for pianists who had stopped recording, or died, long before.
    """
    y = row.get("year", "").strip()
    return y.isdigit() and 1900 <= int(y) < year


def select() -> list[dict]:
    rows = list(csv.DictReader(CANDIDATES.open(encoding="utf-8")))
    # collapse format duplicates: same pianist, work and near-identical length
    seen: dict[tuple, dict] = {}
    for r in rows:
        if dated_before(r, EARLIEST_YEAR):
            continue
        d = float(r["duration_s"])
        if not (MIN_DURATION <= d <= MAX_DURATION):
            continue
        k = (r["pianist"], r["work"], round(d / 3))
        seen.setdefault(k, r)
    uniq = list(seen.values())

    # Drop tracks that are not the same music as their work-mates. The label is at
    # opus level, so a complete set and a single piece from it collide, and whole-set
    # uploads cluster in particular pianists, which would let the model read identity
    # off the track layout.
    med_dur: dict[str, float] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in uniq:
        grouped[r["work"]].append(float(r["duration_s"]))
    for w, ds in grouped.items():
        med_dur[w] = float(np.median(ds))
    kept = []
    for r in uniq:
        ratio = float(r["duration_s"]) / med_dur[r["work"]]
        if 1.0 / DURATION_TOLERANCE <= ratio <= DURATION_TOLERANCE:
            kept.append(r)
    n_mixed = len(uniq) - len(kept)
    uniq = kept

    by_work = defaultdict(set)
    for r in uniq:
        by_work[r["work"]].add(r["pianist"])
    good_works = {w for w, p in by_work.items() if len(p) >= MIN_PIANISTS_PER_WORK}
    pool = [r for r in uniq if r["work"] in good_works]

    counts = Counter(r["pianist"] for r in pool)
    keep = {p for p, c in counts.items() if c >= MIN_RECORDINGS_PER_PIANIST}
    pool = [r for r in pool if r["pianist"] in keep]

    # cap per pianist, preferring shorter recordings (cheaper) and works held
    # by more pianists (better normalisation)
    pool.sort(key=lambda r: (-len(by_work[r["work"]]), float(r["duration_s"])))
    per: Counter = Counter()
    chosen = []
    for r in pool:
        if per[r["pianist"]] >= CAP_PER_PIANIST:
            continue
        per[r["pianist"]] += 1
        chosen.append(r)
    return chosen, {"n_mixed": n_mixed}


def download(row: dict) -> Path | None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    dest = AUDIO / f"{key_of(row)}{Path(row['filename']).suffix or '.audio'}"
    if dest.exists():
        return dest
    safe = urllib.parse.quote(f"{row['identifier']}/{row['filename']}", safe="/")
    url = f"https://archive.org/download/{safe}"
    for attempt in range(3):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=300) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code in (500, 502, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            return None
    return None


def transcribe(row: dict, path: Path, transcriptor) -> list[dict] | None:
    NOTES.mkdir(parents=True, exist_ok=True)
    dest = NOTES / f"{key_of(row)}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    import librosa

    try:
        y, _ = librosa.load(path, sr=MODEL_SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=35.0)
        ev = transcriptor.transcribe(y, None)["est_note_events"]
    except Exception as exc:  # noqa: BLE001
        print(f"      ! transcribe failed: {exc}")
        return None
    notes = [{"midi_note": int(n["midi_note"]), "onset_time": float(n["onset_time"]),
              "offset_time": float(n["offset_time"]), "velocity": int(n["velocity"])}
             for n in ev]
    dest.write_text(json.dumps(notes))
    return notes


def main() -> int:
    OUT.mkdir(exist_ok=True)
    chosen, chosen_meta = select()
    per = Counter(r["pianist"] for r in chosen)
    total = sum(float(r["duration_s"]) for r in chosen)
    print(f"selected {len(chosen)} recordings, {len(per)} pianists, "
          f"{len({r['work'] for r in chosen})} works")
    print(f"  (duration-consistency filter removed {chosen_meta['n_mixed']} "
          f"mixed-granularity tracks at {DURATION_TOLERANCE}x)")
    print(f"total audio {total/60:.0f} min -> transcription ~{total/60*1.1:.0f} min CPU")
    for p, c in per.most_common():
        print(f"   {p:<32}{c:>3}")

    from piano_transcription_inference import PianoTranscription
    print("\nloading transcription model...")
    transcriptor = PianoTranscription(device="cpu")

    records = []
    t0 = time.time()
    for n, row in enumerate(chosen, 1):
        cached = (NOTES / f"{key_of(row)}.json").exists()
        path = download(row)
        if path is None:
            print(f"   [{n}/{len(chosen)}] download failed: {row['pianist']} {row['work']}")
            continue
        notes = transcribe(row, path, transcriptor)
        if not notes:
            continue
        records.append({"pianist": row["pianist"], "work": row["work"],
                        "key": key_of(row), "n_notes": len(notes),
                        # items nest perfectly inside pianists, so cross-validation
                        # holds whole items out rather than random tracks
                        "identifier": row["identifier"],
                        "duration_s": float(row["duration_s"]),
                        "year": row.get("year", "").strip()})
        tag = "cached" if cached else f"{(time.time()-t0)/60:.0f}m elapsed"
        print(f"   [{n}/{len(chosen)}] {row['pianist'][:22]:<23}{row['work']:<9}"
              f"{len(notes):>6} notes  ({tag})")

    (OUT / "tier3_index.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"\ntranscribed {len(records)} recordings in {(time.time()-t0)/60:.0f} min")

    # --- features --------------------------------------------------------
    feats, meta = [], []
    for rec in records:
        notes = json.loads((NOTES / f"{rec['key']}.json").read_text())
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
        meta.append((rec["pianist"], rec["work"]))

    X = np.array(feats, dtype=float)
    y = np.array([m[0] for m in meta])
    works = np.array([m[1] for m in meta])
    print(f"feature matrix {X.shape}, {len(set(y))} pianists, {len(set(works))} works")

    # same leave-one-out per-work normalisation as the MIDI tier
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
    Xn, y, works = Xn[keep], y[keep], works[keep]
    pc = Counter(y)
    keep2 = np.array([pc[v] >= 3 for v in y])
    Xn, y, works = Xn[keep2], y[keep2], works[keep2]
    print(f"after Tier-1-identical restriction: {len(y)} recordings, "
          f"{len(set(y))} pianists, {len(set(works))} works")

    if len(set(y)) < 3:
        print("\nToo few pianists survive to classify. Report the roster, not a number.")
        return 1

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder

    def run(Xu, yy, seeds=10):
        out = []
        for s in range(seeds):
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=s)
            a = []
            for tr, te in skf.split(Xu, yy):
                lr = LogisticRegression(max_iter=4000).fit(Xu[tr], yy[tr])
                a.append(np.mean(lr.predict(Xu[te]) == yy[te]))
            out.append(np.mean(a))
        return float(np.mean(out)), float(np.std(out))

    chance = 1.0 / len(set(y))
    real = run(Xn, y)
    rng = np.random.default_rng(0)
    shuf = run(Xn, rng.permutation(y))
    P = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
        works.reshape(-1, 1))
    piece = run(P, y)
    plus = run(np.hstack([P, Xn]), y)

    print()
    print("=" * 72)
    print(f"TIER 2 — transcribed archival audio, {len(set(y))} pianists")
    print("=" * 72)
    print(f"  chance                {100*chance:>7.2f}%")
    print(f"  shuffled labels       {100*shuf[0]:>7.2f}% +/- {100*shuf[1]:.2f}")
    print(f"  piece identity only   {100*piece[0]:>7.2f}% +/- {100*piece[1]:.2f}")
    print(f"  features              {100*real[0]:>7.2f}% +/- {100*real[1]:.2f}")
    print(f"  piece + features      {100*plus[0]:>7.2f}% +/- {100*plus[1]:.2f}")
    print(f"  lift over chance      {real[0]/chance:>7.1f}x")

    tier1 = json.loads((OUT / "baseline_classifier.json").read_text())
    t1 = tier1["real"]["logreg"]
    t1c = tier1["real"]["chance"]
    print()
    print("  Tier 1 (exact MIDI):  %.2f%% on %d classes (%.1fx chance)"
          % (100 * t1, tier1["real"]["n_classes"], t1 / t1c))
    print("  Tier 2 (transcribed): %.2f%% on %d classes (%.1fx chance)"
          % (100 * real[0], len(set(y)), real[0] / chance))
    print("  -> lift ratio Tier2/Tier1: %.2f" % ((real[0] / chance) / (t1 / t1c)))
    print("     (absolute accuracies are not comparable -- different class counts;")
    print("      the lift over chance is.)")

    (OUT / "tier3_result.json").write_text(json.dumps(
        {"chance": chance, "features": real, "shuffled": shuf, "piece_only": piece,
         "piece_plus": plus, "n_pianists": len(set(y)), "n_recordings": len(y),
         "n_works": len(set(works))}, indent=2), encoding="utf-8")
    print("\n  -> results/tier3_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
