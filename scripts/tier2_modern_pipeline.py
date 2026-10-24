"""Tier 2 - modern audio, paired against note-exact MIDI.

The archival tier asks whether famous pianists can be told apart from old recordings,
which is the interesting question and the weak measurement: 73 recordings, 10 pianists,
and each pianist's tracks drawn from a handful of LP transfers that nest inside their
identity.

This tier asks a narrower question properly. ASAP records for each performance both the
note-exact MIDI and the MAESTRO audio it came from, with offsets into that audio, so the
same performance can be measured twice and the difference is transcription and nothing
else. A competition round is one audio file holding several works, so the session plays
the role the Archive item plays in the archival tier and is held out whole; every
performer here has 3 to 9 sessions, so none is dropped. One of 510 (performer, work)
pairs spans two sessions and there are no exact duplicates, so no performance sits on
both sides of a split. Nothing is truncated to save CPU: tested on MIDI first, balanced
accuracy falls 6% at five minutes, 28% at four and 51% at two.

The limit worth carrying: MAESTRO is a Disklavier recorded in good conditions, with
ground truth from that piano's own sensors, so this measures the cost of transcription
on easy audio. The archival tier supplies the pessimistic bound.

Resumable: every performance caches its notes under a key derived from its ASAP path,
and a session whose spans are all cached is never downloaded.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id.remote_zip import RemoteZip

OUT = ROOT / "results"
NOTES = OUT / "tier2_modern_notes"
CACHE = OUT / "asap_maestro_index.json"
CACHE_WIDE = OUT / "asap_maestro_index_wide.json"
ASAP_RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
MAESTRO_JSON = ("https://storage.googleapis.com/magentadata/datasets/maestro/"
                "v3.0.0/maestro-v3.0.0.json")
MAESTRO_ZIP = ("https://storage.googleapis.com/magentadata/datasets/maestro/"
               "v3.0.0/maestro-v3.0.0.zip")
UA = {"User-Agent": "PianistIdentification/0.1"}

MIN_SESSIONS_PER_PERFORMER = 2
MIN_DURATION = 60.0
MIN_PERFORMANCES_PER_WORK = 3
MIN_RECORDINGS_PER_PERFORMER = 3
MODEL_SR = 16000

# The work rule counts performances of the work, not performers of it, because what
# needs three is the per-work leave-one-out normalisation, which knows nothing about who
# played. Both phrasings select the same 88 performances here, since of 510
# (performer, work) pairs exactly one spans two sessions.
#
# Two pools, differing on normalisation rather than on threshold taste:
#
#   strict (3)  88 performances, 23 performers, 346 min. Leave-one-out z-scoring has at
#               least two others to take a standard deviation from.
#   wide   (2)  270 performances, 64 performers, 1238 min, but 35 works have exactly
#               two performances, where that standard deviation is taken over one value
#               and is zero. Usable only with scale="pooled" in loo_normalise, which
#               separates centring (needs two) from scaling (pooled across works).
#
# An intermediate pool of 158 looks available by relaxing the rule in the loop and
# keeping three in the normaliser. It is incoherent: it admits performers on a lax rule
# then masks 110 of their performances on a strict one, leaving works represented by one
# or two rows whose normalisation came from rows no longer there. Not reported.


def key_of(midi_performance: str) -> str:
    return hashlib.sha1(midi_performance.encode()).hexdigest()[:16]


def _fetch(url: str) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=120).read()


def performer_of(midi_performance: str) -> str | None:
    stem = midi_performance.split("/")[-1].replace(".mid", "")
    m = re.match(r"^([A-Za-z_]+?)\d+M?$", stem)
    return m.group(1) if m else None


def segments() -> list[dict]:
    """Every ASAP performance that has MAESTRO audio, with its span resolved.

    ASAP fills `end` for only 162 of 1067 rows. Where it is blank the segment runs to
    the next performance in the same file, or to the end of the file, which is why the
    rows have to be sorted first.
    """
    rows = list(csv.DictReader(io.StringIO(
        _fetch(ASAP_RAW + "metadata.csv").decode("utf-8", "replace"))))
    mj = json.loads(_fetch(MAESTRO_JSON))
    af, du = mj["audio_filename"], mj["duration"]
    total = ({af[k]: du[k] for k in af} if isinstance(af, dict)
             else dict(zip(af, du)))

    by_file: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["maestro_audio_performance"].strip():
            by_file[r["maestro_audio_performance"].replace("{maestro}/", "")].append(r)

    out = []
    for member, rs in by_file.items():
        rs.sort(key=lambda r: float(r["start"] or 0))
        for k, r in enumerate(rs):
            start = float(r["start"] or 0)
            if r["end"].strip():
                end = float(r["end"])
            elif k + 1 < len(rs):
                end = float(rs[k + 1]["start"] or 0)
            else:
                end = total.get(member, 0.0)
            performer = performer_of(r["midi_performance"])
            if not performer or end - start < MIN_DURATION:
                continue
            out.append({
                "performer": performer,
                "work": r["folder"],
                "session": member,
                "midi": r["midi_performance"],
                "start": start,
                "end": end,
                "duration_s": end - start,
                "key": key_of(r["midi_performance"]),
            })
    return out


def select(segs: list[dict], min_per_work: int = MIN_PERFORMANCES_PER_WORK) -> list[dict]:
    """Apply the thresholds fixed in advance, to a fixed point.

    The rules interact: dropping a thin work can leave a performer below the recording
    minimum, which can push another work below its own. One pass each leaves the pool
    violating the constraints it was meant to enforce, so iterate and assert.
    """
    pool = list(segs)
    while True:
        before = len(pool)

        wc = Counter(s["work"] for s in pool)
        pool = [s for s in pool if wc[s["work"]] >= min_per_work]

        pc = Counter(s["performer"] for s in pool)
        pool = [s for s in pool
                if pc[s["performer"]] >= MIN_RECORDINGS_PER_PERFORMER]

        sess = defaultdict(set)
        for s in pool:
            sess[s["performer"]].add(s["session"])
        pool = [s for s in pool
                if len(sess[s["performer"]]) >= MIN_SESSIONS_PER_PERFORMER]

        if len(pool) == before:
            break

    check_invariants(pool, min_per_work)
    pool.sort(key=lambda s: (s["session"], s["start"]))
    return pool


def check_invariants(pool: list[dict],
                    min_per_work: int = MIN_PERFORMANCES_PER_WORK) -> None:
    """Fail loudly rather than quietly evaluating a pool that breaks the design."""
    if not pool:
        raise SystemExit("selection is empty; thresholds are too strict")
    sess = defaultdict(set)
    for s in pool:
        sess[s["performer"]].add(s["session"])
    wc = Counter(s["work"] for s in pool)
    pc = Counter(s["performer"] for s in pool)
    assert min(wc.values()) >= min_per_work
    assert min(pc.values()) >= MIN_RECORDINGS_PER_PERFORMER
    assert min(len(v) for v in sess.values()) >= MIN_SESSIONS_PER_PERFORMER
    assert len({s["key"] for s in pool}) == len(pool), "cache keys collide"


def transcribe_session(rz, member, rows, transcriptor, librosa):
    """Transcribe every uncached span of one session in a single pass."""
    todo = [r for r in rows if not (NOTES / f"{r['key']}.json").exists()]
    if not todo:
        return 0
    spans = [(r["start"], r["end"]) for r in todo]
    made = 0
    for k, samples, sr in rz.read_spans(member, spans):
        row = todo[k]
        try:
            y = librosa.resample(samples, orig_sr=sr, target_sr=MODEL_SR)
            # No silence trim, unlike the archival pipeline: ASAP gives exact
            # offsets, and trimming would shift the audio against the ground-truth MIDI
            # by a varying amount. The features are differences between onsets and
            # would not notice, but the note-F1 check against ground truth would, and
            # that check is the only thing that can spot a broken transcription.
            ev = transcriptor.transcribe(y, None)["est_note_events"]
        except Exception as exc:  # noqa: BLE001
            print(f"      ! {row['work']} {row['performer']}: {exc}")
            continue
        notes = [{"midi_note": int(n["midi_note"]),
                  "onset_time": float(n["onset_time"]),
                  "offset_time": float(n["offset_time"]),
                  "velocity": int(n["velocity"])} for n in ev]
        (NOTES / f"{row['key']}.json").write_text(json.dumps(notes))
        made += 1
        print(f"      {row['performer'][:16]:<17}{row['work'][:38]:<39}"
              f"{len(notes):>6} notes")
    return made


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    wide = "--wide" in argv
    min_per_work = 2 if wide else MIN_PERFORMANCES_PER_WORK
    index = CACHE_WIDE if wide else CACHE

    NOTES.mkdir(parents=True, exist_ok=True)
    chosen = select(segments(), min_per_work)
    index.write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    print(f"pool: {'wide' if wide else 'strict'} "
          f"(>= {min_per_work} performances per work) -> {index.name}")

    per = Counter(s["performer"] for s in chosen)
    audio_min = sum(s["duration_s"] for s in chosen) / 60
    todo = [s for s in chosen if not (NOTES / f"{s['key']}.json").exists()]
    todo_min = sum(s["duration_s"] for s in todo) / 60
    print(f"selected {len(chosen)} performances, {len(per)} performers, "
          f"{len({s['work'] for s in chosen})} works, "
          f"{len({s['session'] for s in chosen})} sessions")
    print(f"audio {audio_min:.0f} min total; {len(todo)} to transcribe "
          f"({todo_min:.0f} min, ~{todo_min*1.5/60:.1f} h)")
    for p, n in per.most_common():
        ns = len({s['session'] for s in chosen if s['performer'] == p})
        print(f"   {p:<20}{n:>3} performances  {ns:>2} sessions")
    if not todo:
        print("\nnothing to transcribe; cache complete")
        return 0

    import librosa
    from piano_transcription_inference import PianoTranscription
    print("\nloading transcription model...")
    transcriptor = PianoTranscription(device="cpu")
    rz = RemoteZip(MAESTRO_ZIP)
    members = {n.split("/", 1)[1]: n for n in rz.namelist() if n.endswith(".wav")}

    by_session = defaultdict(list)
    for s in chosen:
        by_session[s["session"]].append(s)
    pending = [m for m, rs in by_session.items()
               if any(not (NOTES / f"{r['key']}.json").exists() for r in rs)]

    t0, done = time.time(), 0
    for n, sess in enumerate(pending, 1):
        member = members.get(sess)
        if member is None:
            print(f"   [{n}/{len(pending)}] not in archive: {sess[:60]}")
            continue
        print(f"   [{n}/{len(pending)}] {sess.split('/')[-1][:56]}")
        done += transcribe_session(rz, member, sorted(
            by_session[sess], key=lambda r: r["start"]), transcriptor, librosa)
        el = (time.time() - t0) / 60
        print(f"      {done} done, {el:.0f} min elapsed, "
              f"net {rz.stats['megabytes']/1000:.1f} GB")

    print(f"\ntranscribed {done} performances in {(time.time()-t0)/60:.0f} min")
    print(f"network: {rz.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
