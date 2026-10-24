"""Resolve candidate pianist x work cells into actual downloadable audio files.

`survey_pianists.py` counts mentions in search text, which is an availability estimate.
This opens each item, reads its file list and emits one row per playable track with a
work attached: the difference between "Horowitz has 23 mentions of Op. 35" and "there
are N distinct Horowitz recordings of the B-flat minor Sonata".

It does not merge apparent duplicates. Reissues of one performance and genuine
re-recordings look alike from metadata, and Rubinstein recording the Nocturnes three
times is exactly the data this project needs. Tracks of the same work by the same
pianist whose durations agree within 6% are flagged `dup_group` instead.

**`dup_group` does not work and the output must not be read as if it does.** The 6%
window came from the transfer-speed measurement, on the assumption that two
digitisations of one performance differ by about that much. They do; the flaw is the
converse, since genuine re-recordings differ by about that much too. Cortot's 1926 and
1933 Ballade No. 1 run 503 s and 521 s, 3.6% apart, so the grouping merges two
performances seven years apart. It also misreads multi-movement works, where one opus
number covers several pieces, and 78rpm sides, where one performance appears both as
4-minute fragments and as an 8-minute whole.

The reliable test is the audio one: two files of one performance DTW-align to a
near-diagonal path and two performances do not. Final de-duplication belongs there.
This script's job is to narrow a few thousand candidates to something worth
downloading, and it does not guess dates.

Results cache per item, so the expensive pass happens once.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
CACHE = OUT / "archive_meta"

PIANISTS = [
    # Post-1950 roster. The pre-1950 pianists were dropped after the transcription
    # spike: two digitisations of one 1926 shellac performance transcribe to note sets
    # matching at F1 0.12, so per-note features cannot be built on that material.
    "Arthur Rubinstein", "Vladimir Horowitz", "Guiomar Novaes",
    "Arturo Benedetti Michelangeli", "Sviatoslav Richter", "Claudio Arrau",
    "Alexander Brailowsky", "Byron Janis", "Shura Cherkassky",
    "Samson Francois", "Witold Malcuzynski", "Agustin Anievas",
    "Andrei Gavrilov", "Jorge Bolet", "Earl Wild", "Wilhelm Backhaus",
    "Julian von Karolyi", "Martin Kasik",
]
MAX_ITEMS = 60

AUDIO_FORMATS = {"Flac", "24bit Flac", "VBR MP3", "Ogg Vorbis", "WAVE", "MPEG-4 Audio"}
OPUS = re.compile(r"\bop(?:us|\.)?\s*(\d{1,3})\s*(?:[,/]?\s*(?:no|nr|n)\.?\s*(\d{1,2}))?", re.I)

GENRE = {
    6: "mazurka", 7: "mazurka", 9: "nocturne", 10: "etude", 15: "nocturne",
    17: "mazurka", 18: "waltz", 20: "scherzo", 23: "ballade", 24: "mazurka",
    25: "etude", 27: "nocturne", 28: "prelude", 29: "impromptu", 30: "mazurka",
    31: "scherzo", 32: "nocturne", 33: "mazurka", 34: "waltz", 35: "sonata",
    36: "impromptu", 37: "nocturne", 38: "ballade", 39: "scherzo", 40: "polonaise",
    41: "mazurka", 42: "waltz", 44: "polonaise", 47: "ballade", 48: "nocturne",
    49: "fantaisie", 50: "mazurka", 51: "impromptu", 52: "ballade", 53: "polonaise",
    55: "nocturne", 56: "mazurka", 57: "berceuse", 58: "sonata", 59: "mazurka",
    60: "barcarolle", 61: "polonaise-fantaisie", 62: "nocturne", 63: "mazurka",
    64: "waltz", 66: "fantaisie-impromptu", 68: "mazurka", 69: "waltz", 70: "waltz",
}
NOT_SOLO = {11, 21, 22, 3, 65, 2, 13, 14, 8, 16}

UA = {"User-Agent": "PianistIdentification/0.1"}
DUP_TOLERANCE = 0.06  # the measured transfer-speed spread


def get_json(url: str, timeout: int = 60) -> dict | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def search(pianist: str) -> list[str]:
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
        {
            "q": f'chopin AND ("{pianist}") AND mediatype:audio',
            "fl[]": "identifier",
            "rows": str(MAX_ITEMS),
            "sort[]": "downloads desc",
            "output": "json",
        },
        doseq=True,
    )
    data = get_json(url)
    if not data:
        return []
    return [d["identifier"] for d in data["response"]["docs"]]


def metadata(identifier: str) -> dict | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", identifier)[:120]
    path = CACHE / f"{safe}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    data = get_json(f"https://archive.org/metadata/{identifier}")
    if data is not None:
        path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(0.15)
    return data


def seconds(raw) -> float:
    if raw is None:
        return 0.0
    text = str(raw)
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        return sum(v * 60**i for i, v in enumerate(reversed(parts)))
    try:
        return float(text)
    except ValueError:
        return 0.0


def work_of(*texts: str) -> tuple[str, str] | None:
    for text in texts:
        if not text:
            continue
        for opus, number in OPUS.findall(text):
            op = int(opus)
            if op in NOT_SOLO or op not in GENRE:
                continue
            key = f"op.{op}" + (f"/{int(number)}" if number else "")
            return key, GENRE[op]
    return None


def surname(pianist: str) -> str:
    return pianist.split()[-1].lower()


# Used only to detect that an item holds several performers, never as a roster. The
# question is not whether this pianist is named here but whether the item holds more
# than one pianist: requiring the name in every file dropped single-performer albums
# where only some filenames carry it, taking Horowitz down to 8 tracks.
KNOWN_PIANISTS = (
    "argerich", "pollini", "anda", "ashkenazy", "zimerman", "gilels", "kempff",
    "serkin", "curzon", "katchen", "lipatti", "solomon", "barenboim", "ohlsson",
    "perahia", "vasary", "magaloff", "frankl", "rubinstein", "horowitz",
    "cortot", "hofmann", "rachmaninoff", "pachmann", "friedman", "rosenthal",
    "lhevinne", "paderewski", "novaes", "michelangeli", "richter", "arrau",
    "brailowsky", "janis", "cherkassky", "francois", "malcuzynski", "anievas",
    "gavrilov", "bolet", "backhaus", "karolyi", "kasik", "moiseiwitsch",
    "hambourg", "uninsky", "weissenberg", "entremont", "watts", "fou",
)


OTHER_COMPOSERS = (
    "tchaikovsky", "brahms", "liszt", "schumann", "beethoven", "mozart",
    "scriabin", "prokofiev", "debussy", "ravel", "schubert", "mendelssohn",
    "bach", "haydn", "grieg", "saint-saens", "franck",
)


def blob_of(md: dict) -> str:
    return " ".join(
        str(md.get(k, "")) for k in ("creator", "title", "description", "artist")
    ).lower()


def attributable(
    pianist: str, md: dict, file_text: str, n_tracks: int, multi_performer: bool
) -> bool:
    """Is this track really by this pianist?

    Searching `chopin AND ("Rubinstein")` returns anthologies that merely
    mention him. The first pass credited every track of "The Great Chopin
    Interpreters" to all six pianists named on it, and every roll on a Duo-Art
    compilation to five. Those rows are not merely noisy, they are wrong in the
    one field the whole dataset is keyed on.

    So: a single-performer item may attribute by item metadata, but a
    multi-track item that names several pianists must name this one at file
    level or the track is dropped.
    """
    name = surname(pianist)
    blob = " ".join(
        str(md.get(k, "")) for k in ("creator", "title", "description", "artist")
    ).lower()
    if "chopin" not in blob:
        return False  # keeps film soundtracks and unrelated uploads out

    text = file_text.lower()

    # Opus numbers are not unique across composers: Tchaikovsky's First Concerto is
    # Op. 23 and was being collected as Chopin's Ballade.
    if any(c in text for c in OTHER_COMPOSERS) and "chopin" not in text:
        return False

    # Comparison compilations name several pianists but only one in the item title;
    # one such item filed Argerich, Pollini and Geza Anda tracks under Michelangeli.
    # Per-file attribution is applied only when the item looks like a compilation,
    # since otherwise a single-performer album loses every track whose filename does
    # not repeat the name.
    if multi_performer:
        return name in text
    return name in blob or name in text


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = []

    for pianist in PIANISTS:
        ids = search(pianist)
        kept = 0
        for identifier in ids:
            meta = metadata(identifier)
            if not meta or not meta.get("metadata"):
                continue
            md = meta["metadata"]
            item_title = str(md.get("title", ""))
            item_year = str(md.get("year", "") or md.get("date", ""))[:4]
            audio = [f for f in meta.get("files", []) if f.get("format") in AUDIO_FORMATS]
            # Does any file in this item name the pianist? If so, per-file
            # attribution is available and the item title must not be trusted.
            names_here = set()
            for f in audio:
                ft = f"{f.get('title','')} {f['name']}".lower()
                names_here |= {k for k in KNOWN_PIANISTS if k in ft}
            names_here |= {k for k in KNOWN_PIANISTS if k in blob_of(md)}
            multi_performer = len(names_here) > 1

            for f in audio:
                dur = seconds(f.get("length"))
                if not 45 <= dur <= 2400:
                    # Below 45 s is a fragment. Above 40 min is corrupt
                    # metadata: one Cortot item reports 176,334 s, which is
                    # 49 hours, for a nine-minute Ballade.
                    continue
                file_text = f"{f.get('title','')} {f['name']}"
                if not attributable(
                    pianist, md, file_text, len(audio), multi_performer
                ):
                    continue

                # The item title identifies the work only when the item holds
                # one work. On "Complete Recordings CD 6" it tagged all 45
                # tracks with a single opus lifted from the title, which is not
                # an approximation -- it is 44 wrong labels.
                hit = work_of(str(f.get("title", "")), f["name"])
                if hit is None and len(audio) <= 3:
                    hit = work_of(item_title)
                if hit is None:
                    continue
                work, genre = hit
                rows.append(
                    {
                        "pianist": pianist,
                        "work": work,
                        "genre": genre,
                        "year": item_year if item_year.isdigit() else "",
                        "duration_s": round(dur, 1),
                        "format": f["format"],
                        "identifier": identifier,
                        "filename": f["name"],
                        "item_title": item_title[:90],
                    }
                )
                kept += 1
        print(f"  {pianist:<32} {len(ids):>3} items -> {kept:>4} audio tracks with a work")

    # --- flag probable duplicates, do not merge --------------------------
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["pianist"], row["work"])].append(row)

    for (_, _), bucket in groups.items():
        bucket.sort(key=lambda r: r["duration_s"])
        gid, anchor = 0, None
        for row in bucket:
            if anchor is None or abs(row["duration_s"] - anchor) / max(anchor, 1) > DUP_TOLERANCE:
                gid += 1
                anchor = row["duration_s"]
            row["dup_group"] = gid

    fields = [
        "pianist", "work", "genre", "year", "duration_s", "dup_group",
        "format", "identifier", "filename", "item_title",
    ]
    dest = OUT / "candidate_recordings.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # --- summary ---------------------------------------------------------
    distinct = defaultdict(set)
    for row in rows:
        distinct[row["pianist"]].add((row["work"], row["dup_group"]))

    print()
    print("=" * 84)
    print("VERIFIED: distinct (work, performance) pairs per pianist")
    print("=" * 84)
    print(f"{'pianist':<32}{'tracks':>8}{'distinct':>10}{'works':>8}  top genres")
    print("-" * 84)
    for pianist, pairs in sorted(distinct.items(), key=lambda kv: -len(kv[1])):
        mine = [r for r in rows if r["pianist"] == pianist]
        works = {r["work"] for r in mine}
        genres = sorted({r["genre"] for r in mine})
        print(
            f"{pianist:<32}{len(mine):>8}{len(pairs):>10}{len(works):>8}  "
            f"{', '.join(genres[:5])}"
        )

    print(f"\n  -> {dest.relative_to(ROOT)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
