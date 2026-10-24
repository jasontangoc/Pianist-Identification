"""Which pianists have enough free Chopin recordings for a pianist-centric dataset?

A recording-level split needs several recordings per pianist, not several pianists per
recording. The Ballade No. 1 roster gives roughly one recording each, so the split
cannot be run on most of it. This surveys the Internet Archive per pianist and builds a
pianist x work matrix so the dataset can be chosen on evidence.

Work identity is keyed on opus number, the only field written consistently across
catalogues, languages and upload conventions: titles say "Ballade", "Ballade Nr. 1" and
"g-moll Ballade", but nearly all of them say op. 23 somewhere.

Three caveats for reading the output. Recital LPs and complete-set items hold many
works under one title, so works are undercounted for pianists packaged that way. The
Archive's index is not a discography, so absence means "not found free" rather than
"never recorded". And distinct opus numbers are not distinct recordings, since one
pianist may appear twice with one opus from two uploads of one performance.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

# The Ballade No. 1 roster, plus pianists with large free Chopin catalogues. Weighted
# towards the acoustic and early-electric era, where recordings are public domain or
# close to it, and which the era control needs.
PIANISTS = [
    "Alfred Cortot", "Arthur Rubinstein", "Vladimir Horowitz", "Josef Hofmann",
    "Benno Moiseiwitsch", "Samson Francois", "Byron Janis", "Emil Gilels",
    "Sviatoslav Richter", "Arturo Benedetti Michelangeli", "Andrei Gavrilov",
    "Agustin Anievas", "Mark Hambourg", "Julian von Karolyi", "Martin Kasik",
    "Ignacy Jan Paderewski",
    # not on the Ballade roster, but large free catalogues:
    "Ignaz Friedman", "Moriz Rosenthal", "Vladimir de Pachmann", "Raoul Koczalski",
    "Sergei Rachmaninoff", "Josef Lhevinne", "Guiomar Novaes", "Witold Malcuzynski",
    "Alexander Brailowsky", "Wilhelm Backhaus", "Claudio Arrau", "Dinu Lipatti",
    "Shura Cherkassky", "Jorge Bolet", "Earl Wild", "Solomon Cutner",
    "Percy Grainger", "Ania Dorfmann", "Simon Barere", "Egon Petri",
]

OPUS = re.compile(r"\bop(?:us|\.)?\s*(\d{1,3})\s*(?:[,/]?\s*(?:no|nr|n)\.?\s*(\d{1,2}))?", re.I)

# Opus -> genre, for readability.
GENRE = {
    6: "mazurka", 7: "mazurka", 9: "nocturne", 10: "etude", 15: "nocturne",
    17: "mazurka", 18: "waltz", 20: "scherzo", 21: "concerto", 22: "andante sp.",
    23: "ballade", 24: "mazurka", 25: "etude", 27: "nocturne", 28: "prelude",
    29: "impromptu", 30: "mazurka", 31: "scherzo", 32: "nocturne", 33: "mazurka",
    34: "waltz", 35: "sonata", 36: "impromptu", 37: "nocturne", 38: "ballade",
    39: "scherzo", 40: "polonaise", 41: "mazurka", 42: "waltz", 44: "polonaise",
    47: "ballade", 48: "nocturne", 49: "fantaisie", 50: "mazurka", 51: "impromptu",
    52: "ballade", 53: "polonaise", 55: "nocturne", 56: "mazurka", 57: "berceuse",
    58: "sonata", 59: "mazurka", 60: "barcarolle", 61: "polonaise-f.",
    62: "nocturne", 63: "mazurka", 64: "waltz", 66: "fant-impromptu",
    68: "mazurka", 69: "waltz", 70: "waltz",
}
# Opus numbers that are not solo piano, or too short/atypical to be useful.
EXCLUDE = {11, 21, 22, 3, 65, 2, 13, 14}


def fetch(pianist: str) -> list[str]:
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
        {
            "q": f'chopin AND ("{pianist}") AND mediatype:audio',
            "fl[]": ["title", "description"],
            "rows": "400",
            "output": "json",
        },
        doseq=True,
    )
    req = urllib.request.Request(url, headers={"User-Agent": "PianistIdentification/0.1"})
    try:
        docs = json.load(urllib.request.urlopen(req, timeout=60))["response"]["docs"]
    except Exception as exc:  # noqa: BLE001 - survey script, keep going
        print(f"  ! {pianist}: {exc}", file=sys.stderr)
        return []
    return [f"{d.get('title','')} {str(d.get('description',''))[:400]}" for d in docs]


def works_in(texts: list[str]) -> collections.Counter:
    found: collections.Counter = collections.Counter()
    for text in texts:
        for opus, number in OPUS.findall(text):
            op = int(opus)
            if op in EXCLUDE or op not in GENRE:
                continue
            key = f"op.{op}" + (f"/{int(number)}" if number else "")
            found[key] += 1
    return found


def main() -> int:
    OUT.mkdir(exist_ok=True)
    survey: dict[str, dict] = {}

    for pianist in PIANISTS:
        texts = fetch(pianist)
        works = works_in(texts)
        survey[pianist] = {"items": len(texts), "works": dict(works)}
        print(f"  {pianist:<32} {len(texts):>4} items   {len(works):>3} distinct opus")
        time.sleep(0.25)

    (OUT / "pianist_survey.json").write_text(
        json.dumps(survey, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- who clears the bar? ---------------------------------------------
    print()
    print("=" * 78)
    print("PIANISTS BY DISTINCT SOLO-CHOPIN OPUS NUMBERS FOUND")
    print("=" * 78)
    ranked = sorted(survey.items(), key=lambda kv: -len(kv[1]["works"]))
    for pianist, row in ranked:
        if not row["works"]:
            continue
        top = ", ".join(
            f"{k}({GENRE.get(int(k.split('.')[1].split('/')[0]), '?')[:4]})"
            for k, _ in collections.Counter(row["works"]).most_common(8)
        )
        print(f"  {pianist:<30} {len(row['works']):>3}  {top}")

    # --- densest pianist x work block ------------------------------------
    print()
    print("=" * 78)
    print("SHARED REPERTOIRE  (works held by the most of the top pianists)")
    print("=" * 78)
    core = [p for p, r in ranked if len(r["works"]) >= 6][:14]
    counts = collections.Counter()
    for pianist in core:
        for work in survey[pianist]["works"]:
            counts[work] += 1
    shared = [w for w, c in counts.most_common(14)]

    header = "".join(f"{w.replace('op.',''):>8}" for w in shared)
    print(f"{'pianist':<30}{header}")
    print("-" * (30 + 8 * len(shared)))
    for pianist in core:
        row = "".join(
            f"{'  X  ':>8}" if w in survey[pianist]["works"] else f"{'  .  ':>8}"
            for w in shared
        )
        print(f"{pianist:<30}{row}")
    print(f"{'TOTAL':<30}" + "".join(f"{counts[w]:>8}" for w in shared))

    json.dump(
        {"core": core, "shared_works": shared},
        (OUT / "pianist_core.json").open("w"),
        indent=2,
    )
    print("\n  -> results/pianist_survey.json, results/pianist_core.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
