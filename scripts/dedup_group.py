"""Cluster candidate tracks of one (pianist, work) into distinct performances.

This is the step metadata cannot do. `resolve_candidates.py` narrows thousands of hits
to a few candidates per work; this decides which of those are the same performance
reissued and which are separate recordings, the distinction the recording-level split
depends on. Each candidate is downloaded, every pair compared with `pianist_id.identity`,
and the connected components of the "same performance" relation taken.

Transitivity is assumed and not guaranteed. If A matches B and B matches C but A does
not match C, the component merges all three and the run warns, because that pattern
means the threshold is wrong for this material.

Usage:
    python scripts/dedup_group.py --pianist "Alfred Cortot" --work op.23
    python scripts/dedup_group.py --top 5          # busiest groups
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import identity

CANDIDATES = ROOT / "results" / "candidate_recordings.csv"
AUDIO = ROOT / "data" / "audio" / "candidates"
FPCACHE = ROOT / "results" / "fingerprints"
OUT = ROOT / "results" / "dedup"
UA = {"User-Agent": "PianistIdentification/0.1"}


def local_name(row: dict) -> str:
    stem = hashlib.sha1(
        f"{row['identifier']}/{row['filename']}".encode()
    ).hexdigest()[:16]
    return f"{stem}{Path(row['filename']).suffix or '.audio'}"


def download(row: dict) -> Path | None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    dest = AUDIO / local_name(row)
    if dest.exists():
        return dest
    # Archive.org filenames contain spaces, commas and parentheses, which urlopen
    # rejects rather than encodes, so quote the path segments but not the separators.
    safe = urllib.parse.quote(f"{row['identifier']}/{row['filename']}", safe="/")
    url = f"https://archive.org/download/{safe}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as out:
                while chunk := r.read(1 << 16):
                    out.write(chunk)
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code in (500, 502, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))  # the Archive 500s intermittently
                continue
            print(f"      ! download failed: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"      ! download failed: {exc}")
            return None
    return None


def cached_fingerprint(path: Path) -> np.ndarray:
    """Each file is compared N-1 times; computing its chroma once matters."""
    FPCACHE.mkdir(parents=True, exist_ok=True)
    dest = FPCACHE / f"{path.stem}.npy"
    if dest.exists():
        return np.load(dest)
    fp = identity.fingerprint(path)
    np.save(dest, fp)
    return fp


def compare_cached(a: Path, b: Path):
    fa, fb = cached_fingerprint(a), cached_fingerprint(b)
    original = identity.fingerprint
    identity.fingerprint = lambda p: fa if Path(p) == a else fb
    try:
        return identity.compare(a, b)
    finally:
        identity.fingerprint = original


def components(n: int, edges: set[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def run_group(pianist: str, work: str, rows: list[dict]) -> dict:
    print(f"\n{'='*86}\n{pianist} — {work}   ({len(rows)} candidate tracks)\n{'='*86}")

    paths, kept = [], []
    for row in rows:
        path = download(row)
        if path is None:
            continue
        paths.append(path)
        kept.append(row)
        print(f"   {float(row['duration_s']):>6.0f}s  {row['year'] or '----':<5} "
              f"{row['identifier'][:38]:<39} {row['filename'][:26]}")

    if len(paths) < 2:
        print("   fewer than two usable tracks; nothing to cluster")
        return {"pianist": pianist, "work": work, "clusters": []}

    edges, table = set(), []
    for (i, pa), (j, pb) in itertools.combinations(list(enumerate(paths)), 2):
        v = compare_cached(pa, pb)
        table.append((i, j, v))
        if v.same_performance:
            edges.add((i, j))

    print(f"\n   {'pair':<10}{'resid%':>9}{'slope':>9}  verdict")
    print("   " + "-" * 76)
    for i, j, v in table:
        mark = "SAME" if v.same_performance else "diff"
        resid = "  n/a" if np.isnan(v.residual_pct) else f"{v.residual_pct:>8.3f}"
        slope = "  n/a" if np.isnan(v.slope) else f"{v.slope:>8.3f}"
        print(f"   {i}-{j:<8}{resid}{slope}  {mark}  {v.note}")

    groups = components(len(paths), edges)
    # transitivity check
    for group in groups:
        for i, j in itertools.combinations(sorted(group), 2):
            if (i, j) not in edges and (j, i) not in edges:
                print(f"   ! WARNING non-transitive: {i} and {j} merged indirectly")

    print(f"\n   -> {len(groups)} distinct performance(s)")
    clusters = []
    for n, group in enumerate(sorted(groups, key=min), 1):
        years = sorted({kept[i]["year"] for i in group if kept[i]["year"]})
        durs = [float(kept[i]["duration_s"]) for i in group]
        print(f"      performance {n}: {len(group)} file(s), "
              f"{min(durs):.0f}-{max(durs):.0f}s, years {years or ['unknown']}")
        clusters.append(
            {
                "performance": n,
                "years": years,
                "tracks": [
                    {k: kept[i][k] for k in ("identifier", "filename", "duration_s", "year")}
                    for i in group
                ],
            }
        )
    return {"pianist": pianist, "work": work, "clusters": clusters}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pianist")
    ap.add_argument("--work")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    with CANDIDATES.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["pianist"], row["work"])].append(row)

    if args.pianist and args.work:
        targets = [(args.pianist, args.work)]
    elif args.top:
        SINGLE = {20, 23, 29, 31, 36, 38, 39, 44, 45, 47, 49, 51, 52, 53, 57, 60, 61, 66}

        def single_piece(work: str) -> bool:
            if "/" in work:
                return True
            try:
                return int(work.split(".")[1]) in SINGLE
            except (IndexError, ValueError):
                return False

        ranked = sorted(
            ((k, v) for k, v in buckets.items() if single_piece(k[1]) and len(v) > 1),
            key=lambda kv: -len(kv[1]),
        )
        targets = [k for k, _ in ranked[: args.top]]
    else:
        ap.error("pass --pianist and --work, or --top N")

    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_group(p, w, buckets[(p, w)]) for p, w in targets]
    dest = OUT / "clusters.json"
    dest.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
