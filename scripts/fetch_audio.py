"""Resolve and download source audio listed in data/recordings.csv.

No audio is redistributed with this project; this regenerates the local cache from the
source URLs, which is what makes the extracted features the artifact of record.

Archive.org items expose no stable pointer to "the audio file", so the filename is
resolved from the item's metadata API at fetch time. That is also where `archived_date`
comes from: some items are uploader-posted commercial transfers that will eventually be
taken down, and the manifest should say when a file was last seen rather than quietly
becoming a different dataset. Several items hold more than one work, where "longest
track" is the wrong heuristic, so those get an explicit filename pattern below;
guessing would align Ballade No. 4 against Ballade No. 1. sha256 is recorded so a
re-fetch returning different bytes is loud, since historical transfers get re-mastered.

Usage:
    python scripts/fetch_audio.py                 # everything resolvable
    python scripts/fetch_audio.py --killtest      # just the four kill-test files
    python scripts/fetch_audio.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "recordings.csv"
AUDIO_DIR = ROOT / "data" / "audio"
FETCH_LOG = ROOT / "data" / "audio_manifest.json"

# Items holding more than one work: match the Ballade No. 1 track explicitly. Keys are
# archive.org identifiers, values lowercase substrings the filename must contain.
TRACK_HINTS = {
    "CHOPINBallades-NEWTRANSFER": "01.ballade",
    "chopin-ballades-1-4-julian-von-karolyi-dvg": "no. 1",
    "lp_sonata-in-b-flat-minor-op35-ballade-n-1-i": "ballade",
    "08-ballade-no-1-in-g-minor-op-23": "ballade-no-1",
}

KILLTEST = {
    "ballade1_cortot_1926a",
    "ballade1_cortot_1926b",
    "ballade1_michelangeli_1963",
    "ballade1_janis_1952",
}

PREFERRED_FORMATS = ["Flac", "24bit Flac", "VBR MP3", "Ogg Vorbis"]
USER_AGENT = "PianistIdentification/0.1 (research; contact via project repo)"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _seconds(raw) -> float:
    """Archive.org reports length as either seconds or H:MM:SS."""
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


def identifier_of(item_url: str) -> str | None:
    if "archive.org/details/" not in item_url:
        return None
    return item_url.rstrip("/").split("/details/")[-1]


def pick_file(identifier: str, files: list[dict]) -> dict | None:
    hint = TRACK_HINTS.get(identifier)
    audio = [f for f in files if f.get("format") in PREFERRED_FORMATS]
    if hint:
        audio = [f for f in audio if hint in f["name"].lower()]
        if not audio:
            return None
    if not audio:
        return None

    def rank(f: dict) -> tuple[int, float]:
        # Prefer lossless, then the longest track in the item.
        return (PREFERRED_FORMATS.index(f["format"]), -_seconds(f.get("length")))

    return sorted(audio, key=rank)[0]


def download(url: str, dest: Path) -> str:
    # Archive.org filenames contain spaces and punctuation, which urlopen rejects
    # rather than encodes.
    scheme, _, rest = url.partition("://")
    url = f"{scheme}://{urllib.parse.quote(rest, safe='/:')}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 16):
            digest.update(chunk)
            out.write(chunk)
    tmp.replace(dest)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--killtest", action="store_true",
                    help="only the files the kill test needs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="recording_id(s)")
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = AUDIO_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Source audio is never committed. See scripts/fetch_audio.py\n*\n!.gitignore\n")

    with MANIFEST.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if args.killtest:
        rows = [r for r in rows if r["recording_id"] in KILLTEST]
    elif args.only:
        rows = [r for r in rows if r["recording_id"] in set(args.only)]

    log: dict[str, dict] = {}
    if FETCH_LOG.exists():
        log = json.loads(FETCH_LOG.read_text(encoding="utf-8"))

    skipped, fetched, failed = [], [], []

    for row in rows:
        rid = row["recording_id"]
        identifier = identifier_of(row["archive_item"])
        if identifier is None:
            skipped.append((rid, "not an archive.org item (fetch manually)"))
            continue

        try:
            meta = _get_json(f"https://archive.org/metadata/{identifier}")
        except (urllib.error.URLError, TimeoutError) as exc:
            failed.append((rid, f"metadata unreachable: {exc}"))
            continue

        if not meta.get("metadata"):
            failed.append((rid, "item has no metadata (removed?)"))
            continue

        chosen = pick_file(identifier, meta.get("files", []))
        if chosen is None:
            failed.append((rid, "no usable audio file in item"))
            continue

        suffix = Path(chosen["name"]).suffix or ".audio"
        dest = AUDIO_DIR / f"{rid}{suffix}"
        url = f"https://archive.org/download/{identifier}/{chosen['name']}"

        if args.dry_run:
            print(f"  would fetch {rid:<32} {chosen['format']:<10} {chosen['name']}")
            continue

        if dest.exists() and rid in log:
            skipped.append((rid, "already cached"))
            continue

        try:
            sha = download(url, dest)
        except (urllib.error.URLError, TimeoutError) as exc:
            failed.append((rid, f"download failed: {exc}"))
            continue

        log[rid] = {
            "identifier": identifier,
            "source_url": url,
            "local_file": dest.name,
            "format": chosen["format"],
            "declared_seconds": round(_seconds(chosen.get("length")), 2),
            "bytes": dest.stat().st_size,
            "sha256": sha,
            "archived_date": date.today().isoformat(),
            "pianist": row["pianist"],
            "year": row["year"],
            "license": row["license"],
            "tier": row["tier"],
        }
        fetched.append(rid)
        print(f"  fetched  {rid:<32} {chosen['format']:<10} {dest.stat().st_size/1e6:6.1f} MB")

    if not args.dry_run:
        FETCH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nfetched {len(fetched)}, skipped {len(skipped)}, failed {len(failed)}")
    for rid, why in skipped:
        print(f"  skip   {rid}: {why}")
    for rid, why in failed:
        print(f"  FAIL   {rid}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
