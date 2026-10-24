"""Known-answer tests for the feature extractors, before any result uses them.

Five tests whose correct outcome is known in advance, so the extractors fail loudly
rather than return plausible numbers.

1. MECHANICAL SCORE
   ASAP's midi_score.mid is quantised, so asynchrony and melody lead must read ~0.
   Anything else means the extractor manufactures expression out of nothing.

2. SCALE INVARIANCE
   Rescale a real performance's time axis by 5%, the transfer-speed difference measured
   between two Cortot digitisations. Normalised features must not move; absolute
   millisecond features must move by exactly 5%.

3. INJECTED MELODY LEAD
   Shift every chord's top note earlier by a known amount in the mechanical score; the
   extractor must recover it. Tests sign and magnitude together.

4. REAL PERFORMANCES ARE NOT MECHANICAL
   Real performances must separate clearly from the score, or the features carry no
   interpretive information at all.

5. ONE TRACK MUST BE ONE PIECE
   Concatenate several performances, as a "complete op. 25" upload concatenates twelve
   etudes, and the features must move a lot. This test asserts the failure: it exists
   to justify the archival tier's duration-consistency filter, so a pass here means a
   large drift rather than a small one. Repeating it with no silence between the pieces
   must drift as much, since the cause is pooling heterogeneous material and not the
   gaps, and the two percentile-based IOI features must stay put as the gap grows while
   ioi_cv, built on a mean, must not.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import features as F

ASAP = ROOT / "data" / "asap" / "ballade1"
TOL_ZERO_MS = 6.0  # a quantised score should sit well inside this


def banner(text: str) -> None:
    print(f"\n{'='*74}\n{text}\n{'='*74}")


def main() -> int:
    score_path = ASAP / "midi_score.mid"
    if not score_path.exists():
        print(f"missing {score_path}")
        return 1

    passed, failed = [], []

    def check(name: str, ok: bool, detail: str) -> None:
        (passed if ok else failed).append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # --- 1. mechanical score --------------------------------------------
    banner("TEST 1  mechanical score must show no expression")
    score = F.load_midi(score_path)
    fs = F.extract(score)
    print(f"  {fs['n_notes']} notes, {fs['n_groups']} chord groups, "
          f"beat {fs['beat_sec']*1000:.0f} ms")
    check("score asynchrony ~ 0",
          abs(fs.get("asynchrony_ms_mean", 999)) < TOL_ZERO_MS,
          f"{fs.get('asynchrony_ms_mean', float('nan')):.2f} ms")
    check("score melody lead ~ 0",
          abs(fs.get("melody_lead_ms_mean", 999)) < TOL_ZERO_MS,
          f"{fs.get('melody_lead_ms_mean', float('nan')):.2f} ms")

    # --- 2. scale invariance --------------------------------------------
    banner("TEST 2  scale invariance under a 5% time-axis change (control 5)")
    perf_files = sorted(p for p in ASAP.glob("*.mid") if p.name != "midi_score.mid")
    if not perf_files:
        print("  no performance MIDIs found")
        return 1
    perf = F.load_midi(perf_files[0])
    f1 = F.extract(perf)
    f2 = F.extract(F.time_scale(perf, 1.05))
    print(f"  using {perf_files[0].name}")
    print(f"  {'feature':<28}{'original':>12}{'+5% time':>12}{'change':>10}")
    worst_norm, worst_name = 0.0, ""
    for k in ("asynchrony_beats_mean", "melody_lead_beats_mean", "ioi_log_sd",
              "ioi_ratio_p90_p10", "frac_melody_first"):
        a, b = f1.get(k, np.nan), f2.get(k, np.nan)
        rel = abs(b - a) / max(abs(a), 1e-9)
        if rel > worst_norm:
            worst_norm, worst_name = rel, k
        print(f"  {k:<28}{a:>12.5f}{b:>12.5f}{100*rel:>9.2f}%")
    for k in ("asynchrony_ms_mean", "melody_lead_ms_mean", "beat_sec"):
        a, b = f1.get(k, np.nan), f2.get(k, np.nan)
        print(f"  {k:<28}{a:>12.4f}{b:>12.4f}{100*(b/a-1):>9.2f}%")
    check("normalised features invariant", worst_norm < 0.01,
          f"worst drift {100*worst_norm:.2f}% ({worst_name})")
    check("absolute features scale by 5%",
          abs(f2["beat_sec"] / f1["beat_sec"] - 1.05) < 0.002,
          f"beat x{f2['beat_sec']/f1['beat_sec']:.4f}")

    # --- 3. injected melody lead ----------------------------------------
    banner("TEST 3  injected melody lead must be recovered")
    for inject_ms in (10.0, 30.0):
        groups = F.group_chords(score)
        shifted: list[F.Note] = []
        for g in groups:
            if len(g) >= 2:
                top = max(g, key=lambda n: n.pitch)
                for n in g:
                    shifted.append(
                        F.Note(n.onset - inject_ms / 1000 if n is top else n.onset,
                               n.pitch, n.velocity, n.duration))
            else:
                shifted.extend(g)
        shifted.sort(key=lambda n: (n.onset, n.pitch))
        fi = F.extract(shifted)
        got = fi.get("melody_lead_ms_mean", float("nan"))
        check(f"recovers {inject_ms:.0f} ms lead", abs(got - inject_ms) < 3.0,
              f"measured {got:.2f} ms")

    # --- 4. real performances differ from the score ---------------------
    banner("TEST 4  real performances must be measurably expressive")
    print(f"  {'performance':<20}{'async ms':>10}{'lead ms':>10}"
          f"{'async beats':>13}{'ioi_log_sd':>12}")
    rows = []
    for p in perf_files:
        f = F.extract(F.load_midi(p))
        rows.append(f)
        print(f"  {p.stem:<20}{f.get('asynchrony_ms_mean', np.nan):>10.1f}"
              f"{f.get('melody_lead_ms_mean', np.nan):>10.1f}"
              f"{f.get('asynchrony_beats_mean', np.nan):>13.4f}"
              f"{f.get('ioi_log_sd', np.nan):>12.4f}")
    async_real = np.array([r.get("asynchrony_ms_mean", np.nan) for r in rows])
    check("performances exceed the score's asynchrony",
          np.nanmin(async_real) > fs.get("asynchrony_ms_mean", 0) + 3,
          f"min {np.nanmin(async_real):.1f} ms vs score "
          f"{fs.get('asynchrony_ms_mean', float('nan')):.1f} ms")
    check("performances vary between themselves",
          float(np.nanstd(async_real)) > 1.0,
          f"sd across performances {np.nanstd(async_real):.2f} ms")

    # --- 5. one track must be one piece ---------------------------------
    banner("TEST 5  a whole-set track measures something else")
    keys = F.NORMALISED_KEYS

    def concat(parts, gap):
        out, t = [], 0.0
        for ns in parts:
            for n in ns:
                out.append(F.Note(n.onset + t, n.pitch, n.velocity, n.duration))
            t = out[-1].onset + gap
        return out

    def drift_of(parts, gap):
        base = np.nanmean([F.vector(F.extract(ns)) for ns in parts], axis=0)
        v = F.vector(F.extract(concat(parts, gap)))
        rel = np.abs((v - base) / np.where(np.abs(base) > 1e-12, np.abs(base), np.nan))
        return base, v, rel

    # (a) same piece, different performances -- pooling should be harmless
    same = [F.load_midi(p) for p in perf_files[:6]]
    _, _, rel_same = drift_of(same, 3.0)
    check("pooling performances of ONE piece leaves features alone",
          int(np.nansum(rel_same > 0.15)) <= 1,
          f"{int(np.nansum(rel_same > 0.15))} of {len(keys)} features moved, "
          f"median {100*np.nanmedian(rel_same):.0f}%")

    # (b) different pieces, one pianist: the whole-set upload case. Giltburg's four
    # movements of Chopin's second sonata are what an op. 35 upload is, one track and
    # four pieces.
    mv = sorted((ROOT / "data" / "asap" / "midi").glob("Chopin__Sonata_2__*Giltburg02.mid"))
    if len(mv) < 3:
        check("whole-set case available", False, f"found {len(mv)} movements, need 3+")
    else:
        diff = [F.load_midi(p) for p in mv]
        base, v3, rel3 = drift_of(diff, 3.0)
        print(f"  {'feature':<28}{'per-piece mean':>16}{'as one track':>14}{'drift':>8}")
        for k, b, c in zip(keys, base, v3):
            d = (c - b) / abs(b) if abs(b) > 1e-12 else np.nan
            print(f"  {k:<28}{b:>16.4f}{c:>14.4f}{100*d:>7.0f}%")
        moved = int(np.nansum(rel3 > 0.15))
        check("pooling DIFFERENT pieces moves the features", moved >= 3,
              f"{moved} of {len(keys)} features moved, median "
              f"{100*np.nanmedian(rel3):.0f}% ({len(mv)} movements)")

        _, _, rel0 = drift_of(diff, 0.0)
        check("that drift is the pooling, not the silence between pieces",
              abs(np.nanmedian(rel0) - np.nanmedian(rel3)) < 0.05,
              f"median drift {100*np.nanmedian(rel0):.0f}% with no gap vs "
              f"{100*np.nanmedian(rel3):.0f}% with 3 s")

        i_cv, i_sd, i_p = (keys.index("ioi_cv"), keys.index("ioi_log_sd"),
                           keys.index("ioi_ratio_p90_p10"))
        _, v_small, _ = drift_of(diff, 1.0)
        _, v_big, _ = drift_of(diff, 10.0)
        rob = max(abs(v_big[i_sd] - v_small[i_sd]) / abs(v_small[i_sd]),
                  abs(v_big[i_p] - v_small[i_p]) / abs(v_small[i_p]))
        check("percentile IOI features ignore gap length", rob < 0.05,
              f"log_sd and p90/p10 move {100*rob:.1f}% between 1 s and 10 s gaps")
        cv_move = abs(v_big[i_cv] - v_small[i_cv]) / abs(v_small[i_cv])
        check("ioi_cv does depend on gap length (it uses a mean)", cv_move > 0.10,
              f"ioi_cv moves {100*cv_move:.0f}% between 1 s and 10 s gaps")

    banner(f"{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("  failed: " + ", ".join(failed))
        print("\n  Do not use these features until the failures are understood.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
