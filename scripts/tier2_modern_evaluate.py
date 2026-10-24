"""Tier 2 - what transcription costs, measured on the same performances.

The archival tier can only answer this across datasets: one score on competition
pianists playing one set of works from exact MIDI, another on famous pianists playing
another set from old LPs, and the difference called the cost of transcription when
everything else differs too. Here each performance is measured twice, from its
note-exact MIDI and from its own audio, so the difference is transcription alone.

Four things follow. Note-level F1 of the transcription against ground truth, directly
comparable with the 0.12 measured on shellac. Per-feature survival, since the project's
headline is that voicing beats timing and the two need not degrade alike. Paired
identification on the same performers, works and splits, needing no matched
subsampling. And a within-year control: identification restricted to a single
competition, where the hall, the instrument and the engineers are fixed.

Alignment: MIDI and audio have independent time origins, so note-level comparison needs
a shift, estimated from the onset distributions before any scoring rather than by
sweeping for the best F1. The features never needed it, being differences between
onsets.

The paired arms also carry a randomisation test, 95% bootstrap intervals over sessions
stratified within performer, 2000 permutations, and a per-year null combined across
years. An earlier pass reported "80% of the signal retained" as a ratio of two
differences with no uncertainty on either and no test of the comparison.

Two pools, selected by --wide, differing only in the minimum performances per work. The
strict pool (3) is the published one. The wide pool (2) is three times the size and
needs scale="pooled"; see pianist_id/evaluate.loo_normalise.

  python scripts/tier2_modern_evaluate.py
  python scripts/tier2_modern_evaluate.py --scale pooled
  python scripts/tier2_modern_evaluate.py --wide
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id import features as F
from pianist_id.evaluate import (
    bootstrap_ci,
    estimate_offset,
    leave_one_out,
    loo_normalise,
    multi_group_mask,
    oof_predictions,
    paired_permutation,
    permutation_test,
    permute_by_group,
    retained_ci,
)

OUT = ROOT / "results"
NOTES = OUT / "tier2_modern_notes"
MIDI = ROOT / "data" / "asap" / "midi"
ASAP_RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
UA = {"User-Agent": "PianistIdentification/0.1"}
ONSET_TOLERANCE = 0.05
N_PERM = 2000
N_BOOT = 2000


def local_midi(rel: str) -> Path:
    """Fetch the ground-truth MIDI if it is not already on disk."""
    p = MIDI / rel.replace("/", "__")
    if not p.exists():
        MIDI.mkdir(parents=True, exist_ok=True)
        url = ASAP_RAW + urllib.request.quote(rel)
        try:
            data = urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=60).read()
            p.write_bytes(data)
        except Exception:  # noqa: BLE001
            return p
    return p


def note_f1(ref_notes, est_notes):
    """Greedy onset+pitch matching after estimating the global time offset."""
    ref_on = [n.onset for n in ref_notes]
    est_on = [float(n["onset_time"]) for n in est_notes]
    off = estimate_offset(ref_on, est_on)
    by_pitch: dict[int, list[float]] = {}
    for n in ref_notes:
        by_pitch.setdefault(int(n.pitch), []).append(float(n.onset))
    for v in by_pitch.values():
        v.sort()
    used: dict[int, set] = {p: set() for p in by_pitch}
    hits = 0
    for n in sorted(est_notes, key=lambda n: n["onset_time"]):
        pitch = int(n["midi_note"])
        onset = float(n["onset_time"]) + off
        best, best_d = None, ONSET_TOLERANCE
        for i, t in enumerate(by_pitch.get(pitch, [])):
            if i in used[pitch]:
                continue
            d = abs(t - onset)
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            used[pitch].add(best)
            hits += 1
    prec = hits / max(len(est_notes), 1)
    rec = hits / max(len(ref_notes), 1)
    return {"f1": 2 * prec * rec / max(prec + rec, 1e-9), "precision": prec,
            "recall": rec, "offset": off, "n_ref": len(ref_notes),
            "n_est": len(est_notes)}


def extract_pairs(rows):
    """Rows where both the transcription and the ground-truth MIDI exist."""
    Xa, Xm, meta, fid = [], [], [], []
    for r in rows:
        cached = NOTES / f"{r['key']}.json"
        if not cached.exists():
            continue
        est = json.loads(cached.read_text())
        if len(est) < 200:
            continue
        ns = [F.Note(n["onset_time"], n["midi_note"], n["velocity"],
                     n["offset_time"] - n["onset_time"]) for n in est]
        ns.sort(key=lambda n: (n.onset, n.pitch))
        fa = F.extract(ns)
        mp = local_midi(r["midi"])
        if not mp.exists() or "error" in fa:
            continue
        try:
            ref = F.load_midi(mp)
        except Exception:  # noqa: BLE001
            continue
        fm = F.extract(ref)
        if "error" in fm:
            continue
        va, vm = F.vector(fa), F.vector(fm)
        if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vm))):
            continue
        Xa.append(va)
        Xm.append(vm)
        fid.append(note_f1(ref, est))
        meta.append((r["performer"], r["work"], r["session"],
                     r["duration_s"], len(est), r["key"]))
    return np.array(Xa), np.array(Xm), meta, fid


def load_pairs(index: Path, use_cache=True):
    """Feature extraction, cached against the exact set of rows it came from.

    88 pairs means parsing 176 note files, and the wide pool's 270 takes minutes every
    run for a result that cannot change unless a transcription does. The cache is keyed
    on the sorted row keys, so adding one transcription invalidates it: this project has
    already evaluated eleven cached files its own selection no longer contained.
    """
    rows = json.loads(index.read_text())
    keys = sorted(r["key"] for r in rows
                  if (NOTES / f"{r['key']}.json").exists())
    cache = OUT / f"{index.stem}_features.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["keys"]) == keys:
            print(f"  features from cache ({len(keys)} rows)")
            return (z["Xa"], z["Xm"], [tuple(m) for m in z["meta"].tolist()],
                    list(z["fid"]))
        print("  cache does not match the current selection; re-extracting")
    Xa, Xm, meta, fid = extract_pairs(rows)
    np.savez(cache, Xa=Xa, Xm=Xm, meta=np.array(meta, dtype=object),
             fid=np.array(fid, dtype=object), keys=np.array(keys, dtype=object))
    return Xa, Xm, meta, fid


def ci_str(lo, hi):
    return f"[{100*lo:.2f}, {100*hi:.2f}]"


def main(argv: list[str] | None = None) -> int:
    from sklearn.preprocessing import OneHotEncoder

    argv = sys.argv[1:] if argv is None else argv
    wide = "--wide" in argv
    scale = "pooled" if wide else "per_work"
    if "--scale" in argv:
        scale = argv[argv.index("--scale") + 1]
    if wide and scale != "pooled":
        print("the wide pool requires --scale pooled; see loo_normalise")
        return 1
    n_perm = int(argv[argv.index("--perm") + 1]) if "--perm" in argv else N_PERM
    min_per_work = 2 if wide else 3
    index = OUT / ("asap_maestro_index_wide.json" if wide
                   else "asap_maestro_index.json")
    tag = ("wide" if wide else "strict") + f"-{scale}"
    if not index.exists():
        print(f"{index.name} not found; run the pipeline first")
        return 1

    print(f"pool {index.stem} (>= {min_per_work} performances per work), "
          f"scale={scale}, {n_perm} permutations")
    Xa, Xm, meta, fid = load_pairs(index, use_cache="--no-cache" not in argv)
    if len(meta) < 20:
        print(f"only {len(meta)} paired performances cached; run the pipeline first")
        return 1
    y = np.array([m[0] for m in meta])
    w = np.array([m[1] for m in meta])
    g = np.array([m[2] for m in meta])
    aux = np.array([[m[4], m[4] / (m[3] / 60.0)] for m in meta], dtype=float)

    keep, dropped = multi_group_mask(y, g)
    Xa, Xm, y, w, g, aux = Xa[keep], Xm[keep], y[keep], w[keep], g[keep], aux[keep]
    fid = [f for f, k in zip(fid, keep) if k]

    # One matrix for audio features, MIDI features and artefact proxies, so the
    # z-scoring and the restriction mask are provably identical across all three.
    # Columns are z-scored independently, so stacking only guarantees alignment.
    na, nm = Xa.shape[1], Xm.shape[1]
    stacked, mask = loo_normalise(np.hstack([Xa, Xm, aux]), y, w,
                                  min_per_work=min_per_work, scale=scale)
    An, Mn, auxn = (stacked[:, :na], stacked[:, na:na + nm], stacked[:, na + nm:])
    y, w, g = y[mask], w[mask], g[mask]
    Xa_raw, Xm_raw = Xa[mask], Xm[mask]
    fid = [f for f, k in zip(fid, mask) if k]
    n_cls = len(set(y))
    chance = 1.0 / n_cls

    print("=" * 78)
    print(f"TIER 2 [{tag}] {len(y)} performances, {n_cls} performers, "
          f"{len(set(g))} sessions, {len(set(w))} works")
    print("=" * 78)
    if dropped:
        print(f"  dropped (single session): {', '.join(map(str, dropped))}")

    # ---- 1. note-level fidelity ----------------------------------------
    f1s = np.array([f["f1"] for f in fid])
    offs = np.array([f["offset"] for f in fid])
    print("\n  NOTE FIDELITY vs ground truth")
    print(f"    F1        median {np.median(f1s):.3f}   "
          f"p10 {np.percentile(f1s,10):.3f}   p90 {np.percentile(f1s,90):.3f}")
    print(f"    precision median {np.median([f['precision'] for f in fid]):.3f}   "
          f"recall median {np.median([f['recall'] for f in fid]):.3f}")
    print(f"    estimated offsets: median {np.median(offs):+.2f}s, "
          f"|max| {np.abs(offs).max():.2f}s")
    print(f"    for comparison, two shellac transfers of one performance: 0.12")

    # ---- 2. which features survive -------------------------------------
    print("\n  PER-FEATURE SURVIVAL  (exact vs transcribed, same performances)")
    print(f"    {'feature':<28}{'pearson r':>11}{'spearman':>10}")
    surv = {}
    from scipy.stats import pearsonr, spearmanr
    for i, k in enumerate(F.NORMALISED_KEYS):
        a, b = Xa_raw[:, i], Xm_raw[:, i]
        r = pearsonr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan
        rs = spearmanr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan
        surv[k] = {"pearson": float(r), "spearman": float(rs)}
        print(f"    {k:<28}{r:>11.3f}{rs:>10.3f}")

    # ---- 3. paired identification --------------------------------------
    P = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
        w.reshape(-1, 1))
    rng = np.random.default_rng(0)
    per_class: dict = {}
    pred_trans = oof_predictions(An, y, g)
    pred_exact = oof_predictions(Mn, y, g)
    arms = [
        ("chance", chance),
        ("shuffled (session-level)", leave_one_out(An, permute_by_group(y, g, rng), g)[0]),
        ("piece identity only", leave_one_out(np.zeros((len(y), 1)), y, g, extra=P)[0]),
        ("transcription artefacts", leave_one_out(auxn, y, g)[0]),
        ("EXACT MIDI, session out", leave_one_out(Mn, y, g)[0]),
        ("TRANSCRIBED, session out", leave_one_out(An, y, g)[0]),
        ("transcribed, one track out", leave_one_out(An, y)[0]),
        ("piece + transcribed", leave_one_out(An, y, g, extra=P)[0]),
    ]
    print("\n  IDENTIFICATION  (balanced accuracy, leave-one-session-out, pooled)")
    for name, v in arms:
        print(f"    {name:<30}{100*v:>8.2f}%")

    # ---- 3b. intervals on the two headline arms ------------------------
    # Sessions are resampled within each performer. Unstratified, a draw loses every
    # session of a two-session performer about 13% of the time, and balanced accuracy
    # over a different class count is a different statistic. This resamples the
    # evaluation and not the training, since a duplicated session would sit on both
    # sides of its own split.
    print("\n  95% INTERVALS  (bootstrap over sessions, stratified by performer)")
    lo_t, hi_t, _ = bootstrap_ci(y, pred_trans, g, n_boot=N_BOOT)
    lo_e, hi_e, _ = bootstrap_ci(y, pred_exact, g, n_boot=N_BOOT)
    print(f"    transcribed  {100*dict(arms)['TRANSCRIBED, session out']:.2f}%  "
          f"{ci_str(lo_t, hi_t)}")
    print(f"    exact MIDI   {100*dict(arms)['EXACT MIDI, session out']:.2f}%  "
          f"{ci_str(lo_e, hi_e)}")

    # ---- 3b2. is the score spread across performers, or carried by a few?
    # Balanced accuracy is the mean of per-class recalls, so 7.90% over 23 performers
    # reads the same whether everyone is slightly identifiable or two are identified
    # perfectly and twenty-one never are. A broad weak effect is the claim this project
    # makes; a concentrated one would mean a few distinctive players and no method.
    print("\n  WHO IS IDENTIFIABLE  (per-class recall, pooled predictions)")
    for name, pr in (("transcribed", pred_trans), ("exact MIDI", pred_exact)):
        okp = np.array([q is not None for q in pr])
        rec = {}
        for lab in sorted(set(y)):
            sel = okp & (y == lab)
            rec[lab] = float(np.mean([q == lab for q in pr[sel]])) if sel.any() else 0.0
        vals = np.array(list(rec.values()))
        hit = {k: v for k, v in rec.items() if v > 0}
        top = sorted(hit.items(), key=lambda kv: -kv[1])[:5]
        print(f"    {name:<12} {len(hit)} of {len(rec)} performers above zero, "
              f"summing to {vals.sum():.2f} performer-equivalents")
        print(f"    {'':<12} best: "
              + ", ".join(f"{k} {100*v:.0f}%" for k, v in top))
        per_class[name] = rec

    # ---- 3c. is the drop from exact to transcribed real? ---------------
    # The arms are paired by construction: identical rows, folds and estimator, one
    # input matrix swapped. So under the null that the input makes no difference, which
    # arm a session's predictions came from is exchangeable.
    print("\n  THE DROP ITSELF  (paired, same folds, flipped at session level)")
    d_obs, d_null, d_p = paired_permutation(y, pred_exact, pred_trans, g,
                                            n_perm=n_perm)
    print(f"    exact - transcribed = {100*d_obs:+.2f} points   "
          f"null {100*d_null.mean():+.2f} +/- {100*d_null.std():.2f}   "
          f"p = {d_p:.4f}  (two-sided)")
    r_lo, r_hi, r_bad, _ = retained_ci(y, pred_exact, pred_trans, g, chance,
                                       n_boot=N_BOOT)
    print(f"    retained fraction 95% interval {r_lo:.2f} to {r_hi:.2f}"
          f"   ({r_bad} of {N_BOOT} draws undefined, exact margin <= 0)")

    # ---- 3d. recording-era control, done within a single competition ----
    # A decade probe cannot be run: the archival material for 1968-2000 does not exist
    # (16 qualifying Archive.org items against 224 for 1950-67), and holding out a
    # competition year fails because only 2 of 23 performers appear in more than one,
    # so year nests inside performer just as the LP transfer does.
    #
    # Restricting to one year inverts the problem. A single competition is one hall, one
    # instrument, one set of engineers, one fortnight, so if performers are separable
    # inside it, the recording era cannot be doing the work. Normalisation stays global,
    # being unsupervised with respect to performer; only the classification is
    # restricted.
    #
    # Each year carries its own session-level null and the years are combined into one
    # test, which is what gives the control any power: 15 to 36 performances resolve
    # nothing alone. The years were fixed by the data, and both the unweighted and the
    # performance-weighted margin are reported so the pooling choice is not
    # load-bearing. The per-year numbers were known when the combined test was written.
    print("\n  WITHIN-YEAR CONTROL  (recording conditions held constant)")
    years = np.array([s.split("/")[0] for s in g])
    within, subsets = {}, []
    for yr in sorted(set(years)):
        m = years == yr
        if m.sum() < 8:
            continue
        ysub, gsub = y[m], g[m]
        per_sessions = {p: len({s for q, s in zip(ysub, gsub) if q == p})
                        for p in set(ysub)}
        ok = np.array([per_sessions[p] >= 2 for p in ysub])
        if ok.sum() < 8 or len(set(ysub[ok])) < 4:
            continue
        Xs, ys, gs = An[m][ok], ysub[ok], gsub[ok]
        acc = leave_one_out(Xs, ys, gs)[0]
        ch = 1.0 / len(set(ys))
        yr_rng = np.random.default_rng(int(yr) if yr.isdigit() else 0)
        null = np.array([leave_one_out(Xs, permute_by_group(ys, gs, yr_rng), gs)[0]
                         for _ in range(n_perm)])
        p_yr = float((np.sum(null >= acc) + 1) / (len(null) + 1))
        within[yr] = {"balanced_accuracy": acc, "chance": ch,
                      "n": int(ok.sum()), "n_performers": len(set(ys)),
                      "null_mean": float(null.mean()), "p": p_yr}
        subsets.append((yr, Xs, ys, gs, acc, ch, int(ok.sum())))
        print(f"    {yr}  {len(set(ys)):>2} performers, {int(ok.sum()):>2} "
              f"performances   {100*acc:>6.2f}%   (1/n {100*ch:>5.2f}%, "
              f"null {100*null.mean():>5.2f}%, p = {p_yr:.3f})")
    if not within:
        print("    no year has enough performers with two sessions")

    combined = {}
    if len(subsets) >= 2:
        def stat(accs):
            marg = np.array([a - ch for a, (_, _, _, _, _, ch, _)
                             in zip(accs, subsets)])
            ns = np.array([s[6] for s in subsets], dtype=float)
            return float(marg.mean()), float((marg * ns).sum() / ns.sum())

        obs_u, obs_w = stat([s[4] for s in subsets])
        c_rng = np.random.default_rng(1)
        nulls_u, nulls_w = [], []
        for _ in range(n_perm):
            accs = [leave_one_out(Xs, permute_by_group(ys, gs, c_rng), gs)[0]
                    for _, Xs, ys, gs, _, _, _ in subsets]
            u, wt = stat(accs)
            nulls_u.append(u)
            nulls_w.append(wt)
        nulls_u, nulls_w = np.array(nulls_u), np.array(nulls_w)
        p_u = float((np.sum(nulls_u >= obs_u) + 1) / (len(nulls_u) + 1))
        p_w = float((np.sum(nulls_w >= obs_w) + 1) / (len(nulls_w) + 1))
        combined = {"n_years": len(subsets), "n_perm": len(nulls_u),
                    "unweighted_margin": obs_u, "unweighted_p": p_u,
                    "weighted_margin": obs_w, "weighted_p": p_w,
                    "unweighted_null_mean": float(nulls_u.mean()),
                    "weighted_null_mean": float(nulls_w.mean())}
        print(f"\n    COMBINED over {len(subsets)} years, "
              f"{len(nulls_u)} permutations within each")
        print(f"      mean margin above 1/n     {100*obs_u:+.2f} points   "
              f"null {100*nulls_u.mean():+.2f}   p = {p_u:.4f}")
        print(f"      weighted by performances  {100*obs_w:+.2f} points   "
              f"null {100*nulls_w.mean():+.2f}   p = {p_w:.4f}")

    res = dict(arms)
    exact, trans = res["EXACT MIDI, session out"], res["TRANSCRIBED, session out"]
    retained = (trans - chance) / max(exact - chance, 1e-9)
    print(f"\n    above-chance signal retained through transcription: "
          f"{100*retained:.0f}%")
    print(f"    session-grouping cost: "
          f"{100*(res['transcribed, one track out'] - trans):+.2f} points")

    obs, null, p = permutation_test(An, y, g, n_perm=n_perm)
    print(f"\n    permutation null (session-level, n={n_perm}): "
          f"{100*null.mean():.2f}% +/- {100*null.std():.2f}")
    print(f"    transcribed observed {100*obs:.2f}%  ->  p = {p:.4f}")
    obs_m, null_m, p_m = permutation_test(Mn, y, g, n_perm=n_perm)
    print(f"    exact-MIDI observed  {100*obs_m:.2f}%  ->  p = {p_m:.4f}")
    retained_null = ((trans - null.mean())
                     / max(exact - null_m.mean(), 1e-9))
    print(f"    retained, each arm against its own null "
          f"(({100*trans:.2f}-{100*null.mean():.2f})/"
          f"({100*exact:.2f}-{100*null_m.mean():.2f})): "
          f"{100*retained_null:.0f}%")

    out = OUT / f"tier2_modern_result{'' if tag == 'strict-per_work' else '_' + tag}.json"
    out.write_text(json.dumps({
        "pool": index.stem, "scale": scale, "min_per_work": min_per_work,
        "n_perm": n_perm, "n_boot": N_BOOT,
        "n_performances": int(len(y)), "n_performers": n_cls,
        "n_sessions": int(len(set(g))), "n_works": int(len(set(w))),
        "chance": chance,
        "note_f1": {"median": float(np.median(f1s)),
                    "p10": float(np.percentile(f1s, 10)),
                    "p90": float(np.percentile(f1s, 90))},
        "feature_survival": surv,
        "arms": {k: float(v) for k, v in arms},
        "ci_transcribed": [lo_t, hi_t], "ci_exact": [lo_e, hi_e],
        "paired_drop": {"observed": d_obs, "p": d_p,
                        "null_sd": float(d_null.std())},
        "signal_retained": float(retained),
        "signal_retained_from_null": float(retained_null),
        "signal_retained_ci": [r_lo, r_hi], "retained_undefined_draws": r_bad,
        "per_class_recall": per_class,
        "within_year": within, "within_year_combined": combined,
        "permutation_p_transcribed": p, "permutation_p_exact": p_m,
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> results/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
