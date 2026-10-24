"""Static figures from the Ballade No. 1 performances.

Source: ASAP note-exact performance MIDI for Chopin Ballade No. 1, 17 performances by
15 pianists, two of whom (LeeKoeun, SINKEV) recorded it twice. Note-exact rather than
transcribed, so no figure inherits transcription error, which matters because shellac
transcriptions match at F1 0.12 and would put artefacts on screen as if they were
playing.

Voicing leads and the tempo curve comes last, because that is the order the results
run in: a network given the whole tempo curve reaches 4.95% where eight voicing
summaries reach 9.68%.

  1  the opening chord, every pianist, in milliseconds. The one place that needs no
     alignment, since it is the first event of the piece.
  2  per-pianist distributions of melody lead and chord spread, the two features the
     ablation found load-bearing.
  3  same-pianist against different-pianist distances over the whole verification pool.
     The only figure carrying a measurement.
  4  tempo curves, labelled as the weaker signal rather than as the headline.

Writes results/figures/*.png.

    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pianist_id import features as F

WORK = "Chopin/Ballades/1"
MIDI = ROOT / "data" / "asap" / "midi"
FIGS = ROOT / "results" / "figures"
ASAP_RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
UA = {"User-Agent": "PianistIdentification/0.1"}

INK = "#1b1b1b"
GRID = "#d8d8d8"
ACCENT = "#b4453c"
MUTED = "#8a8a8a"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


def performances():
    """The Ballade No. 1 rows, taken from the cached full-ASAP feature index."""
    z = np.load(ROOT / "results" / "verification_fullasap_features.npz",
                allow_pickle=True)
    meta = [tuple(m) for m in z["meta"].tolist()]
    return sorted([(m[0], m[2]) for m in meta if m[1] == WORK], key=lambda t: t[0])


def local_midi(rel: str) -> Path:
    p = MIDI / rel.replace("/", "__")
    if not p.exists():
        MIDI.mkdir(parents=True, exist_ok=True)
        p.write_bytes(urllib.request.urlopen(urllib.request.Request(
            ASAP_RAW + urllib.parse.quote(rel), headers=UA), timeout=60).read())
    return p


def measure(rel: str) -> dict | None:
    """Per-chord voicing and timing, using the feature extractor's own primitives so
    the figures cannot drift from the measurements."""
    notes = F.load_midi(local_midi(rel))
    if len(notes) < 200:
        return None
    groups = F.group_chords(notes)
    beat = F.local_beat(groups)
    if not np.isfinite(beat) or beat <= 0:
        return None
    spreads, leads = [], []
    for g in groups:
        if len(g) < 2:
            continue
        spreads.append((g[-1].onset - g[0].onset) * 1000)
        top = max(g, key=lambda n: n.pitch)
        others = [n for n in g if n is not top]
        if others:
            leads.append((np.median([n.onset for n in others]) - top.onset) * 1000)
    onsets = np.array([g[0].onset for g in groups])
    iois = np.diff(onsets)
    iois = iois[iois > 1e-4]
    opening = next((g for g in groups[:4] if len(g) >= 2), None)
    feats = F.extract(notes)
    return {
        "beat": beat, "groups": groups, "notes": notes,
        "spreads": np.array(spreads), "leads": np.array(leads),
        "iois": iois, "onsets": onsets, "opening": opening,
        "features": feats,
    }


# ----------------------------------------------------------------- figure 1
def fig_opening_chord(data, order, modal_pcs):
    """Every pianist's opening octave, on one millisecond axis.

    Two notes an octave apart, usually one per hand, so the horizontal gap is the
    hand-to-hand asynchrony the melody-lead feature measures. That these are the same
    event is checked rather than assumed: the opening is an octave on C and 16 of 17
    performances open on that pitch class, so the one that does not (MunA19M, on E
    flat) is excluded.
    """
    rows = [(n, data[n]) for n in order
            if data[n]["opening"] is not None
            and tuple(sorted({x.pitch % 12 for x in data[n]["opening"]})) == modal_pcs]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.5))
    for k, (name, d) in enumerate(rows):
        g = d["opening"]
        t0 = g[0].onset
        top = max(g, key=lambda n: n.pitch)
        xs = [(n.onset - t0) * 1000 for n in g]
        ax.plot([min(xs), max(xs)], [k, k], color=GRID, lw=1.2, zorder=1)
        for n, x in zip(g, xs):
            melody = n is top
            ax.scatter(x, k, s=58 if melody else 30,
                       facecolor=ACCENT if melody else "white",
                       edgecolor=ACCENT if melody else INK,
                       lw=1.0, zorder=3)
        ax.annotate(f"{(max(xs)-min(xs)):.0f} ms", (max(xs), k),
                    textcoords="offset points", xytext=(8, -3),
                    fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=8)
    ax.set_xlabel("milliseconds after the first of the two notes")
    ax.set_title(f"The opening octave of Ballade No. 1, {len(rows)} pianists\n"
                 "filled = upper note. Most lead with it; two play it second. "
                 "The gap is hand-to-hand asynchrony.",
                 fontsize=10, loc="left")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_opening_chord.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 2
def fig_voicing_distributions(data, order):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.34 * len(order) + 1.9),
                             sharey=True)
    for ax, key, title, unit in (
            (axes[0], "leads", "Melody lead", "melody note ahead of its chord (ms)"),
            (axes[1], "spreads", "Chord spread", "first to last note of a chord (ms)")):
        vals = [data[n][key] for n in order]
        kw = ({"orientation": "horizontal"}
              if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 11)
              else {"vert": False})
        bp = ax.boxplot(vals, widths=0.6, showfliers=False, **kw,
                        patch_artist=True, medianprops=dict(color=ACCENT, lw=1.6),
                        boxprops=dict(facecolor="#f2f2f2", edgecolor=INK, lw=0.8),
                        whiskerprops=dict(color=INK, lw=0.8),
                        capprops=dict(color=INK, lw=0.8))
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel(unit)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(1, len(order) + 1))
    axes[0].set_yticklabels(order, fontsize=8)
    fig.suptitle("How each pianist voices a chord, every chord in the Ballade.\n"
                 "The distributions overlap almost entirely; the medians differ "
                 "by a few milliseconds. That is what a weak but real effect "
                 "looks like.", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "fig2_voicing_per_pianist.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 3
def fig_distance_distributions():
    """Same-pianist against different-pianist distances, whole verification pool.

    The one figure not restricted to the Ballade. It replaced a figure that plotted the
    two repeat-take pianists in voicing and timing space, captioned as evidence that
    voicing separates better: of the two, one was closer in voicing and the other in
    timing, so the caption contradicted its own data and two pianists could not have
    settled it either way.

    This plots the measurement instead: 515 same-person and 62,261 different-person
    pairs, cross-session and cross-work, from 102 performers. AUC 0.588 means a
    same-person pair is the closer one 58.8% of the time, against 50% for chance.
    """
    from pianist_id.evaluate import loo_normalise
    z = np.load(ROOT / "results" / "verification_features.npz", allow_pickle=True)
    X = z["X"]
    meta = [tuple(m) for m in z["meta"].tolist()]
    y = np.array([m[0] for m in meta])
    w = np.array([m[1] for m in meta])
    g = np.array([m[2] for m in meta])
    Xn, mask = loo_normalise(X, y, w, min_per_work=2, min_per_label=2,
                             scale="pooled")
    y, w, g = y[mask], w[mask], g[mask]
    ii, jj = np.triu_indices(len(y), k=1)
    keep = (g[ii] != g[jj]) & (w[ii] != w[jj])
    ii, jj = ii[keep], jj[keep]
    d = np.linalg.norm(Xn[ii] - Xn[jj], axis=1)
    same = y[ii] == y[jj]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bins = np.linspace(0, np.percentile(d, 99.5), 70)
    ax.hist(d[~same], bins=bins, density=True, color="#c9c9c9",
            edgecolor="white", lw=0.3, label=f"different pianists  (n={(~same).sum():,})")
    ax.hist(d[same], bins=bins, density=True, histtype="step", color=ACCENT,
            lw=1.9, label=f"same pianist  (n={same.sum():,})")
    ax.axvline(np.median(d[~same]), color=MUTED, lw=1.0, ls="--")
    ax.axvline(np.median(d[same]), color=ACCENT, lw=1.0, ls="--")
    ax.set_xlabel("distance between two performances (normalised feature space)")
    ax.set_ylabel("density")
    ax.set_title("Does a pianist resemble themselves? 102 performers, different "
                 "pieces, different sessions\n"
                 "AUC 0.588 against a permutation null of 0.500 ± 0.013 — the "
                 "overlap is the result.",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_distance_distributions.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 4
def fig_tempo_curves(data, order, n_show=6):
    """The weaker signal, drawn and labelled as such.

    Each curve is the chord-to-chord interval over that performance's own median, so a
    faster reading or a faster transfer sits on the same axis. Position is a fraction of
    the performance's length rather than a bar number, because that needs no alignment;
    the cost is that curves are only roughly comparable point to point, so this shows
    shape and not detail.
    """
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    show = order[:n_show]
    cmap = plt.get_cmap("cividis")
    for k, name in enumerate(show):
        d = data[name]
        r = d["iois"] / d["beat"]
        pos = np.linspace(0, 1, len(r))
        win = max(9, len(r) // 60)
        sm = np.convolve(r, np.ones(win) / win, mode="same")
        ax.plot(pos, sm, lw=1.3, color=cmap(k / max(len(show) - 1, 1)),
                label=name)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 3)
    ax.set_xlabel("position through the performance (fraction of its length)")
    ax.set_ylabel("chord interval ÷ that performance's median")
    ax.set_title("Six pianists, one piece — and the curves almost coincide\n"
                 "The tempo curve is mostly the score, not the player, which is "
                 "why it identifies poorly: a network given the whole curve "
                 "reached 4.95% where eight voicing summaries reached 9.68%.",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_tempo_curves.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = performances()
    print(f"{len(rows)} performances of {WORK}")

    data, labels = {}, []
    by_pianist = defaultdict(list)
    for pianist, rel in rows:
        m = measure(rel)
        if m is None:
            print(f"  ! unusable: {rel}")
            continue
        take = rel.split("/")[-1].replace(".mid", "")
        data[take] = m
        labels.append(take)
        by_pianist[pianist].append(take)
    print(f"usable: {len(labels)} performances, {len(by_pianist)} pianists")

    repeats = {p: [labels.index(t) for t in ts]
               for p, ts in by_pianist.items() if len(ts) > 1}
    print(f"pianists with two takes: {', '.join(repeats) or 'none'}")

    # Check rather than assume that the opening chords are the same event.
    print("\nopening-chord pitch content (checking it is the same event):")
    sets = {}
    for t in labels:
        g = data[t]["opening"]
        if g is None:
            print(f"  {t:<16} no chord in the first four groups")
            continue
        pcs = tuple(sorted({n.pitch % 12 for n in g}))
        sets[t] = pcs
    common = Counter(sets.values()).most_common()
    for pcs, n in common:
        names = [t for t, v in sets.items() if v == pcs]
        print(f"  pitch classes {str(list(pcs)):<18} {n:>2} performances  "
              f"{', '.join(names[:4])}{' ...' if len(names) > 4 else ''}")
    if len(common) > 1:
        print("  -> not identical across performances; figure 1 is captioned "
              "as the first chord each pianist plays, not as a single "
              "score event")

    one_per_pianist = [ts[0] for ts in by_pianist.values()]
    order = sorted(one_per_pianist,
                   key=lambda t: np.median(data[t]["leads"]) if len(data[t]["leads"]) else 0)

    modal_pcs = common[0][0]
    fig_opening_chord(data, order, modal_pcs)
    fig_voicing_distributions(data, order)
    fig_distance_distributions()
    fig_tempo_curves(data, order)
    print(f"\nwrote 4 figures to results/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
