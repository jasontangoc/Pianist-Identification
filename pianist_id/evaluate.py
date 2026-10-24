"""Evaluation machinery shared by every tier.

Two things drive the design. Features are z-scored against other performances of the
same work, leave-one-out, because pianists do not play the same repertoire and raw
values partly encode the piece. And cross-validation is grouped by whatever is nested
inside the label: in the archival tier none of the 44 Archive.org items held two
pianists, so under random k-fold a model can score by recognising the transfer. The
group is the competition session for MAESTRO and the item for Archive.org.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

MIN_GROUPS_PER_LABEL = 2


def loo_normalise(X, y, works, min_per_work=3, min_per_label=3,
                  scale="per_work"):
    """Leave-one-out per-work normalisation, plus the standard restriction.

    Returns the normalised matrix and a mask over the input rows, so every array
    travelling alongside X is filtered by the same rule.

    The two scalings differ in what sets the minimum work size. `per_work` divides by
    the standard deviation of the other performances of the same work, which needs three
    per work: with two, that deviation is taken over one value and is zero, and the row
    would stay in raw units alongside z-scored rows. `pooled` keeps centring per work
    (one other performance is enough to remove the piece effect) and uses a single
    within-work divisor per feature, so min_per_work=2 is honest. The pooled divisor
    excludes the row own deviation, as the per-work mean excludes the row itself.
    """
    if scale not in ("per_work", "pooled"):
        raise ValueError(f"unknown scale {scale!r}")
    if scale == "per_work" and min_per_work < 3:
        raise ValueError(
            "scale='per_work' with min_per_work < 3 leaves rows centred in raw "
            "units alongside z-scored rows; use scale='pooled'")

    Xn = np.zeros_like(X)
    D = np.zeros_like(X)
    centred = np.zeros(len(X), dtype=bool)
    for w in np.unique(works):
        idx = np.where(works == w)[0]
        if len(idx) < min_per_work:
            continue
        block = X[idx]
        for k, i in enumerate(idx):
            others = np.delete(block, k, axis=0)
            mu = others.mean(axis=0)
            if scale == "per_work":
                sd = others.std(axis=0)
                sd[sd < 1e-9] = 1.0
                Xn[i] = (X[i] - mu) / sd
            else:
                D[i] = X[i] - mu
            centred[i] = True

    if scale == "pooled":
        d = D[centred]
        s1, s2, N = d.sum(axis=0), (d ** 2).sum(axis=0), len(d)
        if N < 3:
            raise ValueError("too few centred rows to pool a scale")
        for i in np.where(centred)[0]:
            m = N - 1
            mu_d = (s1 - D[i]) / m
            var = (s2 - D[i] ** 2) / m - mu_d ** 2
            sd = np.sqrt(np.maximum(var, 0.0))
            sd[sd < 1e-9] = 1.0
            Xn[i] = D[i] / sd

    Xn = np.clip(Xn, -6, 6)
    wc = Counter(works)
    mask = np.array([wc[w] >= min_per_work for w in works])
    pc = Counter(y[mask])
    mask &= np.array([pc.get(v, 0) >= min_per_label for v in y])
    return Xn[mask], mask


def independent_groups(y, works, items):
    """Merge source items that cannot be treated as independent.

    Holding out an item stops a test recording sharing a transfer with the training set
    but not a performance. Nine archival (pianist, work) pairs appear in two or more
    items at durations within a percent of each other, which is more likely a reissue of
    one take than two readings. Treating any two items sharing a (label, work) as one
    group errs safe: at worst it merges two real readings and makes the test harder.
    Near-identity in MAESTRO, where one of 510 pairs spans two sessions.
    """
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for it in items:
        find(it)
    shared: dict[tuple, list] = {}
    for lab, w, it in zip(y, works, items):
        shared.setdefault((lab, w), []).append(it)
    for its in shared.values():
        first = its[0]
        for other in its[1:]:
            ra, rb = find(first), find(other)
            if ra != rb:
                parent[ra] = rb
    return np.array([find(it) for it in items])


def multi_group_mask(y, groups, minimum=MIN_GROUPS_PER_LABEL):
    """Rows whose label is represented by more than one independent group.

    A pianist with one source can only be tested against their own session, so a score
    for them measures the session and the playing together.
    """
    by_label: dict[str, set] = {}
    for lab, g in zip(y, groups):
        by_label.setdefault(lab, set()).add(g)
    mask = np.array([len(by_label[lab]) >= minimum for lab in y])
    dropped = sorted(lab for lab, gs in by_label.items() if len(gs) < minimum)
    return mask, dropped


def leave_one_out(X, y, groups=None, extra=None):
    """Leave-one-group-out, scoring pooled out-of-fold predictions.

    Pooling is forced: a held-out group usually holds one performer, so balanced
    accuracy inside a fold is 1 or 0 for the one class present and undefined for the
    rest. One group out at a time also beats k-fold here, because groups are very
    unequal (one transfer holds 9 of Brailowsky's 10 recordings) and a 3-fold split can
    leave a performer with a single training example.

    `groups=None` leaves out one recording at a time: the same estimator without the
    grouping, so the gap between the two is the share of accuracy that was the recording.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    Xu = X if extra is None else np.hstack([extra, X])
    g = np.arange(len(y)).astype(str) if groups is None else np.asarray(groups)
    pred = np.empty(len(y), dtype=object)
    for held in dict.fromkeys(g):
        te = np.where(g == held)[0]
        tr = np.where(g != held)[0]
        if len(set(y[tr])) < 2:
            pred[te] = None
            continue
        lr = LogisticRegression(max_iter=4000, class_weight="balanced").fit(
            Xu[tr], y[tr])
        pred[te] = lr.predict(Xu[te])
    ok = np.array([p is not None for p in pred])
    if ok.sum() == 0:
        return float("nan"), 0.0
    return float(balanced_accuracy_score(y[ok], np.array(list(pred[ok])))), 0.0


def oof_predictions(X, y, groups=None, extra=None):
    """The out-of-fold predictions behind a `leave_one_out` score.

    Same estimator and folds, returning the predictions rather than the score. The
    paired and resampling functions below work on these, so two arms can be compared row
    by row and a resample re-scored without refitting.
    """
    from sklearn.linear_model import LogisticRegression

    Xu = X if extra is None else np.hstack([extra, X])
    g = np.arange(len(y)).astype(str) if groups is None else np.asarray(groups)
    pred = np.empty(len(y), dtype=object)
    for held in dict.fromkeys(g):
        te = np.where(g == held)[0]
        tr = np.where(g != held)[0]
        if len(set(y[tr])) < 2:
            pred[te] = None
            continue
        lr = LogisticRegression(max_iter=4000, class_weight="balanced").fit(
            Xu[tr], y[tr])
        pred[te] = lr.predict(Xu[te])
    return pred


def _score(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def _groups_by_label(y, groups, ok):
    """Distinct groups per label, and the usable rows of each group."""
    by_label: dict = {}
    for lab, g in zip(y[ok], groups[ok]):
        seen = by_label.setdefault(lab, [])
        if g not in seen:
            seen.append(g)
    rows_of = {g: np.where((groups == g) & ok)[0] for g in set(groups[ok])}
    return by_label, rows_of


def bootstrap_ci(y, pred, groups, n_boot=2000, seed=0, alpha=0.05):
    """Confidence interval for balanced accuracy, resampling groups within each label.

    Unstratified, a bootstrap over 67 sessions loses every session of a two-session
    performer about 13% of the time, and balanced accuracy over a different class count
    against a different chance level is a different statistic. Stratifying keeps the
    chance level at 1/n in every draw, at the cost of assuming a performer's own
    sessions are exchangeable.

    Fits and folds are held fixed, so this is an interval on the score given these fits.
    Refitting per draw is not possible here: a duplicated session would sit on both
    sides of its own leave-one-out split.
    """
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    ok = np.array([p is not None for p in pred])
    by_label, rows_of = _groups_by_label(y, groups, ok)
    draws = np.empty(n_boot)
    for j in range(n_boot):
        take = [rows_of[g] for gs in by_label.values()
                for g in rng.choice(gs, size=len(gs), replace=True)]
        idx = np.concatenate(take)
        draws[j] = _score(y[idx], np.array(list(pred[idx])))
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))), draws)


def paired_permutation(y, pred_a, pred_b, groups, n_perm=2000, seed=0):
    """Is the gap between two arms evaluated on the same folds real?

    The arms are paired by construction: same performances, folds and estimator, one
    input matrix swapped. Under the null that the input makes no difference, which arm a
    session's predictions came from is exchangeable, so the test flips that assignment
    at session level. Rows inside a session share a performer and are not exchangeable
    on their own. Two-sided, on the difference in balanced accuracy.
    """
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    ok = np.array([p is not None and q is not None
                   for p, q in zip(pred_a, pred_b)])
    ya, g = y[ok], groups[ok]
    a = np.array(list(pred_a[ok]))
    b = np.array(list(pred_b[ok]))
    observed = _score(ya, a) - _score(ya, b)
    uniq = list(dict.fromkeys(g))
    null = np.empty(n_perm)
    for j in range(n_perm):
        flip = dict(zip(uniq, rng.integers(0, 2, len(uniq)).astype(bool)))
        sw = np.array([flip[u] for u in g])
        null[j] = (_score(ya, np.where(sw, b, a))
                   - _score(ya, np.where(sw, a, b)))
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return observed, null, p


def retained_ci(y, pred_exact, pred_trans, groups, chance,
                n_boot=2000, seed=0, alpha=0.05):
    """Interval for the retained-signal fraction, resampled the same way.

    (transcribed - chance) / (exact - chance) on each stratified draw, so both arms move
    together. Draws with a denominator at or below zero are dropped and counted; with a
    margin this thin that is not rare, so the count is returned rather than hidden.
    """
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    ok = np.array([p is not None and q is not None
                   for p, q in zip(pred_exact, pred_trans)])
    by_label, rows_of = _groups_by_label(y, groups, ok)
    vals, bad = [], 0
    for _ in range(n_boot):
        take = [rows_of[g] for gs in by_label.values()
                for g in rng.choice(gs, size=len(gs), replace=True)]
        idx = np.concatenate(take)
        e = _score(y[idx], np.array(list(pred_exact[idx]))) - chance
        t = _score(y[idx], np.array(list(pred_trans[idx]))) - chance
        if e <= 0:
            bad += 1
            continue
        vals.append(t / e)
    vals = np.array(vals)
    if len(vals) < 20:
        return float("nan"), float("nan"), bad, vals
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))), bad, vals)


def permute_by_group(y, groups, rng):
    """Shuffle labels at group level, preserving the nesting structure.

    Permuting recordings would break the fact that everything from one session shares a
    performer, giving a null for an easier question that sits too low.
    """
    uniq = list(dict.fromkeys(groups))
    groups = np.asarray(groups)
    true = {g: y[np.where(groups == g)[0][0]] for g in uniq}
    shuffled = rng.permutation([true[g] for g in uniq])
    mapping = dict(zip(uniq, shuffled))
    return np.array([mapping[g] for g in groups])


def permutation_test(X, y, groups, n_perm=200, seed=0):
    """Group-level permutation null for a leave-one-group-out score."""
    rng = np.random.default_rng(seed)
    observed = leave_one_out(X, y, groups)[0]
    null = np.array([leave_one_out(X, permute_by_group(y, groups, rng), groups)[0]
                     for _ in range(n_perm)])
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return observed, null, p


# The archival scripts were written before "item" became the wrong general word.
multi_item_mask = multi_group_mask
permute_by_item = permute_by_group


def estimate_offset(ref_onsets, est_onsets, max_shift=10.0, resolution=0.01):
    """Global time offset between two onset sequences, by cross-correlation.

    ASAP MIDI and MAESTRO audio have independent time origins (first onsets run from 0
    to 269 s), so a note-level comparison has to align them first. Sweeping the shift
    for the one that maximises F1 would fit the metric to its own answer, so the peak is
    taken from onset histograms before any scoring. Returns seconds to ADD to the
    estimate to land on the reference timeline.
    """
    ref = np.asarray(sorted(ref_onsets), dtype=float)
    est = np.asarray(sorted(est_onsets), dtype=float)
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    lo = min(ref.min(), est.min()) - max_shift
    hi = max(ref.max(), est.max()) + max_shift
    bins = np.arange(lo, hi + resolution, resolution)
    hr, _ = np.histogram(ref, bins=bins)
    he, _ = np.histogram(est, bins=bins)
    hr = hr - hr.mean()
    he = he - he.mean()
    corr = np.correlate(hr, he, mode="full")
    lags = np.arange(-len(he) + 1, len(hr)) * resolution
    keep = np.abs(lags) <= max_shift
    if not keep.any():
        return 0.0
    return float(lags[keep][int(np.argmax(corr[keep]))])
