"""Known-answer tests for the shared evaluation machinery.

`validate_features.py` does this for the extractors, and every real bug it found came
from a case whose answer was known in advance. The evaluation module had no equivalent,
which mattered once it grew a second normalisation and three resampling tests: the
failure mode of a resampling test is a plausible number.

Each test states its answer before computing it. Run before trusting any tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pianist_id.evaluate import (
    bootstrap_ci,
    leave_one_out,
    loo_normalise,
    oof_predictions,
    paired_permutation,
    permute_by_group,
    retained_ci,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"           {detail}")
    if not ok:
        FAILURES.append(name)


def reference_per_work(X, works, min_per_work=3):
    """The normaliser exactly as it stood before the pooled scaling was added.

    Kept here rather than described, so that the refactor is checked against
    running code and not against a reading of it.
    """
    Xn = np.zeros_like(X)
    for w in np.unique(works):
        idx = np.where(works == w)[0]
        if len(idx) < min_per_work:
            continue
        block = X[idx]
        for k, i in enumerate(idx):
            others = np.delete(block, k, axis=0)
            mu, sd = others.mean(axis=0), others.std(axis=0)
            sd[sd < 1e-9] = 1.0
            Xn[i] = (X[i] - mu) / sd
    return np.clip(Xn, -6, 6)


# ---------------------------------------------------------------- test 1
def test_per_work_unchanged() -> None:
    """Answer known in advance: the refactor must not move a single value.

    The per-work branch was rewritten to share its centring step with the
    pooled branch. Mathematically the same; the test is whether it is also the
    same in floating point, because every published tier number came out of the
    old arithmetic.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 11)) * rng.uniform(0.5, 50, size=11)
    works = np.array([f"w{i % 12}" for i in range(60)])
    y = np.array([f"p{i % 9}" for i in range(60)])
    got, mask = loo_normalise(X, y, works, scale="per_work")
    want = reference_per_work(X, works)[mask]
    worst = float(np.abs(got - want).max())
    check("per-work scaling is bit-for-bit what it was before the refactor",
          np.array_equal(got, want), f"largest difference {worst:.2e}")


# ---------------------------------------------------------------- test 2
def test_per_work_refuses_two() -> None:
    """Answer known in advance: it must refuse, not quietly do the wrong thing.

    With two performances of a work the leave-one-out standard deviation is
    taken over one value, is zero, and gets replaced by 1.0 -- leaving that row
    centred in raw feature units beside rows that are z-scores. The old code
    accepted this silently and that silence is what kept the wide pool out of
    reach, so it is now an error.
    """
    X = np.arange(24.0).reshape(8, 3)
    works = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    y = np.array(["p", "p", "p", "q", "q", "q", "r", "r"])
    try:
        loo_normalise(X, y, works, min_per_work=2, scale="per_work")
    except ValueError:
        check("per-work scaling refuses min_per_work=2", True)
        return
    check("per-work scaling refuses min_per_work=2", False,
          "it returned a matrix instead of raising")


# ---------------------------------------------------------------- test 3
def test_pooled_puts_every_row_on_one_scale() -> None:
    """The exact bug the pooled scaling exists to fix.

    Two works with the *same* within-work spread, one holding four
    performances and one holding two, in a feature whose raw units are large.
    Because the spreads are equal, every row belongs on the same scale, and
    that is the known answer the two treatments are judged against.

    Under the old treatment the four-performance work gets a real standard
    deviation and comes out near 1, while the two-performance work gets the
    1.0 fallback and comes out in raw units -- 1000 before the clip, 6 after
    it. Two scales in one matrix, with the clip hiding how far apart they were.
    Under pooled scaling one divisor covers both, so the two works land within
    a factor of two of each other, the difference left being the leave-one-out
    mean being taken over three others in one work and one in the other.
    """
    X = np.array([[0.0], [1000.0], [0.0], [1000.0], [0.0], [1000.0]])
    works = np.array(["a", "a", "a", "a", "b", "b"])
    y = np.array(["p", "q", "p", "q", "p", "q"])
    Xn, mask = loo_normalise(X, y, works, min_per_work=2, min_per_label=3,
                             scale="pooled")
    old = reference_per_work(X, works, min_per_work=2)[mask]
    new_ratio = abs(Xn[5, 0]) / abs(Xn[1, 0])
    old_ratio = abs(old[5, 0]) / abs(old[1, 0])
    check("pooled scaling puts equal-spread works on one scale",
          new_ratio < 2.0 and old_ratio > 4.0,
          f"four-perf work vs two-perf work: old {abs(old[1,0]):.2f} vs "
          f"{abs(old[5,0]):.2f} (ratio {old_ratio:.1f}x, clipped from "
          f"{1000/abs(old[1,0]):.0f}x), pooled {abs(Xn[1,0]):.2f} vs "
          f"{abs(Xn[5,0]):.2f} (ratio {new_ratio:.2f}x)")


# ---------------------------------------------------------------- test 4
def test_pooled_converges_on_per_work() -> None:
    """Answer known in advance: the two must converge as works get larger.

    Every work drawn from one distribution, so a per-work divisor and a pooled
    one estimate the same quantity and differ only in how well. With three
    performances per work the per-work standard deviation has two degrees of
    freedom and is mostly noise; with sixty it is accurate, and the two
    normalisations must agree almost exactly. If they do not converge, the
    pooled path is doing something other than scaling.

    Worth knowing in its own right: at the three-or-four performances per work
    this project actually has, the two disagree substantially -- so running the
    strict pool under both is a real comparison and not a formality.
    """
    rows = []
    for n in (3, 6, 20, 60):
        rng = np.random.default_rng(11)
        X, works = [], []
        for w in range(12):
            for _ in range(n):
                X.append(rng.normal(size=5))
                works.append(f"w{w}")
        X, works = np.array(X), np.array(works)
        y = np.array([f"p{i % 7}" for i in range(len(X))])
        a, _ = loo_normalise(X, y, works, scale="per_work")
        b, _ = loo_normalise(X, y, works, scale="pooled")
        rows.append((n, float(np.corrcoef(a.ravel(), b.ravel())[0, 1])))
    detail = ", ".join(f"n={n}: r={r:.3f}" for n, r in rows)
    check("pooled converges on per-work as works grow",
          rows[-1][1] > 0.97 and rows[-1][1] > rows[0][1], detail)


# ---------------------------------------------------------------- test 5
def test_pooled_is_leave_one_out() -> None:
    """Answer known in advance: a row must not help set its own divisor.

    Give one row a wildly outlying deviation. Its own divisor is pooled over
    every other row, so it must come out large; the divisor applied to the
    quiet rows is pooled over a set that includes the outlier, so those come
    out small. If the outlier helped scale itself it would be dragged back
    toward the others.
    """
    X = np.zeros((20, 1))
    X[1, 0] = 1.0
    X[3, 0] = 1.0
    X[5, 0] = 80.0
    works = np.array([f"w{i // 2}" for i in range(20)])
    y = np.array([f"p{i % 3}" for i in range(20)])
    Xn, _ = loo_normalise(X, y, works, min_per_work=2, scale="pooled")
    check("pooled divisor excludes the row being normalised",
          abs(Xn[5, 0]) > 5 * abs(Xn[1, 0]),
          f"outlier {Xn[5,0]:+.2f} vs ordinary row {Xn[1,0]:+.2f}")


# ---------------------------------------------------------------- test 6
def test_oof_matches_leave_one_out() -> None:
    """Answer known in advance: identical, since it is the same loop.

    Every paired and resampled test below scores `oof_predictions` instead of
    calling `leave_one_out`, so if the two ever disagreed the whole set of them
    would be describing a different estimator from the headline.
    """
    rng = np.random.default_rng(3)
    y = np.array([f"p{i % 6}" for i in range(48)])
    g = np.array([f"s{i % 16}" for i in range(48)])
    X = np.array([[hash(lab) % 7, i % 5] for i, lab in enumerate(y)], dtype=float)
    X += rng.normal(scale=0.3, size=X.shape)
    from sklearn.metrics import balanced_accuracy_score
    pred = oof_predictions(X, y, g)
    ok = np.array([p is not None for p in pred])
    a = balanced_accuracy_score(y[ok], np.array(list(pred[ok])))
    b = leave_one_out(X, y, g)[0]
    check("oof_predictions scores exactly what leave_one_out reports",
          np.isclose(a, b), f"{a:.6f} vs {b:.6f}")


# ---------------------------------------------------------------- test 7
def test_paired_permutation_on_identical_arms() -> None:
    """Answer known in advance: difference exactly 0 and p = 1.

    Hand the paired test the same predictions twice. Any other answer means it
    is finding structure in its own shuffling.
    """
    y = np.array([f"p{i % 5}" for i in range(40)])
    g = np.array([f"s{i % 13}" for i in range(40)])
    pred = np.array([f"p{(i + 1) % 5}" for i in range(40)], dtype=object)
    obs, null, p = paired_permutation(y, pred, pred.copy(), g, n_perm=200)
    check("paired test on two identical arms gives zero and p = 1",
          obs == 0.0 and p == 1.0 and np.allclose(null, 0.0),
          f"observed {obs:+.4f}, p = {p:.4f}, null max |x| {np.abs(null).max():.1e}")


# ---------------------------------------------------------------- test 8
def test_paired_permutation_finds_a_planted_gap() -> None:
    """Answer known in advance: a large planted gap must come back significant.

    One arm is right about everything, the other wrong about everything. If the
    test cannot see that, it cannot see anything.
    """
    y = np.array([f"p{i % 5}" for i in range(40)])
    g = np.array([f"s{i % 20}" for i in range(40)])
    good = y.astype(object)
    bad = np.array([f"p{(i + 1) % 5}" for i in range(40)], dtype=object)
    obs, _, p = paired_permutation(y, good, bad, g, n_perm=500)
    check("paired test finds a planted perfect-vs-wrong gap",
          np.isclose(obs, 1.0) and p < 0.01,
          f"observed {obs:+.3f}, p = {p:.4f}")


# ---------------------------------------------------------------- test 9
def test_bootstrap_ci_degenerate_cases() -> None:
    """Answer known in advance: perfect predictions give the interval [1, 1].

    A resampling interval that wanders away from 1.0 when every draw scores
    1.0 is resampling something other than what it reports.
    """
    y = np.array([f"p{i % 4}" for i in range(32)])
    g = np.array([f"s{i % 8}" for i in range(32)])
    lo, hi, draws = bootstrap_ci(y, y.astype(object), g, n_boot=300)
    check("bootstrap interval on perfect predictions is [1, 1]",
          lo == 1.0 and hi == 1.0, f"[{lo:.3f}, {hi:.3f}]")


# --------------------------------------------------------------- test 10
def test_bootstrap_keeps_every_class() -> None:
    """Answer known in advance: all classes present in all draws.

    This is the reason the bootstrap is stratified at all. An unstratified
    resample of sessions drops every session of a two-session performer about
    13% of the time, and balanced accuracy over 22 classes is not the statistic
    the report is about. Checked by giving a label the bare minimum of two
    sessions and confirming the interval stays on the 4-class scale: a draw
    that lost a class entirely would score the remaining three and the floor
    of the interval would jump.
    """
    y = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 6 + ["d"] * 6)
    g = np.array([f"{lab}{i % 2}" for i, lab in enumerate(y)])
    pred = np.array(["a"] * 24, dtype=object)  # always the majority class
    lo, hi, draws = bootstrap_ci(y, pred, g, n_boot=500)
    want = 1.0 / 4
    check("stratified bootstrap keeps all four classes in every draw",
          np.allclose(draws, want),
          f"always-predict-one-class scores {want:.4f}; draws span "
          f"[{draws.min():.4f}, {draws.max():.4f}]")


# --------------------------------------------------------------- test 11
def test_retained_ci_recovers_a_known_ratio() -> None:
    """Answer known in advance: two arms built to retain exactly half.

    Four classes, chance 0.25. The exact arm is perfect (1.00, margin 0.75);
    the transcribed arm is built to score 0.625 (margin 0.375), which is half
    the margin. The interval must contain 0.5 and be tight, since both arms are
    deterministic given the draw.
    """
    y = np.array([f"p{i % 4}" for i in range(64)])
    g = np.array([f"s{i % 16}" for i in range(64)])
    exact = y.astype(object)
    trans = y.astype(object).copy()
    # Corrupt exactly half the rows of every session, so each class scores
    # recall 0.500 no matter which sessions a draw happens to pick. The planted
    # ratio is then (0.500 - 0.250) / (1.000 - 0.250) = 1/3 exactly, and a
    # correct interval has to be a point rather than merely contain it.
    for s in sorted(set(g)):
        idx = np.where(g == s)[0]
        for i in idx[: len(idx) // 2]:
            trans[i] = "p0" if y[i] != "p0" else "p1"
    from sklearn.metrics import balanced_accuracy_score
    got = balanced_accuracy_score(y, trans)
    lo, hi, bad, vals = retained_ci(y, exact, trans, g, 0.25, n_boot=400)
    ratio = (got - 0.25) / (1.0 - 0.25)
    check("retained-fraction interval recovers a planted ratio exactly",
          np.isclose(ratio, 1 / 3) and np.isclose(lo, 1 / 3)
          and np.isclose(hi, 1 / 3) and bad == 0,
          f"planted {ratio:.4f}, interval [{lo:.4f}, {hi:.4f}], "
          f"{bad} undefined draws")


# --------------------------------------------------------------- test 12
def test_group_permutation_preserves_nesting() -> None:
    """Answer known in advance: every row of a group keeps one label.

    The permutation null is only a null for this question if it shuffles whole
    groups. A row-level shuffle would break the fact that a session holds one
    performer, which is the structure the grouped estimator is built around.
    """
    rng = np.random.default_rng(5)
    y = np.array([f"p{i % 5}" for i in range(50)])
    g = np.array([f"s{i % 10}" for i in range(50)])
    z = permute_by_group(y, g, rng)
    intact = all(len(set(z[g == s])) == 1 for s in set(g))
    same_multiset = sorted(z.tolist()) == sorted(y.tolist())
    check("group permutation keeps one label per group",
          intact and same_multiset,
          f"groups intact {intact}, label counts preserved {same_multiset}")


def main() -> int:
    print("=" * 78)
    print("KNOWN-ANSWER TESTS - pianist_id/evaluate.py")
    print("=" * 78)
    for fn in (test_per_work_unchanged,
               test_per_work_refuses_two,
               test_pooled_puts_every_row_on_one_scale,
               test_pooled_converges_on_per_work,
               test_pooled_is_leave_one_out,
               test_oof_matches_leave_one_out,
               test_paired_permutation_on_identical_arms,
               test_paired_permutation_finds_a_planted_gap,
               test_bootstrap_ci_degenerate_cases,
               test_bootstrap_keeps_every_class,
               test_retained_ci_recovers_a_known_ratio,
               test_group_permutation_preserves_nesting):
        fn()
    print("=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("12 tests, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
