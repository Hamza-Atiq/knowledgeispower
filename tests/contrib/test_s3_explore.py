"""S3 dev-study logic on tiny synthetic arrays (no real-data run): pass rule, verdicts, guards."""
import numpy as np
import pytest

from contrib.s3_prior import explore_s3 as E

rng = np.random.default_rng(0)


def _se(n, scale):
    return (rng.normal(0, 1, n) * scale) ** 2


def test_pass_needs_ci_and_one_percent():
    base = _se(4000, 1.0)
    big = base * 0.81          # RMSE 10% lower: CI excludes 0 and >= 1%
    tiny = base * 0.995        # RMSE 0.25% lower: CI may exclude 0, but below the 1% SESOI
    assert E.compare(big, [base])["pass"] is True
    c = E.compare(tiny, [base])
    assert c["ci95"][1] < 0 and c["pct_of_ref_rmse"] > -1.0 and c["pass"] is False
    assert E.compare(base, [base])["pass"] is False          # identical: never a pass (SIG_TOL)
    assert E.compare(base, [big])["pass"] is False            # worse: never a pass


def test_null_reference_is_mean_over_nulls():
    base = _se(2000, 1.0)
    nulls = [base * f for f in (1.0, 1.1, 1.2, 1.3, 1.4)]
    c = E.compare(base, nulls)
    ref = np.mean([np.sqrt(x.mean()) for x in nulls])
    assert c["ref_rmse"] == pytest.approx(ref)
    assert c["delta_rmse"] == pytest.approx(np.sqrt(base.mean()) - ref)


def test_empty_selection_is_not_nan_and_never_passes():
    c = E.compare(np.array([]), [np.array([])])
    assert c["pass"] is False and c["n_entries"] == 0 and c["note"] == "no entries"
    assert all(v is None for v in c["ci95"])


def _check(target, mask, vs, ok):
    return {"prior": "humap3_high", "method": "E1", "regime": "low", "stratum": "all",
            "target": target, "mask": mask, "vs": vs, "pass": ok}


def _thirty(ok=True):
    return [_check(t, m, v, ok) for t in E.TARGET_NAMES for m in ("MCAR", "MNAR") for v in E.COMPARATORS]


def test_primary_verdict_all_thirty_must_pass():
    assert E.primary_verdict(_thirty())["verdict"] == "qualifies (dev)"
    one_fail = _thirty()
    one_fail[7]["pass"] = False
    v = E.primary_verdict(one_fail)
    assert v["verdict"] == "does not qualify" and v["n_pass"] == 29 and v["n_checks"] == 30


def test_primary_verdict_ignores_secondaries_and_rejects_wrong_count():
    extra = _thirty() + [{**_check("petrosius2023", "MCAR", "knn", False), "method": "E6"}]
    assert E.primary_verdict(extra)["verdict"] == "qualifies (dev)"
    with pytest.raises(ValueError, match="30"):
        E.primary_verdict(_thirty()[:-1])


def test_sign_check():
    assert E.same_sign([-0.1, -0.2, -0.01]) is True
    assert E.same_sign([-0.1, 0.2, -0.01]) is False
    assert E.same_sign([0.0, -0.1, -0.1]) is False


@pytest.mark.parametrize("mcar,mnar,expected", [
    ("curated", "curated", "curated better"),
    ("humap", "humap", "hu.MAP better"),
    ("curated", "humap", "inconclusive"),
    ("none", "none", "inconclusive"),
    ("empty", "curated", "inconclusive: no matched entries"),
])
def test_matched_verdict(mcar, mnar, expected):
    def res(kind):
        if kind == "empty":
            return {"n_matched": 0, "curated_minus_humap": E.compare(np.array([]), [np.array([])]),
                    "humap_minus_curated": E.compare(np.array([]), [np.array([])])}
        a, b = _se(3000, 1.0), _se(3000, 1.0)
        good = a * 0.5
        cur, hm = {"curated": (good, a), "humap": (a, good), "none": (a, b)}[kind]
        return {"n_matched": 3000, "curated_minus_humap": E.compare(cur, [hm]),
                "humap_minus_curated": E.compare(hm, [cur])}
    assert E.matched_verdict({"MCAR": res(mcar), "MNAR": res(mnar)}) == expected


def test_matched_verdict_requires_strict_significance():
    a = _se(3000, 1.0)
    r = {"n_matched": 3000, "curated_minus_humap": E.compare(a, [a]), "humap_minus_curated": E.compare(a, [a])}
    assert E.matched_verdict({"MCAR": r, "MNAR": r}) == "inconclusive"


def test_quick_requires_explicit_out():
    with pytest.raises(SystemExit, match="--out"):
        E.parse_args(["--quick"])


def test_refuses_dirty_slot_folder(monkeypatch):
    monkeypatch.setattr(E, "_git", lambda *a: " M contrib/s3_prior/humap.py\n" if a[0] == "status" else "abc\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        E.require_clean()
    monkeypatch.setattr(E, "_git", lambda *a: "" if a[0] == "status" else "abc123\n")
    assert E.require_clean() == "abc123"


def test_subsample_matches_explore_ext_rule():
    ct = np.array(["a"] * 10 + ["b"] * 12 + ["c"] * 9 + ["d"] * 11)
    idx = E.subsample(ct, 28, 100)
    rng2 = np.random.default_rng(100)
    exp = np.sort(np.concatenate([rng2.choice(np.where(ct == c)[0], 7, replace=False) for c in sorted(set(ct))]))
    assert np.array_equal(idx, exp)
    assert np.array_equal(E.subsample(ct, 0, 100), np.arange(len(ct)))
