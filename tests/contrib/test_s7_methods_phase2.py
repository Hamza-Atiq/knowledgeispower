"""S7 hardening: imputation, propagation, selection, phase-2 helpers and the benchmark's
significance tolerance.

Edge cases from docs/specs/S7_hardening.md and the S7 row of docs/SLOTS.md: empty graphs,
all-NaN proteins, k > number of candidates, small n for ``phase2.data_graph`` and identical
predictions vs ``SIG_TOL``. Data are small seeded synthetic matrices; expected values come from
the documented contract or from another core function, never from a previous run.
"""

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from scp_graphfs_mcp import phase2
from scp_graphfs_mcp.datasets import from_log2_matrix
from scp_graphfs_mcp.ids import hgnc_table
from scp_graphfs_mcp.methods import (candidates, impute_with_prior, knn_fill, propagate, select_features,
                                     select_random)
from scp_graphfs_mcp.registry import validate_prior

ROOT = Path(__file__).resolve().parents[2]


def _graph(nodes, edges, weight=1.0):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges, weight=weight)
    g.graph["provenance"] = {"source": "s7-toy", "version": "0", "url": "-", "sha256": "-"}
    return g


def _matrix(n_cells=40, n_prot=6, miss=0.15, seed=0):
    """P0..P2 share a latent abundance (a toy 'complex'); the rest are independent."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n_cells)
    cols = [10 + latent + rng.normal(0, 0.2, n_cells) for _ in range(3)]
    cols += [rng.normal(8, 1, n_cells) for _ in range(n_prot - 3)]
    X = np.column_stack(cols)
    X[rng.random(X.shape) < miss] = np.nan
    return X


def _adata(X):
    return from_log2_matrix(X, [f"c{i}" for i in range(X.shape[0])], [f"P{j}" for j in range(X.shape[1])])


COMPLEX = [("P0", "P1"), ("P1", "P2"), ("P0", "P2")]


# --------------------------------------------------------------------------- impute_with_prior


def test_edgeless_graph_is_pure_knn_and_only_nan_changes():
    a = _adata(_matrix())
    X = np.asarray(a.X)
    miss = np.isnan(X)
    out = impute_with_prior(a, _graph(list(a.var_names), []))
    Y = np.asarray(out.X)
    assert np.array_equal(Y[~miss], X[~miss])
    assert np.allclose(Y[miss], knn_fill(a)[miss])
    rep = out.uns["impute_with_prior"]
    assert rep["n_imputed_by_prior"] == 0 and rep["n_imputed_by_knn_fallback"] == int(miss.sum())
    assert np.array_equal(np.asarray(out.layers["was_imputed"]), miss)


@pytest.mark.parametrize("graph", [_graph([], []), _graph(["NOT_A_FEATURE", "é"], [("NOT_A_FEATURE", "é")])],
                         ids=["node_less", "foreign_nodes"])
def test_graph_without_dataset_nodes_falls_back_to_knn(graph):
    a = _adata(_matrix())
    miss = np.isnan(np.asarray(a.X))
    out = impute_with_prior(a, graph)
    assert np.allclose(np.asarray(out.X)[miss], knn_fill(a)[miss])
    assert out.uns["impute_with_prior"]["n_imputed_by_prior"] == 0


def test_imputation_is_deterministic_and_never_zero_fills():
    a = _adata(_matrix())
    g = _graph(list(a.var_names), COMPLEX)
    y1, y2 = np.asarray(impute_with_prior(a, g).X), np.asarray(impute_with_prior(a, g).X)
    assert np.array_equal(y1, y2)
    miss = np.isnan(np.asarray(a.X))
    assert not np.isnan(y1).any() and not (y1[miss] == 0).any()


def test_all_nan_protein_without_partner_stays_nan():
    X = _matrix()
    X[:, 4] = np.nan
    a = _adata(X)
    out = impute_with_prior(a, _graph(list(a.var_names), COMPLEX))  # P4 has no partner
    Y = np.asarray(out.X)
    assert np.isnan(Y[:, 4]).all()  # kNN fallback documents: a never-observed protein cannot be imputed
    obs = ~np.isnan(X)
    assert np.array_equal(Y[obs], X[obs])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="core bug: methods.zscore_stats uses mean 0 / sd 1 for an all-NaN protein, "
                                       "so impute_with_prior fills it with ~0 (log2) from partners; reported to Amit")
def test_all_nan_protein_with_observed_partner_is_not_invented():
    X = _matrix()
    X[:, 4] = np.nan
    a = _adata(X)
    out = impute_with_prior(a, _graph(list(a.var_names), COMPLEX + [("P4", "P0")]))
    assert np.isnan(np.asarray(out.X)[:, 4]).all()  # same contract as the kNN fallback


# --------------------------------------------------------------------------- propagate


def test_propagate_on_edgeless_graph_returns_normalised_seed():
    names = list("abcd")
    p = propagate(_graph(names, []), {"a": 3.0, "c": 1.0}, names)
    assert np.allclose(p, [0.75, 0.0, 0.25, 0.0])  # dangling nodes keep their own mass
    q = propagate(_graph(names, []), {}, names)
    assert np.allclose(q, 0.25)  # all-zero scores -> uniform start, documented in code


@pytest.mark.xfail(strict=True, raises=ZeroDivisionError, reason="core bug: methods.propagate divides by len(s)=0 on a node-less graph "
                                       "(ZeroDivisionError); reported to Amit")
def test_propagate_on_node_less_graph_returns_empty():
    assert propagate(_graph([], []), {}).shape == (0,)


# --------------------------------------------------------------------------- select_features / select_random


@pytest.fixture
def sel_data():
    X = _matrix(n_cells=30, n_prot=8, miss=0.0)
    X[:12, 5] = np.nan  # 60% detection
    X[:, 6] = np.nan    # all-NaN protein
    return _adata(X)


@pytest.mark.parametrize("mode", ["none", "soft_weight"])
def test_k_larger_than_candidates_returns_all_candidates(sel_data, mode):
    a = sel_data
    cand = set(np.array(a.var_names)[candidates(a)])
    assert 0 < len(cand) < a.n_vars
    g = _graph(list(a.var_names), COMPLEX) if mode == "soft_weight" else None
    panel = select_features(a, g, a.n_vars + 5, mode)
    assert len(panel) == len(set(panel)) == len(cand) and set(panel) == cand
    assert select_features(a, g, 0, mode) == []


def test_soft_weight_on_edgeless_graph_equals_hvp(sel_data):
    a = sel_data
    for k in (1, 3, 5):
        assert select_features(a, _graph(list(a.var_names), []), k, "soft_weight") == select_features(a, None, k, "none")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="core bug: methods._top_k slices order[:k], so k<0 "
                                       "returns all but the last |k| ranked candidates instead of none; reported to Amit")
def test_negative_k_returns_no_proteins(sel_data):
    assert select_features(sel_data, None, -1, "none") == []


def test_select_random_edge_cases(sel_data):
    a = sel_data
    cand = set(np.array(a.var_names)[candidates(a)])
    assert select_random(a, 3, 42) == select_random(a, 3, 42)
    big = select_random(a, a.n_vars + 5, 0)
    assert len(big) == len(set(big)) == len(cand) and set(big) == cand
    assert select_random(a, 3, 0, min_detection=1.01) == []  # zero candidates
    assert select_features(a, None, 3, "none", min_detection=1.01) == []


# --------------------------------------------------------------------------- phase2.data_graph (small n)


def _planted_pair(n_cells, n_prot=5, miss=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_cells, n_prot))
    X[:, 1] = X[:, 0] + rng.normal(0, 0.05, n_cells)  # the one strongly correlated pair
    if miss:
        X[rng.random(X.shape) < miss] = np.nan
    return X


@pytest.mark.parametrize("n_cells", [6, 10, 20, 29])
def test_data_graph_small_n_finds_planted_pair(n_cells):
    """Regression for the fixed overlap bug: below 30 cells the threshold must still be reachable."""
    names = [f"P{j}" for j in range(5)]
    g = phase2.data_graph(_planted_pair(n_cells), names, 1)
    assert {frozenset(e) for e in g.edges} == {frozenset(("P0", "P1"))}


def test_data_graph_small_n_with_missing_values():
    names = [f"P{j}" for j in range(5)]
    g = phase2.data_graph(_planted_pair(24, miss=0.15, seed=3), names, 1)
    assert {frozenset(e) for e in g.edges} == {frozenset(("P0", "P1"))}


def test_data_graph_invariants():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (40, 5))
    X[:, 1] = X[:, 0] + rng.normal(0, 0.1, 40)
    X[:, 2] = -X[:, 0] + rng.normal(0, 0.1, 40)  # strongly *negative* pair must never be an edge
    names = [f"P{j}" for j in range(5)]
    n_pairs = 10
    g = phase2.data_graph(X, names, n_pairs - 1)
    assert set(g.nodes) == set(names) and nx.number_of_selfloops(g) == 0
    assert not g.has_edge("P0", "P2")
    r = np.corrcoef(X, rowvar=False)
    for u, v, w in g.edges(data="weight"):
        assert w == 1.0 and r[names.index(u), names.index(v)] > 0
    validate_prior(g, names)


@pytest.mark.xfail(strict=True, raises=ValueError, reason="core bug: phase2.data_graph calls np.argpartition(-vals, n_edges), which "
                                       "raises when n_edges >= number of protein pairs; reported to Amit")
@pytest.mark.parametrize("n_edges", [3, 5])
def test_data_graph_n_edges_at_or_above_number_of_pairs(n_edges):
    rng = np.random.default_rng(2)
    latent = rng.normal(0, 1, 30)
    X = np.column_stack([latent + rng.normal(0, 0.1, 30) for _ in range(3)])  # 3 pairs, all positive
    g = phase2.data_graph(X, ["A", "B", "C"], n_edges)
    assert g.number_of_edges() == 3


def test_data_graph_n_edges_just_below_number_of_pairs():
    rng = np.random.default_rng(2)
    latent = rng.normal(0, 1, 30)
    X = np.column_stack([latent + rng.normal(0, 0.1, 30) for _ in range(3)])
    assert phase2.data_graph(X, ["A", "B", "C"], 2).number_of_edges() == 2
    assert phase2.data_graph(X, ["A", "B", "C"], 0).number_of_edges() == 0


# --------------------------------------------------------------------------- union_graph


def test_union_graph_edge_cases():
    names = [f"P{j}" for j in range(4)]
    a = _graph(names, [("P0", "P1"), ("P2", "P3")], weight=2.0)
    empty = _graph(names, [])
    u = phase2.union_graph(a, empty)
    assert {frozenset(e): w for *e, w in u.edges(data="weight")} == {frozenset(e): w for *e, w in a.edges(data="weight")}
    d = phase2.union_graph(a, a)
    assert all(w == 4.0 for *_, w in d.edges(data="weight"))
    e = phase2.union_graph(empty, empty)
    assert e.number_of_edges() == 0 and set(e.nodes) == set(names)
    for g in (u, d, e):
        validate_prior(g, names)


# --------------------------------------------------------------------------- residual_on_knn / grmf


def test_residual_on_knn_edgeless_graph_equals_loo_knn():
    X = _matrix()
    names = [f"P{j}" for j in range(X.shape[1])]
    out, parts = phase2.residual_on_knn(X, _graph(names, []), names, return_parts=True)
    assert parts["beta"] == 0.0
    K = phase2.loo_knn(X)
    miss = np.isnan(X)
    assert np.array_equal(out[~miss], X[~miss]) and np.allclose(out[miss], K[miss])


def test_residual_on_knn_observed_unchanged_and_all_nan_not_zero_filled():
    X = _matrix()
    X[:, 4] = np.nan
    names = [f"P{j}" for j in range(X.shape[1])]
    with np.errstate(all="ignore"):
        out = phase2.residual_on_knn(X, _graph(names, COMPLEX + [("P4", "P0")]), names)
    obs = ~np.isnan(X)
    assert np.array_equal(out[obs], X[obs])
    assert not (out[:, 4] == 0).any()  # never filled with 0 (no data -> no value)


def test_grmf_tiny_matrix_rank_above_size():
    rng = np.random.default_rng(4)
    X = rng.normal(10, 1, (3, 4))
    X[0, 1] = X[2, 3] = np.nan
    names = list("abcd")
    kw = dict(rank=10, iters=60)
    r1, r2 = phase2.grmf(X, None, names, **kw), phase2.grmf(X, None, names, **kw)
    assert np.array_equal(r1, r2)  # fixed seed
    obs = ~np.isnan(X)
    assert np.array_equal(r1[obs], X[obs]) and np.isfinite(r1).all()
    assert np.array_equal(phase2.grmf(X, _graph(names, []), names, lam_g=1.0, **kw), r1)  # empty graph == no graph
    assert np.array_equal(phase2.grmf(X, _graph(names, [("a", "b")]), names, lam_g=0.0, **kw), r1)
    assert phase2.grmf(X, None, names, full=True, **kw).shape == X.shape
    assert phase2.grmf(X, None, names, seed=1, **kw).shape == X.shape


# --------------------------------------------------------------------------- keratin_mask / select_coverage


@pytest.fixture(scope="module")
def keratin_acc():
    h = hgnc_table()
    kr = h[h["gene_group"].str.contains("Keratin", case=False)]
    accs = sorted({a for ids in kr["uniprot_ids"] for a in ids.split("|") if a})
    assert accs, "HGNC snapshot lists no keratin"
    assert "P28074" not in accs  # PSMB5 is not a keratin (checked against the snapshot)
    return accs[0]


def test_keratin_mask_handles_isoforms_groups_and_non_ascii(keratin_acc):
    ids = [keratin_acc, f"{keratin_acc}-2", f"P28074;{keratin_acc}", "P28074", "é", ""]
    m = phase2.keratin_mask(ids)
    assert m.tolist() == [True, True, True, False, False, False]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="core bug: phase2.keratin_mask([]) returns a float64 array, so "
                                       "select_coverage(exclude=...) fails on '~'; reported to Amit")
def test_keratin_mask_empty_is_bool():
    m = phase2.keratin_mask([])
    assert m.dtype == bool and m.shape == (0,)
    assert phase2.select_coverage(np.zeros((3, 0)), _graph([], []), [], 2, exclude=m) == []


def _coverage_data():
    rng = np.random.default_rng(5)
    X = rng.normal(10, rng.uniform(0.5, 2.0, 7), (20, 7))
    for j, n_missing in enumerate([0, 1, 2, 3, 0, 10, 20]):  # P5: 50% detected, P6: never
        X[:n_missing, j] = np.nan
    return X, [f"P{j}" for j in range(7)]


def test_select_coverage_edgeless_orders_by_detection_then_variance():
    X, names = _coverage_data()
    det = (~np.isnan(X)).mean(0)
    with np.errstate(all="ignore"):
        var = np.nan_to_num(np.nanvar(X, 0))
    cand = [i for i in range(7) if det[i] >= 0.8]
    expected = [names[i] for i in sorted(cand, key=lambda i: (det[i], var[i]), reverse=True)]
    with np.errstate(all="ignore"):
        got = phase2.select_coverage(X, _graph(names, []), names, k=len(names) + 3)
    assert got == expected  # k > candidates -> every candidate exactly once
    assert phase2.select_coverage(X, _graph(names, []), names, k=0) == []


def test_select_coverage_respects_exclude_and_star_hub():
    X, names = _coverage_data()
    star = _graph(names, [("P3", n) for n in names if n != "P3"])
    with np.errstate(all="ignore"):
        assert phase2.select_coverage(X, star, names, k=1) == ["P3"]  # hub covers the most detection mass
        excl = np.array([n == "P3" for n in names])
        got = phase2.select_coverage(X, star, names, k=10, exclude=excl)
    assert "P3" not in got and len(got) == len(set(got))


# --------------------------------------------------------------------------- SIG_TOL: identical predictions


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("_s7_bench_run", ROOT / "benchmarks" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # import only: defines functions, runs nothing
    return mod


def _significant(bench, c):
    return c["ci95"][1] < -bench.SIG_TOL  # exactly the rule used in benchmarks/run.py


def test_identical_predictions_never_significant(bench):
    rng = np.random.default_rng(6)
    se = rng.gamma(1.0, 1.0, 500)
    c = bench.boot_rmse_diff(se, [se.copy()], 200, seed=17)
    assert c["delta_rmse"] == 0.0 and c["ci95"] == [0.0, 0.0] and not _significant(bench, c)
    noisy = se * (1 + 1e-15)  # float noise, as from a different summation order
    assert not _significant(bench, bench.boot_rmse_diff(se, [noisy, se], 200, seed=17))


def test_real_improvement_is_significant(bench):
    rng = np.random.default_rng(6)
    se = rng.gamma(1.0, 1.0, 500)
    assert _significant(bench, bench.boot_rmse_diff(0.5 * se, [se], 200, seed=17))


def test_knn_fallback_on_both_sides_is_not_significant(bench):
    """The scenario named next to SIG_TOL: a prior with no usable partners predicts exactly like kNN."""
    full = _matrix(miss=0.0)
    mask = np.random.default_rng(7).random(full.shape) < 0.1  # held-out entries with known truth
    Xm = full.copy()
    Xm[mask] = np.nan
    a = _adata(Xm)
    knn = knn_fill(a)
    prior = np.asarray(impute_with_prior(a, _graph(list(a.var_names), []), fallback=knn).X)
    se_p, se_k = (prior[mask] - full[mask]) ** 2, (knn[mask] - full[mask]) ** 2
    assert np.array_equal(se_p, se_k)
    assert not _significant(bench, bench.boot_rmse_diff(se_p, [se_k], 200, seed=17))
