"""S7 hardening: every core MCP tool with non-ASCII, empty and wrong-type input.

Contract (src/scp_graphfs_mcp/tools.py docstring; AGENTS.md "How to add an MCP tool"): each tool
returns a JSON-serialisable dict, and bad input returns ``{"error": {...}}`` rather than raising.
Cases that currently raise are kept as strict xfails (core bugs, reported to Amit). Note: the MCP
server wraps tool calls in ``try/except`` (server.py ``on_call_tool``), so a raising tool does not
crash the server; these tests check the tool-level contract.
"""

import json

import numpy as np
import pytest

from scp_graphfs_mcp import tools as T
from scp_graphfs_mcp.complexes import load_complexes
from scp_graphfs_mcp.datasets import from_log2_matrix
from scp_graphfs_mcp.methods import candidates
from scp_graphfs_mcp.registry import TOOLS

NON_ASCII = ["é", "\U0001f9ec", "蛋白质", "PSMB5́", "РSMB5", "مرحبا",
             "P2807４", "\x00", "CORUM:é", "CPX-é"]
TESTED_TOOLS = {"get_complexes", "explain_protein", "build_prior", "propagate_scores", "impute_with_prior",
                "select_panel"}


def _bug(exc, why):
    return pytest.mark.xfail(strict=True, raises=exc, reason=f"core bug: {why}; reported to Amit")


def _is_error(out):
    assert isinstance(out, dict)
    json.dumps(out)  # JSON-serialisable without a default= hook
    return "error" in out and {"code", "message"} <= set(out["error"])


def _ok(out):
    assert isinstance(out, dict)
    json.dumps(out)
    return "error" not in out


def test_every_core_tool_is_covered_here():
    core = {n for n, t in TOOLS.items() if not t.contrib}
    assert core == TESTED_TOOLS


# --------------------------------------------------------------------------- lookup tools


@pytest.mark.parametrize("bad", NON_ASCII + ["", "   ", None, 123, 1.5, ["P28074"], {"id": "P28074"}])
@pytest.mark.parametrize("tool", [T.get_complexes, T.explain_protein], ids=["get_complexes", "explain_protein"])
def test_lookup_tools_bad_or_non_ascii_id_returns_error(tool, bad):
    assert _is_error(tool(bad))


@pytest.mark.parametrize("sources", [
    pytest.param(["bogus"], marks=_bug(ValueError, "tools.get_complexes passes unknown sources to load_complexes, "
                                                   "which raises ValueError (tools.py:34)")),
    pytest.param(["é"], marks=_bug(ValueError, "tools.get_complexes raises on a non-ASCII source name")),
    pytest.param("corum", marks=_bug(ValueError, "tools.get_complexes turns a str sources into a tuple of "
                                                 "characters and raises")),
])
def test_get_complexes_bad_sources_returns_error(sources):
    assert _is_error(T.get_complexes("P28074", sources))


# --------------------------------------------------------------------------- build_prior / propagate_scores


def test_build_prior_non_ascii_proteins_are_reported_unmapped():
    out = T.build_prior(NON_ASCII + ["P28074"], ["corum", "complexportal"])
    assert _ok(out)
    for rep in out["priors"].values():
        assert rep["n_proteins"] == len(NON_ASCII) + 1
        assert set(rep["unmapped"]) == set(NON_ASCII)
        assert rep["unmapped_fraction"] == pytest.approx(len(NON_ASCII) / (len(NON_ASCII) + 1))


@pytest.mark.parametrize("args", [
    ([],), ("P28074",), (None,), ([1, 2],), (["P28074", None],),
    (["P28074"], ["é"]), (["P28074"], ["bogus"]), (["P28074"], [None]),
    (["P28074"], ["string"], 100),
    pytest.param((["P28074"], ["string"], "abc"),
                 marks=_bug(TypeError, "tools.build_prior catches only ValueError; a non-int string_min_score "
                                       "raises TypeError in priors.build_string_graph (priors.py:105)")),
    pytest.param((["P28074"], ["string"], None),
                 marks=_bug(TypeError, "tools.build_prior raises TypeError for string_min_score=None")),
])
def test_build_prior_bad_input_returns_error(args):
    assert _is_error(T.build_prior(*args))


@pytest.fixture(scope="module")
def handle():
    out = T.build_prior(["P28074", "P20618", "é"], ["corum"])
    return out["priors"]["corum"]["handle"]


def test_propagate_non_ascii_seed_and_nodes(handle):
    out = T.propagate_scores(handle, {"é": 1.0, "\U0001f9ec": 2.0})
    assert _ok(out)
    top = {r["protein"]: r for r in out["top"]}
    assert top["é"]["seed_score"] == 1.0 and "\U0001f9ec" not in top  # not a graph node
    assert sum(r["propagated"] for r in out["top"]) == pytest.approx(1.0)


@pytest.mark.parametrize("args", [
    ("é", {"P28074": 1.0}), ("", {"P28074": 1.0}), (None, {"P28074": 1.0}), (123, {"P28074": 1.0}),
    pytest.param(([], {"P28074": 1.0}), marks=_bug(TypeError, "tools.propagate_scores looks up an unhashable "
                                                              "handle in a dict (tools.py:175)")),
])
def test_propagate_bad_handle_returns_error(args):
    assert _is_error(T.propagate_scores(*args))


@pytest.mark.parametrize("scores, extra", [
    ({}, {}), ("P28074", {}), (None, {}), ({"P28074": -1.0}, {}), ({"P28074": float("nan")}, {}),
    ({"P28074": "x"}, {}), ({"P28074": 1.0}, {"alpha": 0.0}), ({"P28074": 1.0}, {"alpha": "x"}),
    pytest.param({"P28074": None}, {}, marks=_bug(TypeError, "tools.propagate_scores catches only ValueError; "
                                                             "a None score raises TypeError (tools.py:181-182)")),
    pytest.param({"P28074": 1.0}, {"alpha": None}, marks=_bug(TypeError, "tools.propagate_scores raises TypeError "
                                                                         "on alpha=None")),
    pytest.param({"P28074": 1.0}, {"top": "x"}, marks=_bug(ValueError, "tools.propagate_scores calls int(top) "
                                                                       "outside the try block (tools.py:185)")),
])
def test_propagate_bad_scores_or_params_return_error(handle, scores, extra):
    assert _is_error(T.propagate_scores(handle, scores, **extra))


# --------------------------------------------------------------------------- file tools


@pytest.fixture(scope="module")
def h5ad(tmp_path_factory):
    """Small real-accession AnnData written under a non-ASCII file name."""
    df = load_complexes(("complexportal",))
    mem = sorted(df[df["members"].map(len).between(4, 10)].sort_values("complex_id")["members"].iloc[0])
    ids = mem[:3] + [f"{mem[3]}-2;é"]  # last feature: isoform + non-ASCII protein group
    rng = np.random.default_rng(0)
    X = rng.normal(10, 1, (25, len(ids)))
    X[rng.random(X.shape) < 0.1] = np.nan
    X[:15, -1] = np.nan  # 40% detected: not a selection candidate
    a = from_log2_matrix(X, [f"c{i}" for i in range(25)], ids)
    d = tmp_path_factory.mktemp("s7_tools")
    p = d / "données_蛋白.h5ad"
    a.write_h5ad(p)
    return p, a


def test_impute_tool_non_ascii_paths_roundtrip(h5ad):
    import anndata as ad

    p, a = h5ad
    out_p = p.with_name("sortie_é.h5ad")
    out = T.impute_with_prior(str(p), "complexportal", str(out_p))
    assert _ok(out) and out["output_path"] == str(out_p)
    res = ad.read_h5ad(out_p)
    X, Y = np.asarray(a.X), np.asarray(res.X)
    obs = ~np.isnan(X)
    assert np.array_equal(Y[obs], X[obs])
    assert out["n_imputed_from_partners"] + out["n_imputed_by_knn_fallback"] == out["n_missing"] == int((~obs).sum())


def test_select_panel_k_above_candidates_returns_at_most_candidates(h5ad):
    p, a = h5ad
    cand = set(np.array(a.var_names)[candidates(a)])
    assert len(cand) < a.n_vars
    for mode in ["soft_weight", "none"]:
        out = T.select_panel(str(p), "complexportal", a.n_vars, mode)
        assert _ok(out)
        panel = [r["protein"] for r in out["panel"]]
        assert len(panel) == len(set(panel)) == len(cand) and set(panel) == cand
        assert [r["rank"] for r in out["panel"]] == list(range(1, len(panel) + 1))


NOT_FOUND = [s + ".h5ad" for s in NON_ASCII[:4]] + ["", "x.csv", "/no/such/é.h5ad"]


@pytest.mark.parametrize("path", NOT_FOUND)
@pytest.mark.parametrize("tool", [T.impute_with_prior, T.select_panel], ids=["impute_with_prior", "select_panel"])
def test_file_tools_missing_or_non_ascii_path_returns_error(tool, path):
    assert _is_error(tool(path))


@pytest.mark.parametrize("tool", [T.impute_with_prior, T.select_panel], ids=["impute_with_prior", "select_panel"])
@pytest.mark.parametrize("path", [
    pytest.param(None, marks=_bug(TypeError, "tools._read calls Path(None) (tools.py:192)")),
    pytest.param(123, marks=_bug(TypeError, "tools._read calls Path(123) (tools.py:192)")),
])
def test_file_tools_non_string_path_returns_error(tool, path):
    assert _is_error(tool(path))


@pytest.mark.xfail(strict=True, raises=OSError, reason="core bug: tools._read does not catch ad.read_h5ad errors, "
                                                      "so a corrupt .h5ad raises OSError (tools.py:195); reported to Amit")
@pytest.mark.parametrize("tool", [T.impute_with_prior, T.select_panel], ids=["impute_with_prior", "select_panel"])
def test_file_tools_corrupt_h5ad_returns_error(tool, tmp_path):
    p = tmp_path / "corrupt_é.h5ad"
    p.write_text("not an HDF5 file", encoding="utf-8")
    assert _is_error(tool(str(p)))


@pytest.mark.xfail(strict=True, raises=KeyError, reason="core bug: tools._graph_for lets priors.build_prior raise "
                                                       "KeyError for an unknown prior name (tools.py:138); reported to Amit")
@pytest.mark.parametrize("prior", ["bogus", "é"])
@pytest.mark.parametrize("tool, kw", [(T.impute_with_prior, {}), (T.select_panel, {"k": 2})],
                         ids=["impute_with_prior", "select_panel"])  # valid k, so the prior lookup is reached
def test_file_tools_unknown_prior_returns_error(tool, kw, prior, h5ad):
    assert _is_error(tool(str(h5ad[0]), prior, **kw))


@pytest.mark.parametrize("kw", [{"k": 0}, {"k": -1}, {"k": "5"}, {"k": 10_000}, {"k": 2.5},
                                {"mode": "é"}, {"mode": "hard_filter"}])
def test_select_panel_bad_k_or_mode_returns_error(h5ad, kw):
    assert _is_error(T.select_panel(str(h5ad[0]), "complexportal", **kw))


@pytest.mark.parametrize("output_path", [
    "out.csv", "é",
    pytest.param(123, marks=_bug(TypeError, "tools.impute_with_prior calls Path(output_path) on a non-string "
                                            "(tools.py:210)")),
])
def test_impute_bad_output_path_returns_error(h5ad, output_path):
    assert _is_error(T.impute_with_prior(str(h5ad[0]), "complexportal", output_path))
