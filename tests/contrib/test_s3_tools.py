"""S3 MCP tools: humap_explain and humap_impute. Expected values are derived in-test from the snapshot."""
import json

import numpy as np
import pytest

from scp_graphfs_mcp.complexes import complexes_for_accession, load_humap3
from scp_graphfs_mcp.datasets import load_petrosius2023
from scp_graphfs_mcp.ids import resolve_symbol, symbols_for_accession, uniprot_for_symbol
from scp_graphfs_mcp.registry import TOOLS, load_contrib
from scp_graphfs_mcp.server import mounted_tools

load_contrib()
TOOL_NAMES = ["humap_explain", "humap_impute"]
BAD_IDS = ["", "   ", None, 123, 1.5, ["P28074"], {"id": "P28074"}, "NOTAGENE123",
           "ınsr", "P28074-٣", "名字", "\U0001f9ec", "P2807４", "РSMB5"]


def _tool(name):
    return TOOLS[name].func


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_tool_registered_as_contrib(name):
    assert name in TOOLS, f"{name} not registered (did contrib.s3_prior fail to import?)"
    assert TOOLS[name].contrib


def test_mounted_only_when_enabled(monkeypatch):
    monkeypatch.delenv("SCP_GRAPHFS_ENABLE_CONTRIB", raising=False)
    assert not set(TOOL_NAMES) & set(mounted_tools())
    assert set(TOOL_NAMES) <= set(mounted_tools(TOOL_NAMES))


# --------------------------------------------------------------------------- humap_explain


def _probe_accession():
    """A hu.MAP member whose first HGNC symbol resolves back to exactly this accession."""
    cx = load_humap3()
    for members in cx.sort_values("confidence")["members"]:
        for a in members:
            syms = symbols_for_accession(a)
            if syms and resolve_symbol(syms[0])[0] == [syms[0]] and uniprot_for_symbol(syms[0]) == [a]:
                return a, syms[0]
    raise AssertionError("no probe accession found")


def test_explain_matches_snapshot():
    acc, _ = _probe_accession()
    out = _tool("humap_explain")(acc)
    cx = load_humap3()
    hit = cx[cx["members"].map(lambda m: acc in m)]
    assert out["uniprot"] == [acc]
    assert {c["complex_id"] for c in out["complexes"]} == set(hit["complex_id"])
    assert out["n_humap_complexes"] == len(hit)
    assert out["n_high_confidence"] == int((hit["confidence"] <= 3).sum())
    for c in out["complexes"]:
        row = hit[hit["complex_id"] == c["complex_id"]].iloc[0]
        assert c["confidence"] == int(row["confidence"])
        assert {p["uniprot"] for p in c["partners"]} == set(row["members"]) - {acc}
    cur = complexes_for_accession(acc)
    assert out["n_curated_complexes"] == {s: int((cur["source"] == s).sum()) for s in ("corum", "complexportal")}
    json.dumps(out)


def test_explain_by_symbol_equals_by_accession():
    acc, sym = _probe_accession()
    a, b = _tool("humap_explain")(acc), _tool("humap_explain")(sym)
    assert b["uniprot"] == a["uniprot"] and b["complexes"] == a["complexes"]


@pytest.mark.parametrize("bad", BAD_IDS)
def test_explain_bad_input_returns_error(bad):
    out = _tool("humap_explain")(bad)
    assert set(out) == {"error"} and {"code", "message"} <= set(out["error"])
    json.dumps(out)


# --------------------------------------------------------------------------- humap_impute


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    a = load_petrosius2023()[:, :300].copy()
    d = tmp_path_factory.mktemp("s3tools")
    p = d / "x.h5ad"
    a.write_h5ad(p)
    return a, p


def test_impute_fills_only_nan_and_reports_counts(small, tmp_path):
    a, p = small
    outp = tmp_path / "y.h5ad"
    out = _tool("humap_impute")(str(p), "humap3_high", str(outp))
    assert "error" not in out, out
    import anndata as ad

    b = ad.read_h5ad(outp)
    X, Y = np.asarray(a.X, float), np.asarray(b.X, float)
    obs = ~np.isnan(X)
    assert np.array_equal(X[obs], Y[obs])
    assert out["n_missing"] == int((~obs).sum())
    assert out["n_imputed_from_partners"] + out["n_imputed_by_knn_fallback"] + out["n_left_missing"] == out["n_missing"]
    assert out["n_left_missing"] == int(np.isnan(Y).sum())
    json.dumps(out)


def test_impute_never_invents_values_for_never_observed_proteins(small, tmp_path):
    """Guard against core methods.py bug (all-NaN protein with a partner gets ~0): such columns stay NaN."""
    from scp_graphfs_mcp.priors import build_prior

    a, _ = small
    g = build_prior("humap3_high", list(a.var_names))
    j = next(i for i, f in enumerate(a.var_names) if g.degree(f) > 0)
    b = a.copy()
    X = np.asarray(b.X, float).copy()
    X[:, j] = np.nan
    b.X = X
    b.layers["raw"] = np.exp2(X)
    p = tmp_path / "allnan.h5ad"
    b.write_h5ad(p)
    out = _tool("humap_impute")(str(p), "humap3_high", str(tmp_path / "allnan_out.h5ad"))
    import anndata as ad

    Y = np.asarray(ad.read_h5ad(tmp_path / "allnan_out.h5ad").X, float)
    assert np.isnan(Y[:, j]).all()
    assert a.var_names[j] in out["never_observed_left_missing"]


def _bad_cases(tmp_path, p):
    corrupt = tmp_path / "corrupt.h5ad"
    corrupt.write_bytes(b"not an h5ad file")
    return [
        dict(h5ad_path=str(tmp_path / "missing.h5ad")),
        dict(h5ad_path=str(p.with_suffix(".csv"))),
        dict(h5ad_path=str(corrupt)),
        dict(h5ad_path=None), dict(h5ad_path=123), dict(h5ad_path=""), dict(h5ad_path="名字.h5ad"),
        dict(h5ad_path=str(p), prior="bogus"), dict(h5ad_path=str(p), prior="é"),
        dict(h5ad_path=str(p), prior=None), dict(h5ad_path=str(p), prior="corum"),
        dict(h5ad_path=str(p), output_path=str(p)),
        dict(h5ad_path=str(p), output_path=str(tmp_path / "out.csv")),
        dict(h5ad_path=str(p), output_path=str(tmp_path / "no_such_dir" / "o.h5ad")),
        dict(h5ad_path=str(p), output_path=123),
    ]


def test_impute_bad_inputs_return_errors_and_write_nothing(small, tmp_path):
    _, p = small
    cases = _bad_cases(tmp_path, p)
    before = sorted(tmp_path.rglob("*"))
    for kw in cases:
        out = _tool("humap_impute")(**kw)
        assert set(out) == {"error"}, (kw, out)
        json.dumps(out)
    assert sorted(tmp_path.rglob("*")) == before, "nothing may be written on error"
