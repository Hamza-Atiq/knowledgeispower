"""S7 hardening: identifier handling, prior graphs, shuffled nulls and registry validation.

Edge cases from docs/specs/S7_hardening.md (empty graphs, protein groups with isoforms,
non-ASCII input). Expected values are derived in-test from the committed snapshots or from
the documented contract; nothing is hard-coded from a previous run.
"""

from collections import Counter

import networkx as nx
import pytest

from scp_graphfs_mcp.complexes import load_complexes
from scp_graphfs_mcp.ids import canonical_accession, extract_accessions, group_accessions, resolve_symbol
from scp_graphfs_mcp.priors import build_complex_graph, build_prior, nulls, shuffled_null
from scp_graphfs_mcp.registry import (METHODS, PRIORS, TOOLS, register_method, register_prior, register_tool,
                                      validate_prior)

NON_ASCII = ["é", "\U0001f9ec", "蛋白质", "P28074́", "Р" + "28074", "P2807４"]


def _prov():
    return {"source": "s7-toy", "version": "0", "url": "-", "sha256": "-"}


def _toy(nodes, edges, cls=nx.Graph):
    g = cls()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges, weight=1.0)
    g.graph["provenance"] = _prov()
    return g


def _complex_with(source="corum", min_size=3):
    """Members of a real complex from the snapshot (first by complex_id with >= min_size members)."""
    df = load_complexes((source,))
    sizes = df["members"].map(len)
    row = df[sizes >= min_size].sort_values("complex_id").iloc[0]
    return sorted(row["members"])


# --------------------------------------------------------------------------- ids


# Syntactic inputs only: suffix numbers are arbitrary, the base accession is PSMB5 (P28074).
@pytest.mark.parametrize("raw", ["P28074-2", "P28074-12", "P28074-PRO_0000026580", " P28074 ", "P28074"])
def test_isoform_and_chain_suffixes_strip_to_canonical(raw):
    assert canonical_accession(raw) == "P28074"


@pytest.mark.parametrize("raw", ["p28074", "P28074-", "P28074-X", "P28074;P20618", ""] + NON_ASCII)
def test_non_accessions_return_none(raw):
    assert canonical_accession(raw) is None


@pytest.mark.xfail(strict=True, reason="core bug: ids._UNIPROT_FULL_RE uses Unicode \\d, so non-ASCII "
                                       "digits pass as an isoform suffix; reported to Amit")
@pytest.mark.parametrize("raw", ["P28074-٣", "P28074-２"])
def test_non_ascii_isoform_digits_rejected(raw):
    assert canonical_accession(raw) is None


def test_protein_group_with_isoforms_dedupes_in_order():
    assert group_accessions("P28074-2;P28074;P20618-PRO_0000026574;P20618") == ["P28074", "P20618"]
    assert group_accessions("P28074;é;\U0001f9ec") == ["P28074"]
    assert group_accessions("") == [] and extract_accessions(None) == []


@pytest.mark.parametrize("q", NON_ASCII + ["", " ", "\x00"])
def test_resolve_symbol_non_ascii_unresolved(q):
    assert resolve_symbol(q) == ([], "unresolved")


@pytest.mark.xfail(strict=True, reason="core bug: ids.resolve_symbol uses str.upper(), so non-ASCII "
                                       "'\\u0131nsr' (dotless i) silently resolves to INSR; reported to Amit")
def test_non_ascii_lookalike_symbol_not_guessed():
    # Precondition derived in-test: the ASCII spelling *is* a current symbol.
    assert resolve_symbol("INSR")[1] == "symbol"
    assert resolve_symbol("ınsr") == ([], "unresolved")


# --------------------------------------------------------------------------- priors: empty / unmapped


@pytest.mark.parametrize("source", ["corum", "complexportal", "string_exp"])
def test_empty_protein_list_gives_empty_valid_graph(source):
    g = build_prior(source, [])
    assert g.number_of_nodes() == 0 and g.number_of_edges() == 0
    cov = g.graph["coverage"]
    assert cov["n_features"] == 0 and cov["unmapped_fraction"] == 0.0 and cov["n_edges"] == 0
    validate_prior(g, [])


@pytest.mark.parametrize("source", ["corum", "complexportal", "string"])
def test_all_unmapped_non_ascii_ids(source):
    g = build_prior(source, NON_ASCII)
    assert set(g.nodes) == set(NON_ASCII) and g.number_of_edges() == 0
    cov = g.graph["coverage"]
    assert cov["unmapped_fraction"] == 1.0 and sorted(cov["unmapped"]) == sorted(NON_ASCII)


# --------------------------------------------------------------------------- priors: isoforms / groups


def test_isoform_features_both_get_partner_edges_and_counted_weights():
    m = _complex_with("corum")
    a, b = m[0], m[1]
    ids = [a, f"{a}-2", b]
    g = build_complex_graph(ids, "corum")
    validate_prior(g, ids)
    assert not any(u == v for u, v in g.edges)
    df = load_complexes(("corum",))
    expected = sum(1 for mem in df["members"] if a in mem and b in mem)
    assert expected >= 1
    assert g.edges[a, b]["weight"] == expected
    assert g.edges[f"{a}-2", b]["weight"] == expected  # isoform maps to the same canonical accession


def test_group_holding_two_subunits_of_one_complex_has_no_self_loop():
    m = _complex_with("corum")
    group = f"{m[0]};{m[1]}-2"
    ids = [group, m[2]]
    for source in ["corum", "complexportal", "string"]:
        g = build_prior(source, ids)
        validate_prior(g, ids)
        assert nx.number_of_selfloops(g) == 0
    assert build_complex_graph(ids, "corum").has_edge(group, m[2])


# --------------------------------------------------------------------------- shuffled nulls


@pytest.mark.parametrize("edges", [[], [("a", "b")]])
def test_null_of_tiny_graph_is_unchanged(edges):
    g = _toy(list("abcd"), edges)
    h = shuffled_null(g, 0)
    assert set(map(frozenset, h.edges)) == set(map(frozenset, g.edges))
    assert set(h.nodes) == set(g.nodes)
    assert h.graph["provenance"]["n_swaps"] == 0 and h.graph["provenance"]["null_of"] == "s7-toy"
    validate_prior(h, list(g.nodes))


def test_null_of_node_less_graph():
    h = shuffled_null(_toy([], []), 3)
    assert h.number_of_nodes() == 0 and h.graph["provenance"]["null_seed"] == 3


@pytest.fixture(scope="module")
def real_prior():
    df = load_complexes(("complexportal",))
    big = df[df["members"].map(len).between(4, 12)].sort_values("complex_id")
    ids = sorted({a for mem in big["members"].iloc[:15] for a in mem})
    ids += [f"{ids[0]}-2", "é"]  # an isoform feature and an unmapped non-ASCII id
    return build_prior("complexportal", ids), ids


def test_null_preserves_degrees_weights_no_self_loops(real_prior):
    g, ids = real_prior
    assert g.number_of_edges() >= 2
    for seed in (0, 1, 7):
        h = shuffled_null(g, seed)
        assert dict(h.degree) == dict(g.degree)
        assert set(map(frozenset, h.edges)) != set(map(frozenset, g.edges))  # the null is not vacuous
        assert Counter(w for *_, w in h.edges(data="weight")) == Counter(w for *_, w in g.edges(data="weight"))
        assert nx.number_of_selfloops(h) == 0
        p = h.graph["provenance"]
        assert 0 <= p["n_swaps"] <= p["n_swaps_target"] == 10 * g.number_of_edges()
        validate_prior(h, ids)


def test_null_is_seeded(real_prior):
    g, _ = real_prior
    a, b = shuffled_null(g, 5), shuffled_null(g, 5)
    assert list(a.edges(data="weight")) == list(b.edges(data="weight"))
    assert nulls(g, 0) == []
    ns = nulls(g, 3, base_seed=10)
    assert [n.graph["provenance"]["null_seed"] for n in ns] == [10, 11, 12]


# --------------------------------------------------------------------------- registry validation


def _bad_builders():
    def selfloop(ids):
        return _toy(ids, [(ids[0], ids[0])])

    def foreign(ids):
        return _toy(list(ids) + ["NOT_IN_IDS"], [(ids[0], "NOT_IN_IDS")])

    def zero_w(ids):
        g = _toy(ids, [])
        g.add_edge(ids[0], ids[1], weight=0.0)
        return g

    def nan_w(ids):
        g = _toy(ids, [])
        g.add_edge(ids[0], ids[1], weight=float("nan"))
        return g

    def neg_w(ids):
        g = _toy(ids, [])
        g.add_edge(ids[0], ids[1], weight=-1.0)
        return g

    def no_prov(ids):
        g = _toy(ids, [(ids[0], ids[1])])
        g.graph.pop("provenance")
        return g

    def partial_prov(ids):
        g = _toy(ids, [(ids[0], ids[1])])
        g.graph["provenance"] = {"source": "x"}
        return g

    def directed(ids):
        return _toy(ids, [(ids[0], ids[1])], nx.DiGraph)

    def multi(ids):
        return _toy(ids, [(ids[0], ids[1])], nx.MultiGraph)

    return [selfloop, foreign, zero_w, nan_w, neg_w, no_prov, partial_prov, directed, multi]


@pytest.mark.parametrize("builder", _bad_builders(), ids=lambda f: f.__name__)
def test_registry_rejects_invalid_contrib_prior(builder):
    name = f"_s7_{builder.__name__}"
    register_prior(name, builder, "s7 invalid prior", contrib=True)
    try:
        with pytest.raises(ValueError):
            build_prior(name, ["P28074", "P20618"])
    finally:
        PRIORS.pop(name)


def test_build_prior_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        build_prior("é_no_such_prior", ["P28074"])


@pytest.mark.parametrize("bad", ["", "has space", "dot.name", "slash/name"])
def test_registry_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        register_prior(bad, lambda ids: None, "x")
    assert bad not in PRIORS


def test_registry_rejects_duplicates_and_empty_methods_and_bad_schema():
    with pytest.raises(ValueError):
        register_prior("corum", lambda ids: None, "dup")
    with pytest.raises(ValueError):
        register_method("_s7_nothing", "no impute, no select")
    assert "_s7_nothing" not in METHODS
    for schema in [{}, {"type": "object"}, {"type": "array", "properties": {}}, "not a dict"]:
        with pytest.raises(ValueError):
            register_tool("_s7_badschema", lambda: {}, "x", schema)
        assert "_s7_badschema" not in TOOLS


def test_validate_prior_accepts_empty_graph_with_provenance():
    validate_prior(_toy([], []), [])
    validate_prior(_toy(["a"], []), ["a", "b"])  # subset of ids is fine
