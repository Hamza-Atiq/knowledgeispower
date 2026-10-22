"""S3 (hu.MAP 3.0 priors): registration, graph rules, snapshot integrity, report counts.

Expected values are recomputed here from the snapshot CSV and the core graphs, never typed in.
"""
import hashlib
import re
import shutil
from itertools import combinations

import pandas as pd
import pytest

from scp_graphfs_mcp.complexes import COMPLEX_PORTAL_PATH, CORUM_PATH, HUMAP3_PATH
from scp_graphfs_mcp.datasets import load_petrosius2023
from scp_graphfs_mcp.ids import canonical_accession, extract_accessions, map_features
from scp_graphfs_mcp.priors import build_prior, shuffled_null
from scp_graphfs_mcp.registry import PRIORS, load_contrib, validate_prior

load_contrib()
PRIOR_NAMES = ["humap3_high", "humap3_union"]
PROV = open("data/PROVENANCE.md", encoding="utf-8").read()


@pytest.fixture(scope="module")
def ids():
    return list(load_petrosius2023().var_names)


@pytest.fixture(scope="module")
def graphs(ids):
    return {n: build_prior(n, ids) for n in PRIOR_NAMES}


def _expected_graph(ids, member_lists):
    """Independent edge rule: features share >= 1 complex; weight = number of shared complexes."""
    mapping, _ = map_features(ids)
    inv = {}
    for f, accs in mapping.items():
        for a in accs:
            inv.setdefault(a, []).append(f)
    w = {}
    for members in member_lists:
        feats = sorted({f for a in members for f in inv.get(a, [])})
        for u, v in combinations(feats, 2):
            w[frozenset((u, v))] = w.get(frozenset((u, v)), 0) + 1
    return w


def _weights(g):
    return {frozenset((u, v)): d["weight"] for u, v, d in g.edges(data=True)}


# --------------------------------------------------------------------------- registration (no silent pass)


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_prior_registered_as_contrib(name):
    # load_contrib() swallows import errors; this must FAIL (not xfail) if the package is broken.
    assert name in PRIORS, f"{name} not registered (did contrib.s3_prior fail to import?)"
    assert PRIORS[name].contrib


# --------------------------------------------------------------------------- graph rules


def test_high_matches_confidence_filtered_csv(ids, graphs):
    raw = pd.read_csv(HUMAP3_PATH, dtype=str).fillna("")
    keep = raw[raw["ComplexConfidence"].astype(int) <= 3]
    expected = _expected_graph(ids, [extract_accessions(s) for s in keep["Uniprot_ACCs"]])
    assert _weights(graphs["humap3_high"]) == expected
    assert graphs["humap3_high"].graph["provenance"]["max_confidence"] == 3


def test_high_is_strict_subset_of_core_humap3(ids, graphs):
    core = _weights(build_prior("humap3", ids))
    high = _weights(graphs["humap3_high"])
    assert set(high) < set(core)
    assert all(high[e] <= core[e] for e in high)


def test_union_is_weight_sum_of_core_graphs(ids, graphs):
    parts = [_weights(build_prior(n, ids)) for n in ("corum", "complexportal", "humap3")]
    expected = {}
    for p in parts:
        for e, w in p.items():
            expected[e] = expected.get(e, 0) + w
    assert _weights(graphs["humap3_union"]) == expected


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_valid_nonempty_and_null_keeps_degrees(ids, graphs, name):
    g = graphs[name]
    validate_prior(g, ids)
    assert g.number_of_edges() > 0
    assert not any(u == v for u, v in g.edges)
    h = shuffled_null(g, 0)
    assert dict(h.degree) == dict(g.degree)


# --------------------------------------------------------------------------- snapshot integrity (raw-or-LF)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("path", [HUMAP3_PATH, COMPLEX_PORTAL_PATH, CORUM_PATH])
def test_snapshot_hash_rule_matches_provenance_on_this_checkout(path):
    from contrib.s3_prior.humap import snapshot_sha256

    raw = path.read_bytes()
    logged = [h for h in (_sha(raw), _sha(raw.replace(b"\r\n", b"\n"))) if h in PROV]
    assert logged, f"neither raw nor LF hash of {path.name} is in data/PROVENANCE.md"
    assert snapshot_sha256(path, logged[0]) == logged[0]


def test_snapshot_accepts_crlf_and_lf_copies_rejects_modified(tmp_path):
    from contrib.s3_prior.humap import PIN, SnapshotError, snapshot_sha256

    lf = HUMAP3_PATH.read_bytes().replace(b"\r\n", b"\n")
    assert _sha(lf) == PIN
    (tmp_path / "lf.csv").write_bytes(lf)
    (tmp_path / "crlf.csv").write_bytes(lf.replace(b"\n", b"\r\n"))
    (tmp_path / "bad.csv").write_bytes(lf[:-2] + (b"X" if lf[-2:-1] != b"X" else b"Y") + lf[-1:])
    assert snapshot_sha256(tmp_path / "lf.csv", PIN) == PIN
    assert snapshot_sha256(tmp_path / "crlf.csv", PIN) == PIN
    with pytest.raises(SnapshotError, match="SHA-256"):
        snapshot_sha256(tmp_path / "bad.csv", PIN)
    with pytest.raises(SnapshotError, match="missing"):
        snapshot_sha256(tmp_path / "nope.csv", PIN)


def test_build_refuses_a_modified_snapshot(tmp_path, ids, monkeypatch):
    from contrib.s3_prior import humap

    bad = tmp_path / "humap.csv"
    shutil.copy(HUMAP3_PATH, bad)
    with open(bad, "ab") as fh:
        fh.write(b"extra")
    monkeypatch.setattr(humap, "SNAPSHOT_PATH", bad)
    with pytest.raises(humap.SnapshotError):
        humap.build_high(ids)


# --------------------------------------------------------------------------- provenance + report


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_provenance_pins_logged_hash(graphs, name):
    from contrib.s3_prior.humap import PIN

    prov = graphs[name].graph["provenance"]
    for k in ("source", "version", "url", "sha256"):
        assert prov.get(k)
    assert prov["sha256"] == PIN and PIN in PROV
    assert prov["url"] in PROV


def test_union_lists_logged_component_hashes(graphs):
    comps = graphs["humap3_union"].graph["provenance"]["components"]
    assert set(comps) == {"corum", "complexportal", "humap3"}
    for h in comps.values():
        assert re.fullmatch(r"[0-9a-f]{64}", h) and h in PROV


def test_report_counts_recomputed_from_csv(ids, graphs):
    raw = pd.read_csv(HUMAP3_PATH, dtype=str).fillna("")
    toks = [t for s in raw["Uniprot_ACCs"] for t in s.split()]
    dropped = [t for t in toks if canonical_accession(t) is None]
    rep = graphs["humap3_high"].graph["humap_report"]
    keep = raw[raw["ComplexConfidence"].astype(int) <= 3]
    assert rep["n_complexes_used"] == len(keep)
    assert rep["n_tokens_dropped_all_levels"] == len(dropped)
    assert rep["dropped_tokens"] == sorted(set(dropped))
    mapping, _ = map_features(ids)
    feat_of = {}
    for f, accs in mapping.items():
        for a in accs:
            feat_of.setdefault(a, set()).add(f)
    # "measured member" = distinct dataset feature; a complex with < 2 of them adds no edge
    lt2 = sum(len(set().union(*[feat_of.get(a, set()) for a in extract_accessions(s)] or [set()])) < 2
              for s in keep["Uniprot_ACCs"])
    assert rep["n_complexes_lt2_measured"] == lt2
