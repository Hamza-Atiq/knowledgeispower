"""hu.MAP 3.0 priors for slot S3: a high-confidence graph and a curated ∪ hu.MAP union.

* ``humap3_high``: edge = the two features share >= 1 hu.MAP complex with ComplexConfidence <= 3
  (1 = extremely high ... 6 = medium); weight = number of shared complexes (same rule as the core priors).
* ``humap3_union``: core ``corum`` ∪ ``complexportal`` ∪ ``humap3`` (all levels); weights summed.

The snapshot is the core one (``data/raw/humap3_complexes.csv``, read via ``complexes.load_humap3``). Its SHA-256 is
checked before every build against the hash logged in ``data/PROVENANCE.md``. On Windows (``core.autocrlf=true``, no
``.gitattributes``) git checks LF text files out as CRLF, so the raw bytes **or** the CRLF->LF bytes may match; CORUM's
original download is itself CRLF, which is why the raw form is accepted too.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

from scp_graphfs_mcp.complexes import COMPLEX_PORTAL_PATH, CORUM_PATH, HUMAP3_PATH, load_humap3
from scp_graphfs_mcp.ids import canonical_accession
from scp_graphfs_mcp.phase2 import union_graph
from scp_graphfs_mcp.priors import _acc_to_features, _finish, build_prior

SNAPSHOT_PATH: Path = HUMAP3_PATH
MAX_CONFIDENCE = 3
# Provenance constants: the hashes logged in data/PROVENANCE.md (tests check they are there).
PIN = "22a545ce45ced9a57a15f3bff75beaf53cf7e254d8d68635668e2c968bf46739"
COMPONENT_PINS = {
    "corum": (CORUM_PATH, "5e40556eb59bb767f3396ba8c080f64c777903ffca9102345fdc35662a2e0282"),
    "complexportal": (COMPLEX_PORTAL_PATH, "29d3aeec7ede35e8686de3cf73620b5871c3e68d7ca069385153a9b28e80012a"),
}
URL = ("https://humap3.proteincomplexes.org/static/downloads/humap3/"
       "hu.MAP3.0_complexes_wConfidenceScores_total15326_wGenenames_20240922.csv")
VERSION = "hu.MAP 3.0 predicted complexes, release file dated 2024-09-22"


class SnapshotError(RuntimeError):
    """The pinned snapshot is missing or its bytes differ from the logged release."""


def snapshot_sha256(path, pin: str) -> str:
    """Return ``pin`` if the SHA-256 of the raw bytes or of the CRLF->LF bytes equals it; else raise.

    Computed fresh on every call (no cache), so a file changed during a session is caught.
    """
    path = Path(path)
    if not path.is_file():
        raise SnapshotError(f"snapshot missing: {path}")
    raw = path.read_bytes()
    seen = {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}
    if pin not in seen:
        raise SnapshotError(f"SHA-256 of {path} (raw or LF-normalised: {sorted(seen)}) does not match the "
                            f"logged release {pin}; restore the file, never re-download silently")
    return pin


def _complex_graph(protein_ids: list[str], member_lists) -> tuple[nx.Graph, list[str], int]:
    inv, unmapped = _acc_to_features(protein_ids)
    g = nx.Graph()
    g.add_nodes_from(protein_ids)
    n_lt2 = 0
    for members in member_lists:
        feats = sorted({f for a in members for f in inv.get(a, [])})
        n_lt2 += len(feats) < 2
        for u, v in combinations(feats, 2):
            w = g.edges[u, v]["weight"] + 1.0 if g.has_edge(u, v) else 1.0
            g.add_edge(u, v, weight=w)
    return g, unmapped, n_lt2


def _dropped_tokens() -> list[str]:
    raw = pd.read_csv(SNAPSHOT_PATH, dtype=str).fillna("")
    return [t for s in raw["Uniprot_ACCs"] for t in s.split() if canonical_accession(t) is None]


def _provenance(**extra) -> dict:
    return {"source": "hu.MAP 3.0 (Fischer et al. 2025, doi 10.1038/s44320-025-00121-5)", "version": VERSION,
            "url": URL, "sha256": snapshot_sha256(SNAPSHOT_PATH, PIN), **extra}


def build_high(protein_ids: list[str], max_confidence: int = MAX_CONFIDENCE) -> nx.Graph:
    prov = _provenance(max_confidence=max_confidence)
    cx = load_humap3()
    keep = cx[cx["confidence"] <= max_confidence]
    g, unmapped, n_lt2 = _complex_graph(list(protein_ids), keep["members"])
    dropped = _dropped_tokens()
    g.graph["humap_report"] = {"n_complexes_used": int(len(keep)), "n_complexes_lt2_measured": int(n_lt2),
                               "n_tokens_dropped_all_levels": len(dropped), "dropped_tokens": sorted(set(dropped))}
    return _finish(g, list(protein_ids), unmapped, prov)


def build_union(protein_ids: list[str]) -> nx.Graph:
    ids = list(protein_ids)
    comps = {name: snapshot_sha256(path, pin) for name, (path, pin) in COMPONENT_PINS.items()}
    prov = _provenance(components={**comps, "humap3": PIN})
    corum, cp, hm = (build_prior(n, ids) for n in ("corum", "complexportal", "humap3"))
    g = union_graph(union_graph(corum, cp), hm)
    g.graph = {}  # union_graph sets sha256 "-"; replace with the logged provenance below
    _, unmapped = _acc_to_features(ids)
    g.graph["humap_report"] = {"edges_by_source": {"corum": corum.number_of_edges(),
                                                   "complexportal": cp.number_of_edges(),
                                                   "humap3": hm.number_of_edges()}}
    return _finish(g, ids, unmapped, prov)
