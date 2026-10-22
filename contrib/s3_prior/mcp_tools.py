"""S3 contrib MCP tools (mounted only with ``--enable-contrib humap_explain,humap_impute``).

Both return JSON-serialisable dicts and never raise: bad input gives ``{"error": {"code", "message"}}``. Inputs are
validated before any computing, so nothing is written on error. Only ASCII identifiers are resolved (look-alike
Unicode letters such as a dotless i are rejected, never upper-cased into a real symbol).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scp_graphfs_mcp.complexes import HUMAP3_PATH, SOURCES, complexes_for_accession
from scp_graphfs_mcp.ids import canonical_accession, resolve_symbol, symbols_for_accession, uniprot_for_symbol

from .humap import MAX_CONFIDENCE, PIN, SnapshotError, snapshot_sha256

IMPUTE_PRIORS = ["humap3_high", "humap3_union", "humap3"]


def _err(code: str, message: str, **kw) -> dict:
    return {"error": {"code": code, "message": message, **kw}}


def _resolve(identifier) -> tuple[list[str], dict]:
    if not isinstance(identifier, str) or not identifier.strip():
        return [], {"how": "invalid"}
    if not identifier.isascii():
        return [], {"how": "unresolved"}
    acc = canonical_accession(identifier)
    if acc:
        return [acc], {"how": "uniprot"}
    syms, how = resolve_symbol(identifier)
    if len(syms) > 1:
        return [], {"how": "ambiguous", "candidates": syms}
    if not syms:
        return [], {"how": "unresolved"}
    return uniprot_for_symbol(syms[0]), {"how": how, "symbol": syms[0]}


def humap_explain(id: str) -> dict:
    """hu.MAP 3.0 complexes of a protein (UniProt accession or gene symbol) with confidence levels and partners."""
    try:
        accs, how = _resolve(id)
        if not accs:
            return _err(f"{how['how']}_identifier", f"could not resolve {id!r} to a UniProt accession",
                        **{k: v for k, v in how.items() if k != "how"})
        snapshot_sha256(HUMAP3_PATH, PIN)
        import pandas as pd

        rows = pd.concat([complexes_for_accession(a, ("humap3",)) for a in accs]).drop_duplicates("complex_id")
        rows = rows.sort_values(["confidence", "complex_id"])
        cur = pd.concat([complexes_for_accession(a) for a in accs]).drop_duplicates("complex_id")
        return {
            "input": id, "uniprot": accs, "symbols": sorted({s for a in accs for s in symbols_for_accession(a)}),
            "resolved_how": how["how"],
            "n_humap_complexes": int(len(rows)),
            "n_high_confidence": int((rows["confidence"] <= MAX_CONFIDENCE).sum()),
            "high_confidence_means": f"ComplexConfidence <= {MAX_CONFIDENCE} (1 = extremely high ... 6 = medium)",
            "complexes": [{"complex_id": r.complex_id, "confidence": int(r.confidence),
                           "partners": [{"uniprot": m, "symbols": symbols_for_accession(m)}
                                        for m in r.members if m not in accs]} for r in rows.itertuples()],
            "n_curated_complexes": {s: int((cur["source"] == s).sum()) if len(cur) else 0 for s in SOURCES},
            "caveat": "hu.MAP complexes are machine-learned predictions from interaction proteomics, trained with "
                      "Complex Portal labels; they are not independent of curation.",
            "provenance": f"data/raw/humap3_complexes.csv, SHA-256 {PIN} (data/PROVENANCE.md)",
        }
    except SnapshotError as exc:
        return _err("snapshot_error", str(exc))
    except Exception as exc:  # noqa: BLE001 (a tool never raises)
        return _err("internal_error", f"{type(exc).__name__}: {exc}")


def _check_paths(h5ad_path, prior, output_path):
    if not isinstance(prior, str) or prior not in IMPUTE_PRIORS:
        return None, None, _err("invalid_input", f"prior must be one of {IMPUTE_PRIORS}")
    if not isinstance(h5ad_path, str) or not h5ad_path.strip():
        return None, None, _err("invalid_input", "h5ad_path must be a non-empty string")
    src = Path(h5ad_path).expanduser()
    if src.suffix != ".h5ad" or not src.is_file():
        return None, None, _err("invalid_input", f"not an existing .h5ad file: {h5ad_path}")
    if output_path is None:
        dst = src.with_name(f"{src.stem}.imputed_{prior}.h5ad")
    elif not isinstance(output_path, str) or not output_path.strip():
        return None, None, _err("invalid_input", "output_path must be a string ending in .h5ad")
    else:
        dst = Path(output_path).expanduser()
    if dst.suffix != ".h5ad":
        return None, None, _err("invalid_input", "output_path must end in .h5ad")
    if dst.resolve() == src.resolve():
        return None, None, _err("invalid_input", "output_path must differ from the input file")
    if not dst.parent.is_dir():
        return None, None, _err("invalid_input", f"output folder does not exist: {dst.parent}")
    return src, dst, None


def humap_impute(h5ad_path: str, prior: str = "humap3_high", output_path: str | None = None) -> dict:
    """Fill missing values from observed hu.MAP partners in the same cell (kNN fallback); writes a new .h5ad.

    Proteins never observed in any cell are left missing (a partner-based value would be invented)."""
    try:
        src, dst, err = _check_paths(h5ad_path, prior, output_path)
        if err:
            return err
        import anndata as ad

        try:
            a = ad.read_h5ad(src)
        except Exception as exc:  # noqa: BLE001
            return _err("invalid_input", f"could not read {h5ad_path} as .h5ad: {type(exc).__name__}")
        X = np.asarray(a.X, float)
        if "raw" not in a.layers:
            a.layers["raw"] = np.exp2(X)
        from scp_graphfs_mcp.methods import impute_with_prior
        from scp_graphfs_mcp.priors import build_prior

        g = build_prior(prior, list(a.var_names))
        res = impute_with_prior(a, g)
        never = np.isnan(X).all(axis=0)
        Y = np.asarray(res.X, float).copy()
        Y[:, never] = np.nan
        res.X = Y
        by_prior = np.asarray(res.layers["imputed_by_prior"]) & ~never[None, :]
        res.layers["imputed_by_prior"] = by_prior
        miss = np.isnan(X)
        n_left = int(np.isnan(Y).sum())
        res.uns["impute_with_prior"] = {k: v for k, v in res.uns["impute_with_prior"].items() if k != "prior"}
        res.uns["impute_with_prior"]["prior"] = {k: str(v) for k, v in g.graph["provenance"].items()}
        res.write_h5ad(dst)
        per_prot = by_prior.sum(0)
        top = np.argsort(-per_prot)[:10]
        return {"output_path": str(dst), "prior": prior, "beta": float(res.uns["impute_with_prior"]["beta"]),
                "n_missing": int(miss.sum()), "n_imputed_from_partners": int(by_prior.sum()),
                "n_imputed_by_knn_fallback": int(miss.sum()) - int(by_prior.sum()) - n_left,
                "n_left_missing": n_left,
                "never_observed_left_missing": [str(f) for f in a.var_names[never]],
                "proteins_most_inferred_from_partners": [
                    {"protein": str(a.var_names[i]), "symbols": symbols_for_accession(a.var_names[i].split(";")[0]),
                     "n_cells_filled_from_partners": int(per_prot[i])} for i in top if per_prot[i] > 0],
                "caveat": "Dev results (contrib/s3_prior/output/s3_dev.json) show whether this prior beats kNN, "
                          "low-rank and its shuffled nulls; check them before trusting partner-based values."}
    except SnapshotError as exc:
        return _err("snapshot_error", str(exc))
    except Exception as exc:  # noqa: BLE001 (a tool never raises)
        return _err("internal_error", f"{type(exc).__name__}: {exc}")


EXPLAIN_SCHEMA = {"type": "object", "properties": {"id": {"type": "string",
                  "description": "UniProt accession (e.g. P28074) or HGNC gene symbol (e.g. PSMB5)"}},
                  "required": ["id"]}
IMPUTE_SCHEMA = {"type": "object", "properties": {
    "h5ad_path": {"type": "string", "description": "existing .h5ad (cells x proteins, log2, NaN = missing)"},
    "prior": {"type": "string", "enum": IMPUTE_PRIORS, "default": "humap3_high"},
    "output_path": {"type": "string", "description": "new .h5ad path (default: <input>.imputed_<prior>.h5ad)"}},
    "required": ["h5ad_path"]}
