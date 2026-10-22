"""S3 dev study, exactly as preregistered in contrib/s3_prior/PREREGISTRATION.md.

    python contrib/s3_prior/explore_s3.py --out contrib/s3_prior/output          # full run -> s3_dev.json
    python contrib/s3_prior/explore_s3.py --quick --out benchmarks/_quick_out/s3  # smoke test, never a verdict

Loops mirror benchmarks/explore_ext.py (dev seeds 100-102, subsample seed = mask seed); the pass rule is the one in
benchmarks/confirm_v5.py (CI upper < -SIG_TOL and >= 1% of the comparator's RMSE). Refuses to run when
contrib/s3_prior/ has uncommitted changes, so the recorded commit is the code that ran.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
from run import SIG_TOL, boot_rmse_diff, make_mask  # noqa: E402

from scp_graphfs_mcp import phase2 as P  # noqa: E402
from scp_graphfs_mcp.complexes import in_any_complex  # noqa: E402
from scp_graphfs_mcp.datasets import (from_log2_matrix, load_leduc2022, load_petrosius2023,  # noqa: E402
                                      load_specht2021)
from scp_graphfs_mcp.ids import group_accessions  # noqa: E402
from scp_graphfs_mcp.methods import impute_with_prior  # noqa: E402
from scp_graphfs_mcp.priors import build_prior, shuffled_null  # noqa: E402
from scp_graphfs_mcp.registry import load_contrib  # noqa: E402

SEEDS = [100, 101, 102]
# regime -> n_cells (0 = all cells); n_cells // n_types per type, as explore_ext.py (Petrosius 30 -> 28)
TARGETS = [("petrosius2023", load_petrosius2023, {"low": 30, "high": 0}),
           ("leduc2022", load_leduc2022, {"low": 30, "high": 200}),
           ("specht2021", load_specht2021, {"low": 30, "high": 200})]
TARGET_NAMES = [t[0] for t in TARGETS]
HUMAP_PRIORS = ["humap3_high", "humap3_union", "humap3"]
PRIMARY_PRIOR = "humap3_high"
COMPARATORS = ["knn", "lowrank", "data_graph", "own_nulls", "complexportal"]
METHODS = ["E1", "E6"]
STRATA = ["all", "in_complex", "not_in_complex", "neutral"]
N_NULLS, N_BOOT, BOOT_SEED, SESOI_PCT = 5, 1000, 17, 1.0
MASKS = ["MCAR", "MNAR"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- rules (unit-tested)


def compare(se_arm: np.ndarray, se_refs: list[np.ndarray], n_boot: int = N_BOOT) -> dict:
    """ΔRMSE(arm − mean over refs), paired bootstrap CI (run.boot_rmse_diff, seed 17) and the confirm_v5 pass rule."""
    if len(se_arm) == 0:
        return {"delta_rmse": None, "ci95": [None, None], "effect_size_dz_sqerr": None, "n_entries": 0,
                "ref_rmse": None, "pct_of_ref_rmse": None, "significant": False, "pass": False, "note": "no entries"}
    c = boot_rmse_diff(se_arm, list(se_refs), n_boot, seed=BOOT_SEED)
    ref = float(np.mean([np.sqrt(r.mean()) for r in se_refs]))
    pct = 100 * c["delta_rmse"] / ref
    sig = bool(c["ci95"][1] < -SIG_TOL)
    return {**c, "ref_rmse": ref, "pct_of_ref_rmse": pct, "significant": sig, "pass": bool(sig and pct <= -SESOI_PCT)}


def primary_verdict(comparisons: list[dict]) -> dict:
    prim = [c for c in comparisons if c["prior"] == PRIMARY_PRIOR and c["method"] == "E1"
            and c["regime"] == "low" and c["stratum"] == "all"]
    expected = len(TARGET_NAMES) * len(MASKS) * len(COMPARATORS)
    if len(prim) != expected:
        raise ValueError(f"expected {expected} primary checks (30), got {len(prim)}")
    n_pass = sum(c["pass"] for c in prim)
    return {"n_checks": len(prim), "n_pass": n_pass,
            "verdict": "qualifies (dev)" if n_pass == len(prim) else "does not qualify",
            "failed": [{k: c[k] for k in ("target", "mask", "vs")} for c in prim if not c["pass"]]}


def same_sign(values) -> bool:
    v = np.asarray(values, float)
    return bool(np.all(v < 0) or np.all(v > 0))


def matched_verdict(by_mask: dict) -> str:
    if any(r["n_matched"] == 0 for r in by_mask.values()):
        return "inconclusive: no matched entries"
    if all(r["curated_minus_humap"]["significant"] for r in by_mask.values()):
        return "curated better"
    if all(r["humap_minus_curated"]["significant"] for r in by_mask.values()):
        return "hu.MAP better"
    return "inconclusive"


def subsample(cell_types: np.ndarray, n_cells: int, seed: int) -> np.ndarray:
    if not n_cells:
        return np.arange(len(cell_types))
    types = sorted(set(cell_types))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.where(cell_types == c)[0], n_cells // len(types), replace=False)
                                   for c in types]))


def _git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def require_clean() -> str:
    dirty = _git("status", "--porcelain", "--", "contrib/s3_prior", ":(exclude)contrib/s3_prior/output")
    if dirty.strip():
        sys.exit(f"refusing: uncommitted changes in contrib/s3_prior/:\n{dirty}")
    return _git("rev-parse", "HEAD").strip()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="seed 100, low-n only: smoke test, never a verdict")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.quick and a.out is None:
        sys.exit("--quick needs an explicit --out (e.g. benchmarks/_quick_out/s3)")
    a.out = a.out or ROOT / "contrib" / "s3_prior" / "output"
    return a


# --------------------------------------------------------------------------- study


def _refs(method: str, prior: str, vs: str) -> list[str]:
    return {"knn": ["knn"], "lowrank": ["lowrank"], "data_graph": [f"{method}|{prior}__data"],
            "own_nulls": [f"{method}|{prior}__null{i}" for i in range(N_NULLS)],
            "complexportal": [f"{method}|complexportal"]}[vs]


def run_target(tname, loader, regimes, seeds, rows, comps, graph_info):
    full = loader()
    names = list(full.var_names)
    in_cx = np.array([in_any_complex(group_accessions(f)) for f in names])
    G = {n: build_prior(n, names) for n in HUMAP_PRIORS + ["complexportal", "corum"]}
    nulls = {p: [shuffled_null(G[p], s) for s in range(N_NULLS)] for p in HUMAP_PRIORS}
    support = np.array([G[PRIMARY_PRIOR].degree(f) > 0 or G["complexportal"].degree(f) > 0 for f in names])
    graph_info[tname] = {n: {"n_edges": g.number_of_edges(), "n_with_edges": g.graph["coverage"]["n_with_edges"]}
                         for n, g in G.items()}
    graph_info[tname]["n_in_complex"] = int(in_cx.sum())
    graph_info[tname]["n_neutral_support"] = int(support.sum())
    log(f"{tname}: graphs + nulls built")
    ct = full.obs["cell_type"].values
    for regime, n_cells in regimes.items():
        for kind in MASKS:
            se: dict[str, list] = {}
            sel: dict[str, list] = {s: [] for s in STRATA}
            for seed in seeds:
                idx = subsample(ct, n_cells, seed)
                X = np.asarray(full.X, float)[idx]
                eligible = (~np.isnan(X)).mean(0) >= 0.5
                mask = make_mask(X, eligible, kind, seed)
                Xm = X.copy()
                Xm[mask] = np.nan
                truth = X[mask]
                _, cols = np.nonzero(mask)
                for s, v in (("all", np.ones(len(cols), bool)), ("in_complex", in_cx[cols]),
                             ("not_in_complex", ~in_cx[cols]), ("neutral", support[cols])):
                    sel[s].append(v)
                K = P.loo_knn(Xm)
                LR = P.grmf(Xm, None, names, lam_g=0.0, full=True)
                preds = {"knn": P._fill(Xm, K), "lowrank": P._fill(Xm, LR)}
                data = {p: P.data_graph(Xm, names, G[p].number_of_edges()) for p in HUMAP_PRIORS}
                for m, base in (("E1", K), ("E6", LR)):
                    for n, g in G.items():
                        preds[f"{m}|{n}"] = P.residual_on_knn(Xm, g, names, K=base)
                    for p in HUMAP_PRIORS:
                        preds[f"{m}|{p}__data"] = P.residual_on_knn(Xm, data[p], names, K=base)
                        for i, h in enumerate(nulls[p]):
                            preds[f"{m}|{p}__null{i}"] = P.residual_on_knn(Xm, h, names, K=base)
                for arm, Y in preds.items():
                    se.setdefault(arm, []).append((Y[mask] - truth) ** 2)
                log(f"{tname} {regime} (n={len(idx)}) {kind} seed {seed}: {len(truth)} masked entries")
            n_used = len(idx)
            SE = {a: np.concatenate(v) for a, v in se.items()}
            S = {s: np.concatenate(v) for s, v in sel.items()}
            for s in STRATA:
                for arm, v in SE.items():
                    per_seed = [float(np.sqrt(x[m].mean())) if m.any() else None for x, m in zip(se[arm], sel[s])]
                    rows.append({"target": tname, "regime": regime, "n_cells": n_used, "mask": kind, "stratum": s,
                                 "arm": arm, "rmse": float(np.sqrt(v[S[s]].mean())) if S[s].any() else None,
                                 "rmse_per_seed": per_seed, "n_entries": int(S[s].sum())})
            for p in HUMAP_PRIORS:
                for m in METHODS:
                    arm = f"{m}|{p}"
                    for vs in COMPARATORS:
                        refs = _refs(m, p, vs)
                        for s in STRATA:
                            c = compare(SE[arm][S[s]], [SE[r][S[s]] for r in refs])
                            deltas = [float(np.sqrt(se[arm][i][sel[s][i]].mean())
                                            - np.mean([np.sqrt(se[r][i][sel[s][i]].mean()) for r in refs]))
                                      if sel[s][i].any() else np.nan for i in range(len(seeds))]
                            comps.append({"target": tname, "regime": regime, "n_cells": n_used, "mask": kind,
                                          "stratum": s, "prior": p, "method": m, "vs": vs, **c,
                                          "delta_per_seed": deltas, "same_sign_all_seeds": same_sign(deltas)})
            log(f"{tname} {regime} {kind}: comparisons done")


def matched_diagnostic(seeds) -> dict:
    full = load_petrosius2023()
    names = list(full.var_names)
    in_cx = np.array([in_any_complex(group_accessions(f)) for f in names])
    G = {n: build_prior(n, names) for n in ["humap3_high", "complexportal", "corum"]}
    out = {"setting": "petrosius2023, all cells, methods.impute_with_prior (phase-1 imputer)", "pairs": {},
           "fallback_share": {}}
    store: dict = {k: {n: {"se": [], "by": [], "incx": []} for n in G} for k in MASKS}
    for kind in MASKS:
        for seed in seeds:
            X = np.asarray(full.X, float)
            eligible = (~np.isnan(X)).mean(0) >= 0.5
            mask = make_mask(X, eligible, kind, seed)
            Xm = X.copy()
            Xm[mask] = np.nan
            am = from_log2_matrix(Xm, full.obs_names, full.var_names, obs=full.obs)
            _, cols = np.nonzero(mask)
            for n, g in G.items():
                res = impute_with_prior(am, g)
                store[kind][n]["se"].append((np.asarray(res.X)[mask] - X[mask]) ** 2)
                store[kind][n]["by"].append(np.asarray(res.layers["imputed_by_prior"])[mask].astype(bool))
                store[kind][n]["incx"].append(in_cx[cols])
            log(f"matched diagnostic {kind} seed {seed}")
    for kind in MASKS:
        st = {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in store[kind].items()}
        out["fallback_share"][kind] = {
            n: {s: float(1 - d["by"][m].mean()) for s, m in (("all", np.ones_like(d["by"])), ("in_complex", d["incx"]),
                                                            ("not_in_complex", ~d["incx"]))}
            for n, d in st.items()}
        for cur in ["complexportal", "corum"]:
            matched = st["humap3_high"]["by"] & st[cur]["by"]
            se_h, se_c = st["humap3_high"]["se"][matched], st[cur]["se"][matched]
            out["pairs"].setdefault(cur, {})[kind] = {
                "n_matched": int(matched.sum()), "n_masked": int(len(matched)),
                "rmse_humap3_high": float(np.sqrt(se_h.mean())) if len(se_h) else None,
                f"rmse_{cur}": float(np.sqrt(se_c.mean())) if len(se_c) else None,
                "curated_minus_humap": compare(se_c, [se_h]), "humap_minus_curated": compare(se_h, [se_c])}
    for cur, by_mask in out["pairs"].items():
        by_mask["verdict"] = matched_verdict({k: by_mask[k] for k in MASKS})
    return out


def main(argv=None):
    a = parse_args(argv)
    commit = require_clean()
    load_contrib()
    t0 = time.time()
    seeds = SEEDS[:1] if a.quick else SEEDS
    rows, comps, graph_info = [], [], {}
    for tname, loader, regimes in TARGETS:
        run_target(tname, loader, {"low": regimes["low"]} if a.quick else regimes, seeds, rows, comps,
                   graph_info)
    out = {"code_commit": commit, "quick": a.quick, "preregistration": "contrib/s3_prior/PREREGISTRATION.md",
           "settings": {"seeds": seeds, "n_nulls": N_NULLS, "null_seeds": list(range(N_NULLS)), "n_boot": N_BOOT,
                        "boot_seed": BOOT_SEED, "sig_tol": SIG_TOL, "sesoi_pct": SESOI_PCT, "mask_frac": 0.1,
                        "eligible_detected": 0.5, "comparators": COMPARATORS, "strata": STRATA},
           "graphs": graph_info}
    if not a.quick:
        out["primary"] = primary_verdict(comps)
        out["per_dataset"] = {t: {"n_pass": sum(c["pass"] for c in comps if c["target"] == t and c["prior"] == PRIMARY_PRIOR
                                                and c["method"] == "E1" and c["regime"] == "low" and c["stratum"] == "all"),
                                  "n_checks": len(MASKS) * len(COMPARATORS)} for t in TARGET_NAMES}
        out["matched_diagnostic"] = matched_diagnostic(seeds)
    else:
        out["primary"] = "quick run: no verdict"
    out["comparisons"] = comps
    out["rmse"] = rows
    out["runtime_seconds"] = round(time.time() - t0, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / ("s3_dev_quick.json" if a.quick else "s3_dev.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {path} ({out['runtime_seconds']} s); primary: {out['primary'] if a.quick else out['primary']['verdict']}")


if __name__ == "__main__":
    main()
