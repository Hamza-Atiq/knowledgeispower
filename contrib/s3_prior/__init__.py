"""S3 extra prior source: hu.MAP 3.0 (high-confidence graph and curated ∪ hu.MAP union). See README.md."""

from scp_graphfs_mcp.registry import register_prior, register_tool

from .humap import build_high, build_union
from .mcp_tools import EXPLAIN_SCHEMA, IMPUTE_SCHEMA, humap_explain, humap_impute

register_prior("humap3_high", build_high,
               "hu.MAP 3.0 complexes with ComplexConfidence <= 3: edge = shared complex, weight = #shared")
register_prior("humap3_union", build_union,
               "CORUM ∪ Complex Portal ∪ hu.MAP 3.0 (all levels): weight = total #shared complexes")
register_tool("humap_explain", humap_explain,
              "A protein's hu.MAP 3.0 predicted complexes (confidence 1 = highest ... 6), partners, and its curated "
              "CORUM / Complex Portal complex counts. Accepts a UniProt accession or gene symbol.", EXPLAIN_SCHEMA)
register_tool("humap_impute", humap_impute,
              "Impute missing values in a .h5ad from observed hu.MAP partners in the same cell (kNN fallback); "
              "never-observed proteins stay missing. Writes a new .h5ad.", IMPUTE_SCHEMA)
