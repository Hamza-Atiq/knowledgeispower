"""S3 extra prior source: hu.MAP 3.0 (high-confidence graph and curated ∪ hu.MAP union). See README.md."""

from scp_graphfs_mcp.registry import register_prior

from .humap import build_high, build_union

register_prior("humap3_high", build_high,
               "hu.MAP 3.0 complexes with ComplexConfidence <= 3: edge = shared complex, weight = #shared")
register_prior("humap3_union", build_union,
               "CORUM ∪ Complex Portal ∪ hu.MAP 3.0 (all levels): weight = total #shared complexes")
