"""armadillo_scoring — source-agnostic artist speed-scoring kit.

Pipeline:  loader -> schema (canonical records) -> signals -> score -> validate

The CORE (schema, signals, score, validate) is source-agnostic.
Swapping data sources should require ONLY a new loader + (optionally) a new
signal configuration. Currently developed on the PUBLIC billboard dataset;
Chartmetric can later feed the exact same schema.
"""

from armadillo_scoring import schema

__all__ = ["schema"]
