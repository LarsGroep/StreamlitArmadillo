"""Source-specific loaders.

Each loader turns ONE raw data source into canonical ArtistRecords
(see armadillo_scoring.schema). This package is the only place in the kit
allowed to know about source formats. A future Chartmetric loader lives here
as chartmetric.py and emits the same canonical attribute names.
"""

from armadillo_scoring.loaders import billboard

__all__ = ["billboard"]
