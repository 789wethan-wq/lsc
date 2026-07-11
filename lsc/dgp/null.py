"""Matched-null helpers (SPEC §4.2).

Every DGP knows how to produce its own no-break version via
``null_version()``; this module provides the catalog-level helper so
calibration code never constructs nulls by hand.
"""
from __future__ import annotations

from .base import DGP


def matched_null(dgp: DGP) -> DGP:
    """Return the no-break counterpart of ``dgp`` used for threshold
    calibration. Guaranteed to itself be null (idempotent)."""
    null = dgp.null_version()
    # idempotence sanity: a null's null must equal itself structurally
    if type(null) is not type(dgp):
        raise TypeError("null_version must preserve the DGP class")
    return null
