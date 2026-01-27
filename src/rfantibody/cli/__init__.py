"""
RFantibody Command Line Interface.

This package provides CLI utilities for:
- Working with Quiver files (protein design databases)
- Running RFdiffusion antibody design
- Running ProteinMPNN sequence design
- Running RF2 structure prediction
"""

from rfantibody.cli.inference import proteinmpnn, rf2, rfdiffusion
from rfantibody.cli.quiver import (
    qvextract,
    qvextractspecific,
    qvfrompdbs,
    qvls,
    qvrename,
    qvscorefile,
    qvslice,
    qvsplit,
)

__all__ = [
    # Quiver utilities
    'qvls',
    'qvextract',
    'qvextractspecific',
    'qvscorefile',
    'qvsplit',
    'qvslice',
    'qvrename',
    'qvfrompdbs',
    # Inference commands
    'rfdiffusion',
    'proteinmpnn',
    'rf2',
]
