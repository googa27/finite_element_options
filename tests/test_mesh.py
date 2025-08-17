"""Tests for mesh creation utilities."""

import numpy as np
import pytest

from src.space.mesh import create_mesh


def test_create_mesh_invalid_dimension() -> None:
    """Mesh creation should reject dimensions higher than three."""
    with pytest.raises(ValueError):
        create_mesh([1.0, 1.0, 1.0, 1.0], 0)


def test_create_mesh_domain_extent() -> None:
    """Generated mesh should span provided domain extents."""
    mesh = create_mesh([1.0, 1.0], 1)
    assert np.allclose(mesh.p.max(axis=1), [1.0, 1.0])
