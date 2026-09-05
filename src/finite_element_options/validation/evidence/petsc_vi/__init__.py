"""Explicit external PETSc variational-inequality evidence boundary."""

from .adapter import (
    EXTERNAL_PROFILE_HINT,
    PetscVISolver,
    PetscVISolverSettings,
    petsc_runtime_doctor,
)
from .benchmark import PetscVIAssessmentConfig, run_petsc_vi_assessment

__all__ = [
    "EXTERNAL_PROFILE_HINT",
    "PetscVIAssessmentConfig",
    "PetscVISolver",
    "PetscVISolverSettings",
    "petsc_runtime_doctor",
    "run_petsc_vi_assessment",
]
