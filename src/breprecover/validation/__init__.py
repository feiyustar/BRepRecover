"""Optional independent validation adapters."""

from breprecover.validation.occ_validator import (
    OCCValidationOutcome,
    OCCTShapeValidator,
    StepFileOutcome,
    validate_step_file,
)
from breprecover.validation.isolated_step import IsolatedSTEPOutcome, IsolatedSTEPValidator

__all__ = [
    "OCCValidationOutcome",
    "OCCTShapeValidator",
    "IsolatedSTEPOutcome",
    "IsolatedSTEPValidator",
    "StepFileOutcome",
    "validate_step_file",
]
