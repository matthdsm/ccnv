"""Layer 0 exception hierarchy for ccnv."""


class CcnvError(Exception):
    """Base exception for all ccnv domain errors."""


class ReferenceError(CcnvError):
    """Raised when reference construction, validation, or cohort aggregation fails."""


class CovariateError(CcnvError):
    """Raised when GC or mappability covariate calculation or loading fails."""


class PredictError(CcnvError):
    """Raised when prediction or segmentation fails."""


class SchemaError(CcnvError):
    """Raised when an artifact violates a storage contract."""


class ExtractError(CcnvError):
    """Raised when extraction cannot proceed with the given inputs."""


class ReferenceQCError(CcnvError):
    """Raised when reference quality control checks fail."""


__all__ = [
    "CcnvError",
    "CovariateError",
    "ExtractError",
    "PredictError",
    "ReferenceError",
    "ReferenceQCError",
    "SchemaError",
]
