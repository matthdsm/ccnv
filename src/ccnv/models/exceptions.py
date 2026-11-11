"""Layer 0 exception hierarchy for ccnv."""


class CcnvError(Exception):
    """Base exception for all ccnv domain errors."""


class ReferenceError(CcnvError):
    """Raised when reference construction, validation, or cohort aggregation fails."""


class CovariateError(CcnvError):
    """Raised when GC or mappability covariate calculation or loading fails."""


class PredictError(CcnvError):
    """Raised when prediction or segmentation fails."""


class BinCompatibilityError(ReferenceError, PredictError):
    """Raised when sample and target bin grids cannot be reconciled without upsampling."""


class SchemaError(CcnvError):
    """Raised when an artifact violates a storage contract."""


class SchemaVersionError(SchemaError):
    """Raised when an artifact uses an unsupported or legacy schema version."""


class ExtractError(CcnvError):
    """Raised when extraction cannot proceed with the given inputs."""


class AlignmentError(ExtractError):
    """Raised when BAM or CRAM header, index, or record reading fails."""


class ReferenceQCError(CcnvError):
    """Raised when reference quality control checks fail."""


__all__ = [
    "AlignmentError",
    "BinCompatibilityError",
    "CcnvError",
    "CovariateError",
    "ExtractError",
    "PredictError",
    "ReferenceError",
    "ReferenceQCError",
    "SchemaError",
    "SchemaVersionError",
]
