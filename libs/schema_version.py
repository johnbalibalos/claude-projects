"""
Central schema versioning for result dataclasses.

Bump SCHEMA_VERSION when making breaking changes to result formats.
This allows detection of incompatible results and potential migration.
"""

# Bump this when result schema changes in incompatible ways
SCHEMA_VERSION = "1.0.0"


def check_schema_compatibility(data: dict, min_version: str = "1.0.0") -> bool:
    """
    Check if loaded data is compatible with current code.

    Args:
        data: Dictionary loaded from results file
        min_version: Minimum compatible version

    Returns:
        True if compatible, False otherwise
    """
    version = data.get("schema_version", "0.0.0")
    return _version_gte(version, min_version)


def _version_gte(version: str, min_version: str) -> bool:
    """Check if version >= min_version using semver comparison."""
    def parse(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))
    return parse(version) >= parse(min_version)
