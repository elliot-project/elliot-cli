"""ELLIOT evaluation platform (import package ``oellm``, distribution ``oellm-eval``)."""

from importlib import metadata

# The version lives only in pyproject.toml, so it follows upstream's version
# bumps on every merge.
try:
    __version__ = metadata.version("oellm-eval")
except metadata.PackageNotFoundError:  # a source tree that was never installed
    __version__ = "unknown"
