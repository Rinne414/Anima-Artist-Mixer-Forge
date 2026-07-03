try:
    from .anima_mixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:
    # Only fall back to the absolute import for the relative-import-context
    # failure (this file imported as a top-level module, e.g. by pytest, where
    # exc.name is None) or when the package itself cannot be resolved. A real
    # ImportError from a missing dependency inside the package (e.g. torch)
    # must surface instead of being masked by the bogus fallback error.
    if exc.name is not None and not (exc.name or "").endswith("anima_mixer"):
        raise
    from anima_mixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
