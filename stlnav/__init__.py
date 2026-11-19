"""Route-scale navigation in the St. Lawrence, linked to the hull-scale simulator icebreaker_jax.py."""
import pathlib
import sys

# icebreaker_jax.py sits next to this package; make it importable wherever the package is imported from
_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
