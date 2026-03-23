"""MoEBench: system feature collection for benchmark-aware scheduling research."""

from moebench.collector import collect_all, collect_dynamic
from moebench.static_features import collect_static

__all__ = ["collect_all", "collect_static", "collect_dynamic", "__version__"]

__version__ = "0.1.0"
