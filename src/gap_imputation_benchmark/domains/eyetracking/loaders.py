"""Eye-tracking loader entry points.

The legacy ``gap_imputation_benchmark.loaders`` modules remain as import
compatibility shims while additional domains receive their own loader modules.
"""

from gap_imputation_benchmark.loaders.gazebase import load_gazebase_reading
from gap_imputation_benchmark.loaders.gazebase_vr import load_gazebase_vr_reading
from gap_imputation_benchmark.loaders.pedrotti import load_pedrotti_reading
from gap_imputation_benchmark.loaders.zuco import load_zuco_normal_reading

__all__ = [
    "load_gazebase_reading",
    "load_gazebase_vr_reading",
    "load_pedrotti_reading",
    "load_zuco_normal_reading",
]
