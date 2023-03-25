from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name = "fmsnepyx",
    description = 'Fast Multi-Scale Neighbour Embedding (cython implementation)',
    version = "0.1.0",
    ext_modules = cythonize([Extension("fmsnepyx", ["fmsne_implem.pyx", "lbfgs.c"])], annotate=False),
    include_dirs=[np.get_include(), '.']
)
