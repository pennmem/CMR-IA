from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

# CMR_IA.pyx compiles to CMR_IA._core, which CMR_IA/__init__.py re-exports.
ext = Extension(
    "CMR_IA._core",
    sources=["CMR_IA/_core.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    ext_modules=cythonize([ext], annotate=True),
)
