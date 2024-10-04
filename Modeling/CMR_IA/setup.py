from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name='CMR_IA',
    ext_modules = cythonize("CMR_IA.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
