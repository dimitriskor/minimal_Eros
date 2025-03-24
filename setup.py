from setuptools import setup, Extension
import pybind11
import os

ext_modules = [
    Extension(
        "event_processing",
        sources=["src/bindings.cpp", "src/surface.cpp", "src/eros.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        libraries=["opencv_core", "opencv_imgproc"],
        extra_compile_args=["-std=c++14"]
    )
]

setup(
    name="event_processing",
    ext_modules=ext_modules,
    zip_safe=False,
)
