from setuptools import setup, find_packages, Extension
import subprocess
import os

import_name = "chipwhisperer_enahnced_plugins"

try:
    import pybind11
except ImportError:
    ext_modules = []
    class CMakeBuild:
        def __init__(self, *args, **kwargs):
            pass
        def run(self):
            raise RuntimeError("pybind11 is required to build the C++ extension.")
else:
    from setuptools.command.build_ext import build_ext
    class CMakeExtension(Extension):
        def __init__(self, name, cmake_src_dir):
            super().__init__(name, sources=[])
            self.cmake_src_dir = os.path.abspath(cmake_src_dir)


    class CMakeBuild(build_ext):
        def run(self):
            # check if build directory exists
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            abs_build_lib = os.path.abspath(self.build_lib)

            for ext in self.extensions:

                cmake_args = [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={abs_build_lib}{os.sep}{ext.name}']

                build_dir = f"{self.build_temp}{os.sep}{ext.name}"
                os.makedirs(build_dir, exist_ok=True)
                # configure
                subprocess.check_call(['cmake', ext.cmake_src_dir] + cmake_args, cwd=build_dir)
                # build
                subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)

            if self.inplace:
                self.copy_tree(self.build_lib, f"lib")


    ext_modules = [
        CMakeExtension(f"{import_name}{os.sep}analyzer{os.sep}attacks", 'cpp_libs')
    ]

setup(
    name=f'{import_name}',
    version='1.0.0',
    license='MIT',
    description='python tools for power analysis-based side-channel attack',

    author='Takuya Kojima',
    author_email='tkojima@hal.ipc.i.u-tokyo.ac.jp',
    url='https://www.tkojima.me',

    install_requires=[
        "PyUSB>=1.2.1",
        "pyvisa>=1.13.0",
        "ftd2xx>=1.3.3",
        "pycryptodome>=3.19.0",
        "matplotlib>=3.8.0",
        "numpy>=1.25.0",
    ],

    packages=find_packages(where='lib',exclude=['notebooks']),
    package_dir={'': 'lib'},

    cmdclass={"build_ext": CMakeBuild},
    ext_modules=ext_modules,

    scripts=[]

)

