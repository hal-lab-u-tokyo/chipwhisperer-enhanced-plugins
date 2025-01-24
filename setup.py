from setuptools import setup, find_packages, Extension
import subprocess
import os
from setuptools.command.develop import develop
from setuptools.command.install import install
from pathlib import Path
import glob
import shutil

import_name = "cw_plugins"

try:
    import pybind11
except ImportError as e:
    ext_modules = []
    class CMakeBuild:
        def __init__(self, *args):
            super().__init__(*args)
            raise RuntimeError("pybind11 is required to build the C++ extension.")

        def run(self):
            raise RuntimeError("pybind11 is required to build the C++ extension.")
else:
    from setuptools.command.build_ext import build_ext

    class CMakeExtension(Extension):
        def __init__(self, name, cmake_src_dir):
            super().__init__(name, sources=[])
            self.cmake_src_dir = os.path.abspath(cmake_src_dir)

    class CMakeBuild(build_ext):

        def __init__(self, *args):
            super().__init__(*args)

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

def post_process(installed_path):
    # find hwh files for pre-built targets
    repo_path = Path(__file__).parent

    sakura_x_shell = repo_path / "hardware" / "sakura-x-shell" / "examples"
    hwh_files = glob.glob(str(sakura_x_shell / "**/*.hwh"), recursive=True)
    for hwh_file in hwh_files:
        p = Path(hwh_file)
        name = p.parent.stem
        shutil.copy(hwh_file, installed_path / "cw_plugins" / "targets" / "hwh_files" / "sakura-x" / name + ".hwh")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)



class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        post_process()

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
        "pycryptodome>=3.19.0",
        "matplotlib>=3.8.0",
        "numpy>=1.25.0",
        "ipyfilechooser",
        "pyelftools"
    ],

    packages=find_packages(where='lib',exclude=['notebooks']),
    package_dir={'': 'lib'},

    cmdclass={"build_ext": CMakeBuild},
    ext_modules=ext_modules,

    scripts=[]

)

