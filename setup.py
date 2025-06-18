from setuptools import setup, find_packages, Extension
import subprocess
import os
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext

from pathlib import Path
import glob
import shutil

import_name = "cw_plugins"


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
            CMAKE_CMD = "cmake3" if shutil.which("cmake3") else "cmake"
            # configure
            try:
                subprocess.check_call([CMAKE_CMD, ext.cmake_src_dir] + cmake_args, cwd=build_dir)
            except subprocess.CalledProcessError as e:
                print("CMake configuration failed")
                print(e)
                return
            # build
            subprocess.check_call([CMAKE_CMD, '--build', '.', '-j', f"{os.cpu_count()}"], cwd=build_dir)

        if self.inplace:
            self.copy_tree(self.build_lib, f"lib")


ext_modules = [
    CMakeExtension(f"{import_name}{os.sep}analyzer{os.sep}attacks", 'cpp_libs')
]

def post_process(installed_path):
    print("Search for hwh files of example designs")
    # find hwh files for pre-built targets
    repo_path = Path(__file__).parent

    sakura_x_shell = repo_path / "hardware" / "sakura-x-shell" / "examples"
    hwh_files = glob.glob(str(sakura_x_shell / "**/*.hwh"), recursive=True)
    copy_dst = installed_path / "targets" / "hwh_files" / "sakura-x"

    if not copy_dst.exists():
        print("create directory", copy_dst)
        copy_dst.mkdir(parents=True)
    for hwh_file in hwh_files:
        p = Path(hwh_file)
        name = p.parent.stem
        shutil.copy(hwh_file, str(copy_dst) + "/" + name + ".hwh")
        print("Adding", hwh_file, "to", copy_dst)

    cw305_shell = repo_path / "hardware" / "cw305-shell" / "examples"
    hwh_files = glob.glob(str(cw305_shell / "**/*.hwh"), recursive=True)
    copy_dst = installed_path / "targets" / "hwh_files" / "cw305"
    if not copy_dst.exists():
        print("create directory", copy_dst)
        copy_dst.mkdir(parents=True)
    for hwh_file in hwh_files:
        p = Path(hwh_file)
        name = p.parent.stem
        shutil.copy(hwh_file, str(copy_dst) + "/" + name + ".hwh")
        print("Adding", hwh_file, "to", copy_dst)

    bit_files = glob.glob(str(cw305_shell / "**/*.bit"), recursive=True)
    copy_dst = installed_path / "targets" / "bitstreams" / "cw305"
    if not copy_dst.exists():
        print("create directory", copy_dst)
        copy_dst.mkdir(parents=True)
    for bit_file in bit_files:
        p = Path(bit_file)
        name = p.parent.stem
        shutil.copy(bit_file, str(copy_dst) + "/" + name + ".bit")
        print("Adding", bit_file, "to", copy_dst)


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        installed_path = Path(self.install_lib) / import_name
        post_process(installed_path)


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        installed_path = Path(self.install_lib) / import_name
        post_process(installed_path)

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
        "pyelftools",
        "nest_asyncio",
        "h5py",
        "pytest"
    ],

    packages=find_packages(where='lib',exclude=['notebooks']),
    package_dir={'': 'lib'},
    include_package_data=True,

    cmdclass={"build_ext": CMakeBuild, "develop": PostDevelopCommand, "install": PostInstallCommand},
    ext_modules=ext_modules,

    scripts=[]

)

