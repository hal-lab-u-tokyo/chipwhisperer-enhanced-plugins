from setuptools import setup, find_packages, Extension
import subprocess
import os
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import_name = "cw_plugins"


class CMakeBuild(build_ext):
    """Build all CMake targets once, including during PEP 660 installs."""

    def run(self):
        import sys
        import pybind11

        # get_ext_fullpath respects both package_dir and inplace/editable mode.
        output_dir = Path(self.get_ext_fullpath(self.extensions[0].name)).resolve().parent
        build_dir = Path(self.build_temp).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        cmake = "cmake3" if shutil.which("cmake3") else "cmake"
        subprocess.check_call([
            cmake, str(Path(__file__).resolve().parent / "cpp_libs"),
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir.parent}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ], cwd=build_dir)
        subprocess.check_call([
            cmake, "--build", ".", "--parallel",
            os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1)),
        ], cwd=build_dir)
        for ext in self.extensions:
            if not ext.optional and not Path(self.get_ext_fullpath(ext.name)).is_file():
                raise RuntimeError(f"CMake did not produce {ext.name}")

    def _built_extensions(self):
        # GPU targets are optional and depend on the available toolkits.
        return [ext for ext in self.extensions
                if not ext.optional or Path(self.get_ext_fullpath(ext.name)).is_file()]

    def get_outputs(self):
        return [str(Path(self.build_lib) / self.get_ext_filename(ext.name))
                for ext in self._built_extensions()]

    def get_output_mapping(self):
        if not self.inplace:
            return {}
        return {str(Path(self.build_lib) / self.get_ext_filename(ext.name)):
                self.get_ext_fullpath(ext.name)
                for ext in self._built_extensions()}


module_prefix = "cw_plugins.analyzer.attacks.cpa_algorithms"
ext_modules = [
    Extension(f"{module_prefix}.{name}", sources=[], optional=optional)
    for name, optional in [
        ("model_kernel", False), ("cpa_kernel", False), ("socpa_kernel", False),
        ("cpa_cuda_kernel", True), ("socpa_cuda_kernel", True),
        ("cpa_opencl_kernel", True), ("socpa_opencl_kernel", True),
    ]
]


def hardware_files():
    """Map package-relative destinations to available hardware artifacts."""
    root = Path(__file__).resolve().parent / "hardware"
    files = {}
    for board in ("sakura-x", "cw305"):
        examples = root / f"{board}-shell" / "examples"
        for source in examples.rglob("*.hwh"):
            files[Path("targets/hwh_files") / board / f"{source.parent.stem}.hwh"] = source
        if board == "cw305":
            for source in examples.rglob("*.bit"):
                files[Path("targets/bitstreams/cw305") / f"{source.parent.stem}.bit"] = source
    for suffix, directory in (("hwh", "hwh_files"), ("bit", "bitstreams")):
        source = root / "VexRiscv_SCA/bitstream" / f"prebuilt_cw305.{suffix}"
        if source.is_file():
            files[Path("targets") / directory / "cw305" / f"VexRiscv.{suffix}"] = source
    return files


class BuildPy(build_py):
    """Include hardware data in wheels and beside sources for editable installs."""

    def run(self):
        super().run()
        destination = (Path(self.get_package_dir(import_name)) if self.editable_mode
                       else Path(self.build_lib) / import_name)
        for relative, source in hardware_files().items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def get_outputs(self, include_bytecode=1):
        return list(dict.fromkeys(super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / import_name / relative)
            for relative in hardware_files()
        ]))

    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        if self.editable_mode:
            mapping.update({
                str(Path(self.build_lib) / import_name / relative):
                str(Path(self.get_package_dir(import_name)) / relative)
                for relative in hardware_files()
            })
        return mapping


# Keep compiler intermediates and wheel staging outside the source checkout.
with TemporaryDirectory(prefix="cw-plugins-build-") as build_root:
    setup(
        name=f'{import_name}',
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
            "h5py",
            "pytest"
        ],

        packages=find_packages(where='lib',exclude=['notebooks']),
        package_dir={'': 'lib'},
        include_package_data=True,

        cmdclass={"build_ext": CMakeBuild, "build_py": BuildPy},
        ext_modules=ext_modules,

        options={"build": {"build_base": build_root}},

        scripts=[]

    )
