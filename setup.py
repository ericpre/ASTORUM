import os
import platform
import re
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name : str, sourcedir : str = "") -> None:
        Extension.__init__(self, name, sources = [])
        self.sourcedir = os.path.abspath(sourcedir)



class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ("opt-flags=", "o", "optimisation flags for C++ compiler")
    ]

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        os.chdir("cpp")
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_args, cmake_args = self._generate_args(ext)
            
        if os.getenv("OPT_FLAGS"):
            opt_flags = os.getenv("OPT_FLAGS")
            
        else:
            opt_flags = None
            
        if opt_flags:
            cmake_args += ["-DOPT_FLAGS=" + opt_flags]

        if os.getenv("USE_OMP"):
            cmake_args += ["-DUSE_OMP:STR=" + os.getenv("USE_OMP")]

        env = os.environ.copy()
        
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        build_dir = os.path.join(os.getcwd(), "build")
        os.makedirs(build_dir, exist_ok = True)
        
        subprocess.check_call(
            ["cmake", os.getcwd()] + cmake_args, cwd = build_dir, env = env
        )
        
        subprocess.check_call(
            ["cmake", "--build", os.getcwd(), "--target", "install"] + build_args, cwd = build_dir
        )

    def _generate_args(self, ext):
        cmake_args = ["-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
            
        else:
            # In macOS, gcc/g++ is aliased to clang/clang++.
            gxx = os.getenv("CXX_COMPILER", "g++")
            
            if gxx is None:
                raise RuntimeError(
                    "gcc/g++ must be installed to build the following extensions: "
                    + ", ".join(e.name for e in self.extensions)
                )

            cmake_args += ["-DCMAKE_CXX_COMPILER=" + gxx]
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

            if platform.system() == "Darwin":
                # This is for building Python package on GitHub Actions, whose architecture is x86_64.
                # Without specifying the architecture explicitly,
                # binaries for arm64 is built for x86_64 while cibuildwheel intends to build for arm64.
                archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
                if len(archs) > 0:
                    cmake_args += [
                        "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                    ]

            n_cpus = os.cpu_count()
            build_args += [f"-j{n_cpus}"]

        return build_args, cmake_args

setup(
    name = "BlockWiseSmoothNMF",
    version = "0.1.0",
    author = "Sebastian Cozma",
    author_email = "sebastian.cozma@epfl.ch",
    description = "Block-wise implementaion of the SmoothNMF algorithm in C++.",
    long_description = "",
    ext_modules = [CMakeExtension("BlockWiseSmoothNMF")],
    cmdclass = {"build_ext": CMakeBuild},
    zip_safe = False,
    include_package_data = True,
    package_data = {"databases" : [
        "databases/200keV_xrays_transformed.json",
        "databases/300keV_xrays_transformed.json",
        "databases/interpolated_mass_absorption_coefficients.json",
        "databases/interpolated_SDD_efficiency.txt",
        "databases/periodic_table_number.json",
        "databases/periodic_table_symbols.json",
        "databases/siegbahn_to_iupac.json"
    ]},
    extras_require = {"dev": ["pytest>=6.0"]},
    install_requires = [
        "hyperspy>=2.0.0",
        "exspy",
        "dask",
        "jupyterlab",
        "pybind11"
    ],
    python_requires = ">=3.11"
)