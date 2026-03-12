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
            
    def find_highest_gcc_version(self):
        # Potential paths where g++ might be installed
        possible_paths = [
            "/opt/homebrew/bin",                # Apple Silicon (M1/M2)
            "/usr/local/bin",                   # macOS Intel (default for Homebrew)
            "/home/linuxbrew/.linuxbrew/bin",   # Linuxbrew on Linux
            "/usr/bin",                         # Default for Linux (apt, yum, etc.)
        ]
        
        gxx_versions = []
        
        # Search for all installed g++ versions in possible paths
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # List all g++ executables in the directory
                    output = subprocess.check_output([f"ls {path}/g++-*"], shell=True, universal_newlines=True)
                    # Find versions that match g++-<version>
                    gxx_versions += re.findall(r"g\+\+-(\d+)", output)
                except subprocess.CalledProcessError:
                    # Path exists but no g++-* files found
                    continue

        if gxx_versions:
            # Convert to integers and sort in descending order to find the highest version
            gxx_versions = sorted(map(int, gxx_versions), reverse=True)
            # Return the highest version of g++
            return f"g++-{gxx_versions[0]}"
        else:
            return None

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
            ["cmake", "-S", os.getcwd(), "-B", build_dir] + cmake_args, cwd = os.getcwd(), env = env
        )
        
        subprocess.check_call(
            ["cmake", "--build", build_dir, "--target", "install"] + build_args, cwd = os.getcwd()
        )

    def _generate_args(self, ext):
        cmake_args = ["-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            n_cpus = os.cpu_count()
            build_args += ["--parallel", f"{n_cpus}"]
            
        else:
            gxx = os.getenv("CXX_COMPILER", "g++")
            
            if platform.system() == "Darwin":
                archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
                
                if len(archs) > 0:
                    cmake_args += [
                        "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                    ]
                    
                gxx = self.find_highest_gcc_version() or "g++"
                
            if gxx is None:
                raise RuntimeError("No suitable g++ version found.")

            n_cpus = os.cpu_count()
            build_args += ["--", f"-j{n_cpus}"]
            cmake_args += ["-DCMAKE_CXX_COMPILER=" + gxx]
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        return build_args, cmake_args

setup(
    name = "astorum",
    version = "0.2.3",
    author = "Sebastian Cozma",
    author_email = "sebastian.cozma@epfl.ch",
    description = "Analytical STEM Out-of-core Resource for Unified Multimodal data",
    license = "GPLv3",
    license_files = ("LICENSE"),
    long_description = "",
    packages = ['astorum'],
    ext_modules = [CMakeExtension("core")],
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
