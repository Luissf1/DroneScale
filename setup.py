"""
Setup script for the fuzzy-adaptive-quadrotor-pid package.
"""

from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent

# Read README (safe fallback if missing)
README_PATH = HERE / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

# Read requirements (safe fallback if missing)
REQ_PATH = HERE / "requirements.txt"
REQUIREMENTS = (
    REQ_PATH.read_text(encoding="utf-8").splitlines()
    if REQ_PATH.exists()
    else ["numpy", "scipy", "matplotlib", "pandas", "seaborn", "tqdm", "pyyaml"]
)

setup(
    name="fuzzy-adaptive-quadrotor-pid",
    version="1.0.0",
    description=(
        "Fuzzy adaptive scaling of PSO-optimized PID controllers for quadrotor UAVs"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Luis Adrian Silva Reyes",
    author_email="luis.silva19@tectijuana.edu.mx",
    url="https://github.com/yourusername/fuzzy-adaptive-quadrotor-pid",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[r for r in REQUIREMENTS if r and not r.startswith("#")],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)