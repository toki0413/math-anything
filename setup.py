#!/usr/bin/env python3
"""setup.py for math-anything

A mathematical semantic layer for computational materials science.
Extract equations, constraints, and relationships from simulation inputs.

Install with: pip install math-anything
"""

from setuptools import setup, find_namespace_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Math Anything - Mathematical structure extraction for computational materials science"

setup(
    name="math-anything",
    version="1.0.0",
    author="toki",
    author_email="",
    description="Mathematical structure extraction for computational materials science engines (VASP, LAMMPS, Abaqus, Ansys, COMSOL, GROMACS, Multiwfn)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/crested-ibis-0413/math-anything",  # 国内镜像，GitHub: https://github.com/toki0413/math-anything
    packages=find_namespace_packages(
        include=[
            "math_anything",
            "math_anything.*",
        ],
        where="math-anything/core",
    ),
    package_dir={"": "math-anything/core"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[
        "click>=8.0.0",
        "prompt-toolkit>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "mcp": [
            "mcp>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "math-anything=math_anything.cli:main",
        ],
    },
    zip_safe=False,
    project_urls={
        # 国内镜像地址
        "Bug Reports": "https://gitee.com/crested-ibis-0413/math-anything/issues",
        "Source": "https://gitee.com/crested-ibis-0413/math-anything",
        "Documentation": "https://gitee.com/crested-ibis-0413/math-anything#readme",
        # GitHub 镜像地址
        "GitHub Issues": "https://github.com/toki0413/math-anything/issues",
        "GitHub Source": "https://github.com/toki0413/math-anything",
    },
)