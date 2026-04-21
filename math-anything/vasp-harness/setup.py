#!/usr/bin/env python3
"""setup.py for math-anything-vasp

Install with: pip install -e .
"""

from setuptools import find_namespace_packages, setup

setup(
    name="math-anything-vasp",
    version="0.1.0",
    author="Math Anything Contributors",
    author_email="",
    description="VASP harness for Math Anything - Extracts mathematical structures from DFT calculations",
    long_description="""
# Math Anything - VASP Harness

Extracts mathematical structures from VASP (Vienna Ab initio Simulation Package)
density functional theory calculations.

## Features

- **Governing Equations**: Kohn-Sham equations, Hohenberg-Kohn theorem
- **Boundary Conditions**: Periodic boundary conditions, Bloch theorem
- **Numerical Methods**: Plane wave basis, SCF iteration
- **Computational Graph**: Explicit/implicit loop distinction for SCF
- **Electronic Structure**: Eigenvalue problem representation

## Usage

```python
import math_anything as ma

harness = ma.load_harness("vasp")
schema = harness.extract({
    "incar": "INCAR",
    "poscar": "POSCAR",
    "kpoints": "KPOINTS",
    "outcar": "OUTCAR"
})

schema.save("dft_model.json")
```
""",
    long_description_content_type="text/markdown",
    url="https://github.com/math-anything/vasp-harness",
    packages=find_namespace_packages(include=["math_anything.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "math-anything-core>=0.1.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    zip_safe=False,
)
