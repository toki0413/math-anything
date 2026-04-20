#!/usr/bin/env python3
"""setup.py for math-anything-lammps

Install with: pip install -e .
"""

from setuptools import setup, find_namespace_packages

setup(
    name="math-anything-lammps",
    version="0.1.0",
    author="Math Anything Contributors",
    author_email="",
    description="LAMMPS harness for Math Anything - Extracts mathematical structures from LAMMPS simulations",
    long_description="""
# Math Anything - LAMMPS Harness

Extracts mathematical structures (governing equations, boundary conditions,
numerical methods, computational graphs) from LAMMPS molecular dynamics
simulations.

## Features

- Governing Equations: Newton's laws, Hamiltonian dynamics
- Boundary Conditions: Periodic, fixed, deform (with tensor-complete expression)
- Numerical Methods: Velocity Verlet, thermostat integrators
- Computational Graph: Explicit/implicit loop distinction
- Conservation Properties: Energy, momentum, angular momentum tracking

## Usage

```python
import math_anything as ma

harness = ma.load_harness("lammps")
schema = harness.extract({
    "input": "in.deform",
    "log": "log.lammps"
})

schema.save("model.json")
```
""",
    long_description_content_type="text/markdown",
    url="https://github.com/math-anything/lammps-harness",
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
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    zip_safe=False,
)