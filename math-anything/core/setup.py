#!/usr/bin/env python3
"""setup.py for math-anything-core

Install with: pip install -e .
"""

from setuptools import setup, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="math-anything-core",
    version="0.1.0",
    author="Math Anything Contributors",
    author_email="",
    description="Math Anything - Mathematical structure extraction for computational materials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/math-anything/core",
    packages=find_namespace_packages(include=["math_anything", "math_anything.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
        ],
        "lammps": [
            "math-anything-lammps",
        ],
    },
    entry_points={
        "console_scripts": [
            "math-anything=math_anything.cli:main",
        ],
    },
    zip_safe=False,
)