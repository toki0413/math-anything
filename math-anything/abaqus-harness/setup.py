#!/usr/bin/env python3
"""setup.py for math-anything-abaqus"""

from setuptools import setup, find_namespace_packages

setup(
    name="math-anything-abaqus",
    version="0.1.0",
    author="Math Anything Contributors",
    description="Abaqus harness for Math Anything",
    packages=find_namespace_packages(include=["math_anything.*"]),
    python_requires=">=3.10",
    install_requires=["math-anything-core>=0.1.0"],
)