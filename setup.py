#!/usr/bin/env python3
"""
Setup script for Axobrier Python package.
Supports editable installation: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="axobrier",
    version="0.2.0",
    description="Axobrier: Deterministic System 1 Choice Routing Engine for Autonomous AI Agents",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "axobrier=axobrier.cli:main",
        ],
    },
)
