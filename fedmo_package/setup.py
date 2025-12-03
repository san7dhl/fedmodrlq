"""
FedMO-DRLQ Package Setup
========================
Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="fedmo_drlq",
    version="0.1.0",
    author="Sandhya",
    author_email="sandhya@nitsikkim.ac.in",
    description="Federated Multi-Objective Deep Reinforcement Learning for Quantum Cloud Computing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sandhya/fedmo-drlq",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "gymnasium>=0.28.0",
        "simpy>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "tensorboard>=2.9.0",
        ],
    },
)
