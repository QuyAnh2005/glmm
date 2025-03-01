"""Setup configuration for glmm package."""

from setuptools import setup, find_packages

setup(
    name="glmm",
    version="0.1.0",
    description="Bayesian Generalized Linear Mixed Models for Portfolio Credit Risk",
    author="Quy-Anh Dang",
    author_email="dangquyanh150101@gmail.com",
    packages=find_packages(include=["glmm", "glmm.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
