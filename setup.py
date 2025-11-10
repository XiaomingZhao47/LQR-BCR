# setup script for LQR Block Cyclic Reduction

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lqr-block-cyclic-reduction",
    version="0.1.0",
    author="Xiaoming Zhao",
    author_email="xiaoming.zhao.gr@dartmouth.edu",
    description="Hermitian Positive Definite Block Cyclic Reduction for LQR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XiaomingZhao47/LQR-BCR",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
        ],
    },
)