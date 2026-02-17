"""Setup script for DeepSV"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="deepsv",
    version="2.0.0",
    description="Deep Learning-based Structural Variant Calling for Long Deletions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DeepSV Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pysam>=0.21.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    entry_points={
        "console_scripts": [
            "deepsv-generate-images=scripts.generate_training_images:main",
            "deepsv-train=scripts.train_model:main",
            "deepsv-call=scripts.call_deletions:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

