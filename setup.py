"""Setup script for CADC — Context-Aware Deletion Caller"""
from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="cadc",
    version="1.0.0",
    description="Context-Aware Deletion Caller: CNN + FiLM-conditioned HyenaDNA for genomic deletion detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CADC Contributors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pysam>=0.21.0",
        "pyfaidx>=0.7.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.16.0",
        "einops>=0.6.1",
        "h5py>=3.9.0",
    ],
    entry_points={
        "console_scripts": [
            "cadc-generate=scripts.generate_fused_dataset:main",
            "cadc-precompute=scripts.precompute_hyenadna_embeddings:main",
            "cadc-train=scripts.train_fused_model:main",
            "cadc-train-prior=scripts.train_sequence_prior:main",
            "cadc-call=scripts.call_fused_deletions:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
