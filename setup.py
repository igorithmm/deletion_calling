"""Setup script for DeepSV"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="deepsv",
    version="2.5.0",
    description="Deep Learning-based Structural Variant Calling for Long Deletions (DeepSV 2.5)",
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
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
        "umap-learn>=0.5.3",
        "joblib>=1.3.0",
        "huggingface-hub>=0.16.0",
        "einops>=0.6.1",
    ],
    entry_points={
        "console_scripts": [
            "deepsv-generate-images=scripts.generate_training_images:main",
            "deepsv-train=scripts.train_model:main",
            "deepsv-call=scripts.call_deletions:main",
            "deepsv-generate-image-tensor=scripts.generate_image_tensor_dataset:main",
            "deepsv-train-image-tensor=scripts.train_image_tensor_model:main",
            "deepsv-generate-alignment-tensor=scripts.generate_tensor_dataset:main",
            "deepsv-train-alignment-tensor=scripts.train_tensor_model:main",
            "deepsv-download-dnabert2=scripts.download_dnabert2:main",
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

