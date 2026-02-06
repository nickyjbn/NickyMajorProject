from setuptools import setup, find_packages

setup(
    name="ai-training-memory-system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    author="AI Training Memory System Team",
    description="Memory-augmented AI system for continuous learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
