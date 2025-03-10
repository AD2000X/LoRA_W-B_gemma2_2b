from setuptools import setup, find_packages

setup(
    name="gemma2-lora-finetune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.14.0",
        "jax>=0.4.23",
        "keras>=3.0.0",
        "keras-nlp>=0.18.0",
        "wandb>=0.19.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "textstat>=0.7.5",
        "nltk>=3.8.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "scikit-learn>=1.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tuning Gemma 2 models using LoRA with Weights & Biases integration",
    keywords="gemma, lora, nlp, fine-tuning, keras, jax",
    url="https://github.com/yourusername/gemma2-lora-finetune",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)