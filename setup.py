# Update: Xinyu Yan, 2025-06-21
# TL;DR: This file is used to setup the project

from setuptools import setup, find_packages


# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Oblivionis",
    version="0.0.1",
    author="Fuyao Zhang, Xinyu Yan",
    author_email="hi.fyzhang@gmail.com, Tristan_Yan@outlook.com",
    description="A Unified Framework for Large Language Model Unlearning on Private Data via Federated Learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fyzhang1/FedLLMUnlearning",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,  # Uses requirements.txt
    extras_require={
        "lm-eval": [
            "lm-eval==0.4.8",
        ],  # Install using `pip install .[lm-eval]`
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install .[dev]`
    },
    python_requires=">=3.11",
)
