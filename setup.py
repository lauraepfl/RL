from setuptools import setup, find_packages

setup(
    name="rl_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "gymnasium>=0.26.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
)
