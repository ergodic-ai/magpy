from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="causal-magpy",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    author="ergodic.ai",
    author_email="andre@ergodic.ai",
    description="A package for causal discovery and causal inference algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ergodic-ai/magpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
