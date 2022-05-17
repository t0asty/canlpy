from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

requirements = [
    "pytorch-lightning==0.8.5",
    "pandas",
    "spacy",
    "torch",
]

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="canlpy",
    version="0.0.1",
    description="A Python Knowledge Enhanced LM library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Squalene/canlpy",
    classifiers=[],
    author="EPFL NLP Lab",
    author_email="antoine.masanet@epfl.ch, julian.schnitzler@epfl.ch",
    license="Apache License 2.0",
    keywords="natural-language-processing nlp",
    packages=find_packages(exclude=["examples", "experiments"]),
    install_requires=requirements,
    python_requires=">=3.6.9",
    # $ pip install -e .[dev,test]
    # extras_require={
    #     "dev": ["pytest", "flake8", "black", "mypy"],
    #     "test": ["pytest"],
    # },
    package_data={},
    include_package_data=True,
    data_files=[],
    entry_points={},
)