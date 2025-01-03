import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Biomedical Augmentation for Text Package",
    version="0.0.1",
    author="LauraBergomi",
    author_email="laura.bergomi01@universitadipavia.it",
    description="A Biomedical domain-specific Text Augmentation tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bmi-labmedinfo/...",
    packages=setuptools.find_packages(),
    license="",
    classifiers=[
        "License :: Free for non-commercial use",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent"
    ],
    install_requires=["numpy", "scikit-learn", "torch",...],
    extras_require={
        "dev": ...
    },
    python_requires='>=3.8'
)