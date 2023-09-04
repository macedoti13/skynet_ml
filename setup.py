import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skynet_ml",
    version="0.0.8",
    author="Thiago Macedo",
    author_email="thialmacedo@gmail.com",
    description="A hand made machine learning library from scratch, by Thiago Macedo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/macedoti13/skynet_ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)