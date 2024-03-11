from setuptools import setup, find_packages

VERSION = "0.0.5"
DESCRIPTION = "GEARLM"
LONG_DESCRIPTION = "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="GEARLM",
    version=VERSION,
    author="Hao Kang",
    author_email="hkang342@gatech.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "AI"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
)
