'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-05-20 23:54:25
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-21 16:18:17
Description: 
'''
"""
something is from https://github.com/pypa/sampleproject
thanks to https://zetcode.com/python/package/
"""

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

requires = [
    "oneflow >= 0.8.0",
]

setup(
    name = "bann_flow",
    version = "0.0.1",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        # "Programming Language :: Python :: 3.11",# wait to numpy
        # "Programming Language :: Python :: 3 :: Only",
    ],
        
    keywords = ["mbapy", "deeplearning", "oneflow"],
    description = "basic for all in neural network for oneflow",
    long_description = long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7, <3.11",
    license = "MIT Licence",

    url = "https://github.com/BHM-Bob/bann_flow",
    author = "BHM-Bob G",
    author_email = "bhmfly@foxmail.com",
    
    packages=["bann_flow", "bann_flow/paper"],
    
    include_package_data = True,
    platforms = "any",
    
    install_requires=requires,
)

# pip install .

# or

# python setup.py sdist
# twine upload dist/bann_flow-0.0.1.tar.gz
# pip install --upgrade bann_flow