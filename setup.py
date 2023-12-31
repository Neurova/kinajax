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

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="kinajax",  # required
    version="1.0.0",  # required
    author="Jacob Buffa", # optional
    author_email="jacobbuffa10@gmail.com", #optional
    maintainer = "Jacob Buffa", #optional
    description="Package used for Evaluating Rigid Body Dynamics",  # optional
    packages=['kinajax'],
    long_description=long_description,  # optional
    long_description_content_type="text/markdown",  # 0ptional (see note above)
    install_requires = [
        "numpy", "scipy", "pandas", "jax", "pyodbc", "requests", "pyyaml"
    ]
)
