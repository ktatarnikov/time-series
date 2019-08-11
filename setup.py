import os
import sys

from setuptools import find_packages, setup


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_version():
    with open(get_script_path() + '/VERSION', 'r') as f:
        return f.read()


def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
        return requirements


setup(name="time-series",
      version=get_version(),
      packages=find_packages(),
      install_requires=get_requirements(),
      package_data={
          '': ['*.yaml'],
      },
      data_files=[("", ["VERSION"])],
      entry_points={},
      author="ktatarnikov",
      author_email="ktatarnikov@gmail.com",
      description="TimeSeries",
      license="Apache2",
      keywords="time series")
