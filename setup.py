import os
import sys
from setuptools import setup, find_packages

def get_script_path():
  return os.path.dirname(os.path.realpath(sys.argv[0]))

def get_version():
  with open(get_script_path() + '/VERSION', 'r') as f:
    return f.read()

setup(
  name="time-series",
  version=get_version(),
  packages=find_packages(),

  install_requires=[
      'mock==2.0.0',
      'responses==0.9.0',
      'tabulate==0.8.2',
      'Requests==2.20.0',
      'PyYAML==4.2b1',
      'yamale==1.7.1',
      'pytest',
      'keras==2.2.4',
      'pandas==0.24.1',
      'scikit-learn==0.20.2',
      'matplotlib==3.0.2',
      'tensorflow==1.12.0'
  ],

  package_data={
      '': ['*.yaml'],
  },

  entry_points={
  },

  author="ktatarnikov",
  author_email="ktatarnikov@gmail.com",
  description="TimeSeries",
  license="Apache2",
  keywords="time series"
)
