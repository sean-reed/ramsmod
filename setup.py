from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
  name = 'ramsmod',
  packages = find_packages(exclude=['*.tests',]),
  version = '0.6',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Python 3 library for reliability data analysis',   # Give a short description about your library
  author = 'Sean Reed',                   # Type in your name
  author_email = 'sean@sean-reed.com',      # Type in your E-Mail
  url = 'https://github.com/user/sean-reed',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/sean-reed/ramsmod/archive/v0.6.tar.gz',
  install_requires= requirements,
  classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)