from setuptools import setup, find_packages
import os

setup(
    name='ThinkGrasp',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    description="ThinkGrasp",
    author='Yaoyao Qian',
    author_email='qian.ya@northeastern.edu',
    install_requires=[line for line in open('requirements.txt').readlines() if "@" not in line],
)
