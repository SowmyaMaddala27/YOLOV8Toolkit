# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 22:33:08 2023

@author: sowmya
"""

from setuptools import find_packages, setup

# from typing import List

def get_requirements(file_path):
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
      name='YOLOv8Toolkit',
      version='0.0.1',
      author='Sowmya',
      author_email='sowmyamaddala5@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt'))

# pip install -r requirements.txt in source dir
