from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """This function will return the list of requirements from requirement.txt file"""

    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [requ.replace("\n","") for requ in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Keshav",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)