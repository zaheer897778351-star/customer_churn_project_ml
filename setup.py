from setuptools import find_packages,setup
from typing import List


dot = "-e ."
def get_requriments(filepath:str)->List[str]:
    '''
    This gives required librays to install
    '''
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n","") for i in requirements]
        if dot in requirements:
            requirements.remove(dot)

    return requirements  


setup(
    name="customer churn project",
    version='0.0.1',
    author="Zaheer Ahmad",
    author_email="zaheer897778351@gmail.com",
    packages= find_packages(),
    install_requires= get_requriments('req.txt')
)