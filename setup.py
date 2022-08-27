from setuptools import setup, find_packages
from typing import List

#Declaring varible  for setup functions
PROJECT_NAME="item-sales-predictor"
VERSION="0.0.2" #Change Version if any new library is installed
AUTHOR="Anurag P Ekka"
DESCRIPTION="Predicts sale of an item in a store."
PACKAGES=["store"]
REQUIREMENTS_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements_list()->List[str]: #List->[str]: Function returns list of strings
    """
    Description: This function is goib=ng to return list of requiremnts 
    mentioned in requirement.txt file

    return this function is going eo reutun a list which contain name
    of libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENTS_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
        #if HYPHEN_E_DOT in requirement_list:
        #    requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list
        
setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    #packages=PACKAGES,
    packages=find_packages(), #find_packages(): returns all the folders 
                              # having __init__.py(developer defined packages)
    install_requires=get_requirements_list()
)

#if __name__=="__main__":
#    get_requirements_list()