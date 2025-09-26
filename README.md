# EDDDe
Electron Density Derived Descriptors for molecular learning and property prediction

# Installation

Git repo
    git clone --recurse-submodules PROJECT_URL

Install python env:
    cd EDDDe/
    pipenv install

Download welQrate:
    https://vanderbilt.app.box.com/v/WelQrate-Datasets/folder/270082782052
    Download the folder and move it to data/qsar/

**This is currently broken**
Install gau2grid:
    cd EDDDe/
    pipenv shell
    cd ElektroNN/gau2grid
    python setup.py install
    exit