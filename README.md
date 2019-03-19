# PyRoboLearn

This repository contains the code for the *PyRoboLearn* (PRL) framework: a Python framework for Robot Learning.
This framework revolves mainly around 7 axes: simulators, worlds, robots, interfaces, learning tasks (= environment and policy), learning models, and learning algorithms. 

## Requirements

The framework has been tested with Python 2.7 and Ubuntu 16.04 and 18.04. We plan to migrate soon to Python 3.5.

## Installation

1. First download the `pip` Python package manager and create a virtual environment for Python 2.7 as described in the following link: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
On Ubuntu, in the terminal, you can type to download and install `pip` and `virtualenv`: 
```bash
sudo apt install python-pip
sudo pip install virtualenv
```

You can then create the virtual environment by typing:
```bash
virtualenv -p /usr/bin/python2.7 <virtualenv_name>
# activate the virtual environment
source <virtualenv_name>/bin/activate
```
where `<virtualenv_name>` is a name of your choice for the virtual environment. For instance, it can be `py2.7`.

To deactivate the virtual environment, just type:
```bash
deactivate
```

2. clone this repository and install the requirements and the setup.py
```bash
git clone https://github.com/robotlearn/pyrobolearn
cd pyrobolearn
pip install numpy cython
pip install http://github.com/cornellius-gp/gpytorch/archive/alpha.zip
pip install -e .  # this will install pyrobolearn as well as all the required packages (so no need for: pip install -r requirements.txt)
```

## How to use it?

Check the `README.md` file in the `examples` folder.

## Citation

```
@misc{delhaisse2019pyrobolearn,
    author = {Delhaisse, Brian and Rozo, Leonel, and Caldwell, Darwin},
    title = {PyRoboLearn: A Python Framework for Robot Learning Practitioners},
    howpublished = {\url{https://github.com/robotlearn/pyrobolearn}},
    year=2019,
}
```
