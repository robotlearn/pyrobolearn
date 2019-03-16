# PyRoboLearn

This repository contains the code for the *PyRoboLearn* (PRL) framework: a Python framework for Robot Learning.
This framework revolves mainly around 7 axes: simulators, worlds, robots, interfaces, learning tasks (= environment and policy), learning models, and learning algorithms. 

## Requirements

The framework has been tested with Python 2.7 and Ubuntu 16.04.

## Installation

1. First download the `pip` Python package manager and create a virtual environment for Python 2.7 as described in the following link: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
On Ubuntu, in the terminal, you can type: 
```bash
sudo apt install python-pip
sudo pip install virtualenv
```

2. clone this repository and install the requirements and the setup.py
```bash
pip install -r requirements.txt
pip install -e .
```

## Citation

```
@misc{delhaisse2019pyrobolearn,
    author = {Delhaisse, Brian and Rozo, Leonel, and Caldwell, Darwin},
    title = {PyRoboLearn: A Python Framework for Robot Learning Practitioners},
    howpublished = {\url{https://github.com/robotlearn/pyrobolearn}},
    year=2019,
}
```
