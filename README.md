# PyRoboLearn

This repository contains the code for the *PyRoboLearn* (PRL) framework: a Python framework for Robot Learning.
This framework revolves mainly around 7 axes: simulators, worlds, robots, interfaces, learning tasks (= environment and policy), learning models, and learning algorithms. 

**Warning**: The development of this framework is ongoing, and thus some substantial changes might occur. Sorry for the inconvenience.


## Requirements

The framework has been tested with Python 2.7 and Ubuntu 16.04 and 18.04. We also tested parts of it with Python 3.5 on Ubuntu 16.04 and so far so good, but there might be some errors that escaped my attention.


## Installation

1. First download the `pip` Python package manager and create a virtual environment for Python as described in the following link: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
On Ubuntu, in the terminal, you can type to download and install `pip` and `virtualenv`: 

- In Python 2.7:
```bash
sudo apt install python-pip
sudo pip install virtualenv
```

- In Python 3.5:
```bash
sudo apt install python3-pip
sudo pip install virtualenv
```

You can then create the virtual environment by typing:
```bash
virtualenv -p /usr/bin/python<version> <virtualenv_name>
# activate the virtual environment
source <virtualenv_name>/bin/activate
```
where `<version>` is the python version you want to use (select between `2.7` or `3.5`), and `<virtualenv_name>` is a name of your choice for the virtual environment. For instance, it can be `py2.7` or `py3.7`.

To deactivate the virtual environment, just type:
```bash
deactivate
```

2. clone this repository and install the requirements and the setup.py

In Python 2.7:
```bash
git clone https://github.com/robotlearn/pyrobolearn
cd pyrobolearn
pip install numpy cython
pip install http://github.com/cornellius-gp/gpytorch/archive/alpha.zip  # this is for Python 2.7
pip install -e .  # this will install pyrobolearn as well as the required packages (so no need for: pip install -r requirements.txt)
```

In Python 3.5:
```bash
git clone https://github.com/robotlearn/pyrobolearn
cd pyrobolearn
pip install numpy cython
pip install gpytorch  # this is for Python 3.5
pip install -e .  # this will install pyrobolearn as well as the required packages (so no need for: pip install -r requirements.txt)
```

Depending on your computer configuration and the python version you use, you might need to install also the following packages through `apt-get`:
```bash
sudo apt-get install python-tk  # if python 2.7
sudo apt-get install pytho3-tk  # if python 3.5
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


## Acknowledgements

Currently, we mainly use the PyBullet simulator. 
- *PyBullet, a Python module for physcis simulation for games, robotics and machine learning*,
Erwin Coumans and Yunfei Bai, 2016-2019
- references for each robot, model, and others can be found in the corresponding class documentation

