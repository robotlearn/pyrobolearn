# DMPyTorch

This repository contains the code for the *DMPyTorch* library; a PyTorch library for Dynamic Movement Primitives. If you want to use with robots you can have a look at the [`pyrobolearn` framework](https://github.com/robotlearn/pyrobolearn)

**Warning**: The development of this framework is ongoing, and thus some substantial changes might occur. Sorry for the inconvenience.


## Requirements

The framework has been tested with Python 2.7 and 3.5 on Ubuntu 16.04 and 18.04.


## Installation

1. First download the `pip` Python package manager and create a virtual environment for Python as described in the following link: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
On Ubuntu, you can install `pip` and `virtualenv` by typing in the terminal: 

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
where `<version>` is the python version you want to use (select between `2.7` or `3.5`), and `<virtualenv_name>` is a name of your choice for the virtual environment. For instance, it can be `py2.7` or `py3.5`.

To deactivate the virtual environment, just type:
```bash
deactivate
```

2. clone this repository and install the requirements by executing the setup.py

In Python 2.7 or 3.5:
```bash
git clone https://github.com/robotlearn/dmpytorch
cd dmpytorch
pip install numpy
pip install -e .  # this will install dmpytorch as well as the required packages (so no need for: pip install -r requirements.txt)
```

Depending on your computer configuration and the python version you use, you might need to install also the following packages through `apt-get`:
```bash
sudo apt install python-tk  # if python 2.7
sudo apt install python3-tk  # if python 3.5
```

## How to use it?

Check the `README.md` file in the `examples` folder.

## Citation

If you use `dmpytorch`, please cite:
```
@misc{delhaisse2019dmpytorch,
    author = {Delhaisse, Brian and Rozo, Leonel, and Caldwell, Darwin},
    title = {DMPyTorch: a PyTorch Library for Dynamic Movement Primitives},
    howpublished = {\url{https://github.com/robotlearn/dmpytorch}},
    year=2019,
}
```

## Acknowledgements

Parts of the code were inspired by the work from [Travis DeWolf](https://github.com/studywolf) and his library [`pydmps`](https://github.com/studywolf/pydmps). His blog which explains DMPs pretty well can be found [here](https://studywolf.wordpress.com/category/robotics/dynamic-movement-primitive/).


## References

1. "Dynamical movement primitives: Learning attractor models for motor behaviors", Ijspeert et al., 2013
2. PyDMPs (from DeWolf, 2013): https://github.com/studywolf/pydmps
3. "Biologically-inspired Dynamical Systems for Movement Generation: Automatic Real-time Goal Adaptation and Obstacle Avoidance", Hoffmann et al., 2009
4. "Policy Search for Motor Primitives in Robotics", Kober et al., 2010


## TODO

- [ ] implement "Orientation in Cartesian Space Dynamic Movement Primitives", Ude et al., 2014
- [ ] implement "Action Sequencing using Dynamic Movement Primitives", Nemec et al., 2011
- [ ] implement "A Generalized Path Integral Control Approach to Reinforcement Learning", Theodorou et al., 2010
- [ ] test DMP with NN

