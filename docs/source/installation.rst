Installation
============

There are 2 ways to install the PyRoboLearn framework.

1. via :ref:`Docker`
2. using a :ref:`Virtual Environment`


.. _Docker:

Docker
-------

At the moment the docker is a self contained Ubuntu image with all the libraries installed. When launched we have access to a Python3.6 interpreter and we can import pyrobolearn directly.
In the future, ROS may be splitted in another container and linked to this one.

1. Install Docker and nvidia-docker

.. code-block:: bash

    sudo apt-get update
    sudo apt install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable # you should replace bionic by your version
    sudo apt update
    sudo apt install docker-ce
    sudo systemctl status docker # check that docker is active

2. Build the image

.. code-block:: bash

    docker build -t pyrobolearn .

3. You can now start the python interpreter with every library already installed

.. code-block:: bash
    
    docker run -p 11311:11311 -v catkin_ws:/pyrobolearn/catkin_ws/ -ti pyrobolearn python3


To open an interactive terminal in the docker image use:

.. code-block:: bash

    docker run -p 11311:11311 -v catkin_ws:/pyrobolearn/catkin_ws/ -ti pyrobolearn /bin/bash


4. If the GPU is not recognized in the interpreter, you can install ``nvidia-docker``

.. code-block:: bash
    
    curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -sL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install nvidia-docker2
    sudo pkill -SIGHUP dockerd

And use:

.. code-block:: bash
    
    nvidia-docker run -p 11311:11311 -v catkin_ws:/pyrobolearn/catkin_ws/ -ti pyrobolearn


.. _Virtual Environment:

Virtual Environment
-------------------

0. Prerequisites: install the following packages on your Ubuntu system

.. code-block:: bash

    sudo apt-get install cmake gfortran


1. First download the ``pip`` Python package manager and create a virtual environment for Python as described in the following link: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/

On Ubuntu, you can install ``pip`` and ``virtualenv`` by typing in the terminal: 

- In Python 2.7:

.. code-block:: bash

    sudo apt install python-pip
    sudo pip install virtualenv

- In Python 3.5:

.. code-block:: bash

    sudo apt install python3-pip
    sudo pip install virtualenv


You can then create the virtual environment by typing:

.. code-block:: bash

    virtualenv -p /usr/bin/python<version> <virtualenv_name>
    # activate the virtual environment
    source <virtualenv_name>/bin/activate

where ``<version>`` is the python version you want to use (select between ``2.7`` or ``3.5``), and ``<virtualenv_name>`` is a name of your choice for the virtual environment. For instance, it can be ``py2.7`` or ``py3.5``.

To deactivate the virtual environment, just type:

.. code-block:: bash

    deactivate


2. clone this repository and install the requirements by executing the ``setup.py``

In Python 2.7:

.. code-block:: bash

    git clone https://github.com/robotlearn/pyrobolearn
    cd pyrobolearn
    pip install numpy cython
    pip install http://github.com/cornellius-gp/gpytorch/archive/alpha.zip  # this is for Python 2.7
    pip install -e .  # this will install pyrobolearn as well as the required packages (so no need for: pip install -r requirements.txt)

In Python 3.5:

.. code-block:: bash

    git clone https://github.com/robotlearn/pyrobolearn
    cd pyrobolearn
    pip install numpy cython
    pip install gpytorch  # this is for Python 3.5
    pip install -e .  # this will install pyrobolearn as well as the required packages (so no need for: pip install -r requirements.txt)


Depending on your computer configuration and the python version you use, you might need to install also the following packages through ``apt-get``:

.. code-block:: bash

    sudo apt install python-tk  # if python 2.7
    sudo apt install python3-tk  # if python 3.5

