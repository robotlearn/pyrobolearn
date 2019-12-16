## Python Tutorials

Hey! So you want to learn Python? Good :thumbsup: you are at the right place! :smile: 

Here is what you have to know:

- There are mainly 2 versions of Python: Python 2 and Python 3. The most known version of the former is Python 2.7, and is currently installed by default on Linux and MacOSX systems. You can try it by typing `python` in the terminal. To exit, push `Ctrl+D` or type `exit()`. Note that Python 2.7 will stop to be maintained in 2020. Now, I am pretty sure that you are asking yourself "what is the difference between Python 2.7 and Python 3?". Well, to be short "Python 2.x is legacy, Python 3.x is the present and future of the language" as mentioned [here](https://wiki.python.org/moin/Python2orPython3). However, if you are using ROS/Gazebo, I would still recommend to use Python 2.7 for now. Pratically, these 2 versions are similar except for few details, the `print` statements where you don't have to put parenthesis in Python 2.7, and for few specific libraries. Personally, most of my code works in Python 3 and Python 2.7.

- Check first this ["Tutorial: Learn Python in 10 Minutes"](https://www.stavros.io/tutorials/python/) to get a general understanding of the language. It assumes that you already know at least one programming language such as C++, Java or Matlab. Otherwise, it will take you more than 10minutes :stuck_out_tongue_winking_eye:

- Then check this [one](https://learnxinyminutes.com/docs/python/) for Python 2.7 or this [one](https://learnxinyminutes.com/docs/python3/) for Python 3.

- Well now, you just have to practice! :grin:

## About Libraries/Modules

The libraries (aka modules) that are useful for roboticists and/or machine learning engineers are:
* `numpy` for vectors, matrices, and tensors. Here is the [tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html).
* `matplotlib` for plotting.
    * check [here](https://matplotlib.org/gallery.html) to see the gallery (basically it shows you what you can do with it).
    * check [this tutorial](https://matplotlib.org/users/beginner.html) to learn on how to use it. The first link `Pyplot tutorial` should be enough to have a basic understanding.
        * if you are interested by 3D plots, check this [one](https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)
        * and if you want animations, look at this [blog (and the webpages it links to)](https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/)
* `scipy` for scientific and mathematical tools such as optimization, interpolation, signal processing, etc. You can find the tutorials at the following [link](https://docs.scipy.org/doc/scipy/reference/tutorial/).
    * If you are not satisfied with `scipy.optimize.minimize`, you can have a look at [`cvxpy`](http://www.cvxpy.org/en/latest/) (which uses different solvers such as [`cvxopt`](http://cvxopt.org/)) for convex optimization. For QP problems with cvxopt, you can have a look at this small [tutorial](https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf). For general nonlinear optimization, you can check [`nlopt`](https://nlopt.readthedocs.io/en/latest/), or [`ipopt`](https://pypi.python.org/pypi/ipopt) with the corresponding [documentation](http://pythonhosted.org/ipopt/). Most of these softwares are written in C/C++/Fortran but provides python wrappers to call the various functions.
* `sklearn` for machine learning algorithms. You can have a look at this library with the tutorials and documentations at the following [link](http://scikit-learn.org/stable/).
* `pandas` when working with tables and data structures. It also provides you data analysis tools. Check [here](https://pandas.pydata.org/pandas-docs/stable/10min.html) for a "10min" tutorial.

More specific libraries:
* Deep Learning (or if you want to work with Tensors, or use automatic derivation tools)
    * `TensorFlow` (aka `TF`) which can be found [here](https://www.tensorflow.org/).
    * `PyTorch` at the following [webpage](http://pytorch.org/).
        * Difference between `PyTorch` and `TF`? You can have a look at this [blog](https://awni.github.io/pytorch-tensorflow/) and this [medium post](https://medium.com/towards-data-science/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b). Quickly, TF is being developed by Google, is more stable, has a bigger community, deals very nicely with the hardware part, has a syntax close to numpy, and is the current tool to use if you want to develop a software product. Meanwhile, PyTorch has a better integration with Python, allows you to use dynamic graphs (instead of static ones as in TF) and experiment new ideas faster. Hovewer, it is still in its early phase and the syntax used is currently different from numpy.
    * `Keras`: this library is in the process of being integrated into TensorFlow. Quick description: it provides higher functionalities and is used on top of Theano or TensorFlow.
    * `Theano`: this library is no longer maintained.
* Gaussian Processes
    * `GPy`: the repo can be found [here](https://github.com/SheffieldML/GPy). Note that `sklearn` also allows you to use basic GPs but it is not as complete as `GPy`. For tutorials, have a look [here](http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb)
* Reinforcement Learning
    * RL Environments
        * `OpenAI-Gym`: the repo is [here](https://github.com/openai/gym), the documentation can be found [here](https://gym.openai.com/docs/), and the environments are [here](https://gym.openai.com/envs/).
* Dynamic Movement Primitives
    * `pydmps`: the repo can be found [here](https://github.com/studywolf/pydmps) and the tutorials (which are very nice) are on this [blog](https://studywolf.wordpress.com/category/robotics/dynamic-movement-primitive/).
* `rospy`: this library allows you to use [ROS](http://www.ros.org/) (the Robot Operating System). Check [here](http://wiki.ros.org/rospy) and [here](http://wiki.ros.org/rospy_tutorials) for the tutorials.

C/C++ libraries with Python support:
* `OpenCV`: library for computer vision.
* `KDL`: kinematics and dynamics library.
* `RBDL`: rigid body dynamics library.

## Nice tools

Here are nice tools that you should know:
* [`Jupyter Notebook`](http://jupyter.org/). As mentioned on their webpage, "The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text".
* [`pip`](https://pip.pypa.io/en/stable/) which is tool that allows you to install Python packages/modules/libraries. For instance, to install `numpy` you would type in the terminal: `pip install numpy`.
* [`Anaconda`](https://www.anaconda.com/) which allows you to create virtual environments and install packages like `pip`.
* [`Cython`](http://cython.org/): this one is not really a tool, but allows you to wrap C/C++ classes and functions, and use them in Python. You could also use it to write `Cython` code which would be faster than pure Python code. Having said that, it takes quite a bit of time to learn this language, and effectively use it.

## For Matlab Users

So you want to move from Matlab? Good, you are definitely at the right place :wink:
Here are few webpages that could be useful for you:
* The following [link](http://www.pyzo.org/python_vs_matlab.html) explains the difference between Matlab and Python. I would also add that Python has a bigger community (because it is free), and is becoming **the** programming language to use for machine learning.
* Numpy for Matlab Users: [Link1](http://mathesaurus.sourceforge.net/matlab-numpy.html) and [Link2](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html).

That's all folks! :clap:
