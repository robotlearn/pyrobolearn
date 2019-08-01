Python wrapper for RaiSim
=========================

This folder contains a python wrapper around RaiSim (``raisimLib`` and ``raisimOgre``) using ``pybind11``.

Parts of the wrappers were taken and modified from (or inspired by) the code given in the ``raisimGym/raisim_gym/env/``
folder. If you use these wrappers in PRL, please acknowledge their contribution as well by citing [1-4].


How to use the wrappers?
~~~~~~~~~~~~~~~~~~~~~~~~

In order to use the wrappers, you will have to install at least
`raisimLib <https://github.com/leggedrobotics/raisimLib>`_ and
`raisimOgre <https://github.com/leggedrobotics/raisimOgre>`_. You will also have to install
`pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ as we use this to wrap the C++ code.

If you followed the installation procedure of ``raisimLib`` and/or ``raisimOgre``, you will have the two following
environment variables defined:

- WORKSPACE: workspace where you clone your git repos (e.g., ~/raisim_workspace)
- LOCAL_BUILD: build directory where you install exported cmake libraries (e.g., ~/raisim_build)


Before compiling the code in this repo, you will have to move or copy the `extras` folder (that you can find in this
repo) in the `$LOCAL_BUILD/include/ode/` folder. This `extras` folder contains some missing header files for ODE which
are necessary in order, for instance, to load meshes with Raisim. This can be done by:

.. code-block:: bash

    cp -r extras $LOCAL_BUILD/include/ode/


Now, you can finally compile the python wrappers from the ``raisim_wrapper`` folder by typing:

.. code-block:: bash

    mkdir build && cd build
    cmake -DPYBIND11_PYTHON_VERSION=$PYTHON_VERSION -DCMAKE_PREFIX_PATH=$LOCAL_BUILD ..
    make

where ``$PYTHON_VERSION`` is the Python version you wish to use. For instance, ``PYTHON_VERSION=3.5``.


Once it has been compiled, you can access to the Python library ``raisim`` in your code with:

.. code-block:: python

    import raisim

    print(dir(raisim))


We follow mostly the naming convention defined in ``raisimLib`` and ``raisimOgre``, however we follow the PEP8 guideline.
Thus, a C++ method like:

.. code-block:: cpp

    getComPosition()

becomes

.. code-block:: python

    get_com_position()


Note that in the original ``raisimLib``, the authors sometimes use their own defined data types for vectors and
matrices (such as ``Vec<n>``, ``Mat<n,m>``, ``VecDyn``, and ``MatDyn``). When using the python wrappers, these
datatypes are converted back and forth to numpy arrays as this is the standard in Python.


References
~~~~~~~~~~

- [1] "Per-contact iteration method for solving contact dynamics", Hwangbo et al., 2018
- [2] raisimLib: https://github.com/leggedrobotics/raisimLib
- [3] raisimOgre: https://github.com/leggedrobotics/raisimOgre
- [4] raisimGym: https://github.com/leggedrobotics/raisimGym
- [5] pybind11: https://pybind11.readthedocs.io/en/stable/


Troubleshooting
~~~~~~~~~~~~~~~

- ``fatal error: Eigen/*: No such file or directory``
    - If you have Eigen3 installed on your system, you probably have to replace all the ``#include <Eigen/*>`` by
      ``#include <eigen3/Eigen/*>``. You can create symlinks to solve this issue:

    .. code-block:: bash

        cd /usr/local/include
        sudo ln -sf eigen3/Eigen Eigen
        sudo ln -sf eigen3/unsupported unsupported

    or you can replace the ``#include <Eigen/*>`` by ``#include <eigen3/Eigen/*>``.


Citation
~~~~~~~~

If the code presented here was useful to you, we would appreciate if you could cite the original authors:

.. code-block:: latex

    @article{hwangbo2018per,
        title={Per-contact iteration method for solving contact dynamics},
        author={Hwangbo, Jemin and Lee, Joonho and Hutter, Marco},
        journal={IEEE Robotics and Automation Letters},
        volume={3},
        number={2},
        pages={895--902},
        year={2018},
        publisher={IEEE}
    }


If you still have some space in your paper for the references, you can add the following citation:

.. code-block::

    @misc{delhaisse2019raisimpy
        author = {Delhaisse, Brian},
    	title = {RaiSimPy: A Python wrapper for RaiSim},
    	howpublished = {\url{https://github.com/robotlearn/raisimpy}},
    	year=2019,
	}

Otherwise, you can just add me in the acknowledgements ;)

If you use ``raisimpy`` through the `pyrobolearn <https://github.com/robotlearn/pyrobolearn>`_ framework, you can cite
this last one instead (but you still have to cite the authors of Raisim).
