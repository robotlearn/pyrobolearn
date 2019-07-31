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

Then you will have to compile the code from the ``raisim_wrapper`` folder by typing:

.. code-block:: bash

    mkdir build && cd build
    cmake -DPYBIND11_PYTHON_VERSION=$PYTHON_VERSION -DCMAKE_PREFIX_PATH=$LOCAL_BUILD ..
    make

where ``PYTHON_VERSION=2.7 or 3.*`` and ``LOCAL_BUILD`` is the build directory where we installed the exported cmake
libraries (as described in [2-4]).

Once it has been compiled, you can access to the Python library ``raisim`` in your code with:

.. code-block:: python

    import raisim

    print(dir(raisim))


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