PyRoboLearn
===========

PyRoboLearn is a Python framework in robot learning for education and research. PyRoboLearn is meant to be a free and open-source tool.

Goal


Problem formulation
-------------------

General idea.

- lack of benchmarks
- lack of flexibility and modularity
- lack of generalization
- high coupling

For instance:

Full example.


Main idea of PyRoboLearn and solution to above problem.


Hardware/Software requirements
------------------------------

The PyRoboLearn framework has been tested on Ubuntu 16.04 and 18.04, with Python 2.7, 3.5 and 3.6.


Design Decisions
----------------

While designing PRL, we focused on the five following features:

- modularity: design a module (i.e. class) for each different concept
- abstraction: add a layer of abstraction for combination of low-level modules
- reusability: easy to reuse the different modules and to combine them
- low coupling between the different modules
- flexibility: this is mainly achieved by favoring composition over inheritance.


The Python language has been selected.


.. figure:: ../UML/pyrobolearn_uml.png
	:alt: UML diagram of PyRoboLearn
	:align: center

	UML diagram of PyRoboLearn