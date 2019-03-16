#!/usr/bin/env python
"""Define the Model abstract class from which all learning models inherit from.

Dependencies: None
"""


from abc import ABCMeta, abstractmethod

__author__ = "Brian Delhaisse"
__copyright__ = "Copyright 2018, PyRoboLearn"
__credits__ = ["Brian Delhaisse"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brian Delhaisse"
__email__ = "briandelhaisse@gmail.com"
__status__ = "Development"


class Model(object):
    r"""(Learning Base) Model (aka parametrized model)

    This abstract class must be inherited by any learning models, and provides a common API for all these models.
    It is often a simple wrapper over a specific learning model implemented in a certain library.

    The learning model has no direct knowledge about the inputs and outputs (such as states / actions), and can be
    used out of the box. In our framework, we refer these learning models as the 'inner models', and the class that
    connects the inputs/outputs (such as states and actions) with the inner model as the 'outer model'.
    Outer models are thus learning models that maps states / actions to states / actions, while inner models often
    maps arrays to arrays (where an array can represent a scalar, a vector, a matrix, a tensor,...). In other words,
    the outer model is a wrapper around the inner model but knows how to deal with state / action inputs and outputs.

    For instance, neural networks are a popular kind of inner models which map arrays to arrays. These inner models
    can be used to represent outer models such as rl (which map states to actions), dynamic models (which map
    states and actions to the next states), value estimators (which, for example, map states to a real number),
    transformation mappings (which map states to states, or actions to actions). Transformation mappings can
    for instance be used to map a human kinematic state to a robot kinematic state.

    .. seealso:
        * https://www.codecademy.com/en/forum_questions/512cd091ffeb9e603b005713
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Initialize the learning model.
        """
        self._models = []  # TODO: should be a directed graph

        self._input_shape = None
        self._output_shape = None

    ##############
    # Properties #
    ##############

    @property
    def models(self):
        return self._models

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    ##################
    # Static Methods #
    ##################

    @staticmethod
    def is_parametric():
        """
        Return True if the model is parametric.

        Returns:
            bool: True if the model is parametric.
        """
        raise NotImplementedError

    @staticmethod
    def is_linear():
        """
        Return True if the model is linear (wrt the parameters). This can be for instance useful for some learning
        algorithms (some only works on linear models).

        Returns:
            bool: True if it is a linear model
        """
        raise NotImplementedError

    @staticmethod
    def is_recurrent():
        """
        Return True if the model is recurrent. This can be for instance useful for some learning algorithms which
        change their behavior when they deal with recurrent learning models.

        Returns:
            bool: True if it is a recurrent model.
        """
        raise NotImplementedError

    @staticmethod
    def is_deterministic():
        """
        Return True if the model is deterministic.

        Returns:
            bool: True if the model is deterministic.
        """
        return not Model.is_probabilistic()

    @staticmethod
    def is_probabilistic():  # is_stochastic
        """
        Return True if the model is probabilistic.

        Returns:
            bool: True if the model is probabilistic.
        """
        raise NotImplementedError

    @staticmethod
    def is_discriminative():
        """
        Return True if the model is discriminative, that is, if the model estimates the conditional probability
        :math:`p(y|x)`.

        Returns:
            bool: True if the model is discriminative.
        """
        raise NotImplementedError

    @staticmethod
    def is_generative():
        """
        Return True if the model is generative, that is, if the model estimates the joint distribution of the input
        and output :math:`p(x,y)`. A generative model allows to sample from it.

        Returns:
            bool: True if the model is generative.
        """
        raise NotImplementedError

    # TODO: isClassifier, isRegressive, isSequential

    ###########
    # Methods #
    ###########

    def has_models(self):
        return len(self._models) > 0

    def add_model(self, model):
        if not isinstance(model, Model):
            raise TypeError("Expecting the model to be an instance of Model.")
        if self.has_models():
            # check that the output size of the last model is equal to the input size of the new model
            last_model = self._models[-1]
            if last_model.output_dims() != model.input_dims():
                # TODO
                pass
        self._models.append(model)

    @abstractmethod
    def _predict(self, x=None):
        """
        Given a possible input, predict the output. The input doesn't always have to be given. For instance,
        generative models generate data without any inputs.
        """
        raise NotImplementedError

    def predict(self, x):
        if self.has_models():
            return [model.predict(x) for model in self.models]
        return self._predict(x)

    @abstractmethod
    def parameters(self):
        """
        Return an iterator over the parameters of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def named_parameters(self):
        """
        Return an iterator over the model parameters, yielding both the name and the parameter itself.
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self):
        """
        Return the parameters in the form of a list or dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def hyperparameters(self):
        """
        Return an iterator over the hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_hyperparams(self):
        """
        Return the hyperparameters in the form of a list or dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_dims(self):
        raise NotImplementedError

    # alias
    # input_dims = getInputDims

    @abstractmethod
    def get_output_dims(self):
        raise NotImplementedError

    # alias
    # output_dims = getOutputDims

    @abstractmethod
    def save(self, filename):
        """
        Save the model in memory.

        Args:
            filename (str): file to save the model in.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, filename):
        """
        Load a model from memory.

        Args:
            filename (str): file that contains the model.

        Returns:
            None
        """
        raise NotImplementedError

    def latex(self):
        """
        Returns the latex equations that describe the learning model.
        This function can also be called __repr__ and/or __str__.
        """
        raise NotImplementedError

    def bibtex(self):
        """
        Returns the references of a learning model in the bibtex format.
        """
        raise NotImplementedError

    def concatenate(self, model):
        """
        Concatenate sequentially two models.
        """
        raise NotImplementedError

    #############
    # Operators #
    #############

    def __repr__(self):
        if self.has_models():
            lst = [self.__class__.__name__ + '(']
            for model in self._models:
                lst.append('\t' + model.__repr__() + ',')
            lst.append(')')
            return '\n'.join(lst)
        else:
            return self.__class__.__name__

    def __len__(self):
        if self.has_models():
            return len(self._models)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __add__(self, other):
        """
        Define how to combine two models together.
        There are two ways to combine a model, sequentially (one after another in time), or in parallel.
        """
        pass

    def __lshift__(self, other):
        """
        Define how to concatenate/sequence two models inline.

        Examples:
            nn = Model()
            nn1 = MLP()
            nn2 = MLP()
            nn << nn1 << nn2  # same as nn << (nn1 >> nn2)
        """
        pass

    def __rshift__(self, other):
        """
        Define how to concatenate/sequence two models. The input of the first model is given to the output
        of the second model.
        """
        # if same model check `rshift()` fct in the corresponding class
        # if different models, it is defined here
        if isinstance(self, NN):
            if isinstance(self, NN):
                pass  # TODO: call rshift()
            elif isinstance(other, CPG):
                pass
            elif isinstance(other, DMP):
                pass
            else:
                raise NotImplementedError("Do not know how to concatenate {} and {}.".format(type(self).__name__,
                                                                                             type(other).__name__))
        elif isinstance(self, GP):
            if isinstance(other, GP):
                pass  # TODO: call rshift
            elif isinstance(other, DMP):
                pass
            else:
                raise NotImplementedError("Do not know how to concatenate {} and {}.".format(type(self).__name__,
                                                                                             type(other).__name__))

        elif isinstance(self, GMM):
            if isinstance(other, GMM):
                pass
            elif isinstance(other, DMP):
                pass
            else:
                raise NotImplementedError("Do not know how to concatenate {} and {}.".format(type(self).__name__,
                                                                                             type(other).__name__))

        else:
            raise NotImplementedError("Do not know how to concatenate {} and {}.".format(type(self).__name__,
                                                                                         type(other).__name__))

    def __floordiv__(self, other):
        """
        Define how to recursively sequence two models.
        """
        pass

    def __mod__(self, other):
        """
        Define how much to wait when producing the output.
        Basically, the output of a model is put into a FIFO queue of size :math:`s` which is specified by the given
        argument. Thus at any time steps :math:`t`, the output actually produced is :math:`o_{t-s}`.
        """
        if not isinstance(other, int):
            raise TypeError("Expecting an integer")

    def __mul__(self, other):
        pass

    def __getitem__(self, key):
        pass

