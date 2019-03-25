
# General model
from .model import Model

# General Learning models #

# basics: Linear, Polynomial, PCA
from .basics import *

# Gaussian
from .gaussian import Gaussian, MVN      # MVN is an alias

# GMM/GMR
from .gmm import GMM

# GP
from .gp import GPR

# HMM
# from .hmm import *

# DNN
from .nn import *


# Learning model for trajectories (movement primitives) #

# CPG
from .cpg import *

# DMP
from .dmp import *

# ProMP
from .promp import *

# KMP
from .kmp import *
