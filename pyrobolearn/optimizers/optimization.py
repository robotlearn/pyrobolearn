

class TrustRegion(object):
    r"""Trust Region

    "Trust region is a term used in mathematical optimization to denote the subset of the region of the objective
    function that is approximated using a model function (often a quadratic). If an adequate model of the objective
    function is found within the trust region, then the region is expanded; conversely, if the approximation is poor,
    then the region is contracted. Trust-region methods are also known as restricted-step methods.

    The fit is evaluated by comparing the ratio of expected improvement from the model approximation with the actual
    improvement observed in the objective function. Simple thresholding of the ratio is used as the criterion for
    expansion and contraction; a model function is "trusted" only in the region where it provides a reasonable
    approximation.

    Trust-region methods are in some sense dual to line-search methods: trust-region methods first choose a step size
    (the size of the trust region) and then a step direction, while line-search methods first choose a step direction
    and then a step size." [1]

    References:
        [1] https://en.wikipedia.org/wiki/Trust_region
    """
    pass


class LineSearch(object):
    r"""Line Search

    "In optimization, the line search strategy is one of two basic iterative approaches to find a local minimum
    :math:`\mathbf{x}^*` of an objective function :math:`f:\mathbb{R}^{n} \to \mathbb{R}`. The other approach is trust
    region.

    The line search approach first finds a descent direction along which the objective function :math:`f` will be
    reduced and then computes a step size that determines how far :math:`\mathbf{x}` should move along that direction.
    The descent direction can be computed by various methods, such as gradient descent, Newton's method and
    Quasi-Newton method. The step size can be determined either exactly or inexactly.

    Here is an example gradient method that uses a line search in step 4.

    1. Set iteration counter k = 0, and make an initial guess :math:`\mathbf{x}_{0}` for the minimum
    2. Repeat:
    3.     Compute a descent direction :math:`\mathbf{p}_k`
    4.     Choose :math:`\alpha_k` to 'loosely' minimize :math:`h(\alpha)=f(\mathbf{x}_k + \alpha \mathbf{p}_k)` over
            :math:`\alpha \in \mathbb{R}_{+}`
    5.     Update :math:`\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k`, and :math:`k = k + 1`
    6. Until :math:`|| \nabla f( \mathbf{x}_k ) || < tolerance

    At the line search step (4) the algorithm might either exactly minimize :math:`h`, by solving
    :math:`h'(\alpha _{k})=0`, or loosely, by asking for a sufficient decrease in :math:`h`. One example of the former
    is conjugate gradient method. The latter is called inexact line search and may be performed in a number of ways,
    such as a backtracking line search or using the Wolfe conditions.

    Like other optimization methods, line search may be combined with simulated annealing to allow it to jump over
    some local minima." [1]

    References:
        [1] https://en.wikipedia.org/wiki/Line_search
    """
    pass


class BacktrackingLineSearch(LineSearch):
    r"""Backtracking Line Search

    "In (unconstrained) minimization, a backtracking line search, a search scheme based on the Armijo-Goldstein
    condition, is a line search method to determine the maximum amount to move along a given search direction.
    It involves starting with a relatively large estimate of the step size for movement along the search direction,
    and iteratively shrinking the step size (i.e., "backtracking") until a decrease of the objective function is
    observed that adequately corresponds to the decrease that is expected, based on the local gradient of the
    objective function." [1]

    References:
        [1] https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    pass
