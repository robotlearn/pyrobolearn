

class Processor(object):
    r"""Processor

    This class describes pre- and post- processors. Specifically, it describes how to process the data before and
    after the policy.
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


class PreProcessor(Processor):
    r"""Preprocessor

    It processes the input data before giving it to the policy/controller. For instance, it can be a state estimator
    such as a kalman filter.
    """

    def __init__(self, inputs, outputs):
        super(PreProcessor, self).__init__(inputs, outputs)


class PostProcessor(Processor):
    r"""Postprocessor

    It processes the data outputted by the policy. For instance, the policy could output cartesian positions,
    and we could process this data using an inverse kinematic scheme to output joint data.
    """

    def __init__(self, inputs, outputs):
        super(PostProcessor, self).__init__(inputs, outputs)