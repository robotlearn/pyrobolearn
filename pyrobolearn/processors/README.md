## Processors

Processors are functions that are applied to the inputs (respectively outputs) of an approximator/learning model before (respectively after) being processed by it. Processors might have parameters but these are not trainable/optimizable and are thus fixed and given at the beginning.

These processors allow for instance to scale the input and/or output of a learning model. They accept as inputs a State/Action, numpy array, or torch Tensor.

They can for instance be used to normalize the input state before it is fed to the policy.
