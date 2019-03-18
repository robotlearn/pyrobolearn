## Learning models

In this folder, we provide the various learning models. These can be categorized into two categories: movement primitives and general function approximators.

These include:
- Central pattern generators (CPG; the version provided by )
- Dynamic movement primitives (DMP)
- Probabilistic movement primitives (ProMP)
- Kernelized movement primitives (KMP)
- Linear models
- PCA models
- Polynomial models
- Gaussian mixture models (GMM) with its regression counterpart (GMR)
- Gaussian processes (GP; it uses/wraps the [GPyTorch](https://github.com/cornellius-gp/gpytorch) library)
- Neural networks (currently, only MLP are provided)

TODO:
- [ ] finish to implement the models
- [ ] provide multiple tests/examples for each model
- [ ] implement other models such as HMMs


#### what to check/look next?

Check the `approximators`, `policies`, `values`, and `dynamics` folders.
