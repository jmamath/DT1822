# DT1822
## Context
This is a project I have worked on after attending probai https://probabilistic.ai/. I validated a PhD level course in probabilistic artificial intelligence.

I had a month to implement two generative models on multiple datasets, and try to improve performance.

Every notebook can be run in google colab with Python 3.
Make sure to enable the GPU to make the experiments faster.
Go to Runtime => Change runtime type => Set Hardware accelerator to GPU.

For every implementation of Bayes by Backdrop we will need to upload the __init__.py and the diag_normal_mixture.py files in the execution environment. If this is not done you won’t be able to execute the library import. 
In fact, the pyro native MixtureOfDiagNormals lack numerical stability, it often output nan when computing log probabilities.
Thankfully a solution was proposed in the following thread: https://github.com/pyro-ppl/pyro/pull/1917. So we took this implementation instead of the native MixtureOfDiagNormals.

Dropout_1 refers to the first implementation of Dropout.
Dropout_2 refers to the second implementation of Dropout.

BbB_1 refers to the first implementation of BbB.
BbB_2 refers to the second implementation of BbB.

Compare_objectives_Dropout is used to output the RMSE, MPIW and PICP comparisons.


## Requirements
* gpyopt - an open source library for bayesian optimization
* pyro - to do variational inference
* pycuda - to run pytorch with cuda on google colab 
