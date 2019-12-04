# DT1822
## Context
This is a project I have worked on after attending probai https://probabilistic.ai/. I passed a PhD level course (7.5 credits) from Norwegian University of Science and Technology (NTNU) in Probabilistic Artificial Intelligence. The project is available by looking at *dt8122_probabilistic_ai_2019.pdf*.
The transcript is available at *karakterutskrift.pdf*


I had a month (12 June to 12 July) to implement two generative models on multiple datasets, and try to improve their performance. Check the report *report.pdf*

Every notebook can be run in google colab with Python 3.
Make sure to enable the GPU to make the experiments faster.
Go to Runtime => Change runtime type => Set Hardware accelerator to GPU.

For every implementation of Bayes by Backprop we will need to upload the __init__.py and the diag_normal_mixture.py files in the execution environment. If this is not done you won’t be able to execute the library import. 
In fact, the pyro native MixtureOfDiagNormals lack numerical stability, it often output nan when computing log probabilities.
Thankfully a solution was proposed in the following thread: https://github.com/pyro-ppl/pyro/pull/1917. So we took this implementation instead of the native MixtureOfDiagNormals.

Dropout_1 refers to the first implementation of Monte Carlo Dropout.
Dropout_2 refers to the second implementation of Monte Carlo Dropout.

BbB_1 refers to the first implementation of Bayes by Backprop.
BbB_2 refers to the second implementation of Bayes by Backprop.

Compare_objectives_Dropout is used to output the RMSE, MPIW and PICP comparisons.


## Requirements
* gpyopt - an open source library for bayesian optimization
* pyro - to do variational inference
* pycuda - to run pytorch with cuda on google colab 
