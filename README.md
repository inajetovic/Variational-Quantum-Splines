# Running the repeated experimental setting
Install requirement.txt with python 3

Run the ``` exper_run.py ``` script with the fololowing command line paramters
- ``` -sp ``` name of the saving file and path (NOTE the results will be saved with json formatting)
- ```-mi ``` maximum number of iteration for the COBYLA optimizator
- ``` -en``` number of experiments to launch
- ```-func``` name of the function to approximate, canbe choosen between _sigmoid_ _tanh_, _elu_,_relu_, and _sin_.
- ```-nq``` number of qubits 
- ```-h``` outputs list of possible parameters

Running the script without any parameter is the same as running:
```
python exper_run.py -sp results.json -mi 300 -en 25 -func sigmoid -nq 3
```

The saved file is a Dataset containing the following features:
- Condition number ```float``` 
- norm(yk)  ```float```
- training_cost ```list[float]```
- exe_time ```float``` (seconds)
- in_train_weight ```list[list[float]]```
- RSS_q ``float``
- RSS_h `float`
- seed `int`

# VQSplines and GVQSplines 

(WORK IN PROGRESS)

This repository contains the code to reproduce the results presented in the thesis work
[Variational Quantum Splines - Moving Beyond Linearity for Quantum Activation Functions]

# Description

The first part of this thesis works starts with the aim to reformulate in a Variational fashion the QSPlines work present in the paper [Quantum Splines for Non-Linear Approximations](https://dl.acm.org/doi/pdf/10.1145/3387902.3394032) in order to move from
a fault-tolerant model to a NISQ one. 
In the second part a new methodology is proposed, in order to tackle the same problem in a more generalized way and overcome many limitations of the previous works. Just like the QSplines method 4 activation functions are studied,
namely *sigmoid*, *hyperbolic tangent (tanh)*, *rectified linear unit (relu)* and *exponential linear unit (elu)*.

### Usage

The code is divided in four main parts:
- *vqls.py* contains the class used to exploit the VQLS algorithm and takes inspiration from the VQLS pennylane implementation [https://pennylane.ai/qml/demos/tutorial_vqls.html].
- *utils.py* contains all the libraries, functions and custom routines used.
- *vqsplines* is the code to exploit the VQSpline proposed method to  compute the B-spline approximation of the desired activation function.
- *gvqsplines* is the code to exploit the GVQSpline proposed method to  compute the B-spline approximation of the desired activation function.

### Requirements 
all the requirements are present in the *requirements.txt* file.