
# VQSplines and GVQSplines 

This repository contains the code to reproduce the results presented in the research work titled '_Variational Quantum Splines - Moving Beyond Linearity for Quantum Activation Functions_'.

## Description

The first part of work starts with the aim to reformulate in a Variational fashion the QSPlines work present in the paper [Quantum Splines for Non-Linear Approximations](https://dl.acm.org/doi/pdf/10.1145/3387902.3394032) in order to move from a fault-tolerant model to a NISQ one. 
In the second part a new methodology is proposed, in order to tackle the same problem in a more generalized way and overcome many limitations of the previous works. 

As explained in the paper 4 function are studied, *sigmoid*, *rectified linear unit (relu)*, *exponential linear unit (elu)* and the *sin* function.


## Repository sructure
- **experiments:** folder containing the notebook to reproduce the experiments.
    - **experiment_GQS.ipynb:** file used to train and evaluate a GQSpline model. It can be used to reproduce the best performing GQSpline models for each target function.
    - **experiment_VQS.ipynb:** file used to train and evaluate a VQSpline model. It can be used to reproduce the best performing VQSpline models for each target function.
- **results:** folder containing the results of the experiment as dataframe in json format.
- **plots:** folder containing the plots of the models for each function.
- **run_gqsplines.py:** script to perform and save multiple experiments (GQSpline).
- **run_vqsplines.py:** script to perform and save multiple experiments (VQSpline).
- **vqls.py:** file containing the implementation of the VQLS and the Quantum inner product. 
- **vqspline_v2.py:** scrtipt used to train the model (VQSpline).
- **gqspline_v2.py:** scrtipt used to train the model (GQSpline).
- **utils.py:** collection of utility functions.
### Launch multiple experiments GQSplines

Run the ``` run_gqsplines.py ``` script with the fololowing command line paramters
- ``` -sp ``` name of the saving file and path (NOTE the results will be saved with json formatting)
- ```-mi ``` maximum number of iteration for the COBYLA optimizator
- ``` -en``` number of experiments to launch
- ```-func``` name of the function to approximate, canbe choosen between _sigmoid_ _tanh_, _elu_,_relu_, and _sin_.
- ```-nq``` number of qubits 
- ```-h``` outputs list of possible parameters

Running the script without any parameter is the same as running:
```
python run_gqsplines.py -sp results.json -mi 300 -en 25 -func sigmoid -nq 3
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

### Launch multiple experiments VQSplines

Run the ``` run_vqsplines.py ``` script with the fololowing command line paramters
- ``` -sp ``` name of the saving file and path (NOTE the results will be saved with json formatting)
- ```-mi ``` maximum number of iteration for the COBYLA optimizator
- ``` -en``` number of experiments to launch
- ```-func``` name of the function to approximate, canbe choosen between _sigmoid_ _tanh_, _elu_,_relu_, and _sin_.
- ```-stp``` number of knots 
- ```-h``` outputs list of possible parameters

Running the script without any parameter is the same as running:
```
python run_gqsplines.py -sp results.json -mi 300 -en 25 -func sigmoid -stp 20
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


## Installation
To run the code and reproduce the results of the paper, it is recommended to re-create the same testing environment following the procedure below.

*Note: it's assumed Anaconda is installed*
- First clone the repository
- Second, create a conda environment from scratch and install the requirements specified in the requirements.txt file:  
    ```
    # enter the repository
    cd project_folder

    # create an environment with desired dependencies found in the requirements.txt file
    conda env create
    pip install -r requirements.txt
    ```
