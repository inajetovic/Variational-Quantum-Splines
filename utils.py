from ctypes import ArgumentError
from sympy import elliptic_f
from vqls import *
import math
from scipy.linalg import block_diag
from scipy.interpolate import splrep




def get_func(mode = 'gqs'):
    if(mode == 'gqs') :
        return  {'sigmoid': .0,'tanh': 1.0,'elu':.12, 'relu':.0, 'sin':2}, {'sigmoid': sigmoid_t,'elu': elu_t, 'relu': relu_t, 'sin':sin_m}, 0, 1#'tanh': tanh_t,
    elif(mode == 'vqs'):
        return {'sigmoid': .0,'tanh': 1.0,'elu':0, 'relu':0, 'sin':1}, {'sigmoid': sigmoid,'tanh': tanh,'elu': elu, 'relu': relu, 'sin':sin_o}, -1,1

    else:
        raise ArgumentError(f'No mode called {mode}')



def sin_m(x,z=2):
    return 1/2*math.sin(x*pi*z)+1/2

def sin_o(x,z=2):
    return math.sin(x*pi*z)

def elu(z, c = 0, alpha = .3):
	return c + z if z >= 0 else c + alpha*(math.e**z -1)

def elu_t(z,c=0,alpha = .3):
    res=0
    if z >= .4:
        res= c + z  -.4
    else:
        
        res=c + alpha*(math.e**(z-.4) - 1)

    return res

def leakyrelu(input_value,c=0.05):
    if input_value > 0:
        return input_value
    else:
        return c*input_value

def relu(x, c = 0):
    return c + max(0.0, x)

def relu_t(x,c=0):
    return c + max(0.0, x-0.43)

def tanh(x, c = 1):
  return (c + np.tanh(x))*c/2

def tanh_t(x, c = 1):
  return (c + np.tanh(2*x-1))*c/2

def sigmoid_t(x, c=0):
    return c + 1 / (1 + exp(-8 * (x-1/2))) 

def sigmoid(x, c=0):
    return c + 1 / (1 + exp(-4 * x))

def B(x, k, i, t): #from DeBoor/scipy

   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0

   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)

   if t[i+k+1] == t[i+1]:  
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)


   return c1 + c2

def GeneralizedVQS_System(n_steps,label,inputs,samples,scaled=False):

    """
    n_steps: K (dimensionality of the problem)
    func(string): Activation (or Any non linear) function) 
        -sigmoid
        -tanh
        -elu
        -relu
    inputs: X
    samples: xx
    scaled: tackle func's Y outputs, with norm equal to 1.
    """

    func_dict = {'sigmoid': .0,'tanh': 1.0,'elu':.12, 'relu':.0, 'sin':2}
    func_out = {'sigmoid': sigmoid_t,'tanh': tanh_t,'elu': elu_t, 'relu': relu_t, 'sin':sin_m}

    
    f_i = func_dict[label]
    func = func_out[label]
        
    #Knots List
    T = [inputs[0]]
    for el in inputs:
        T.append(el)

    #print('Knots list',T)
    #print('T dim',len(T))
    #print('x_dim',len(inputs))

    #Problem Condition check
    assert n_steps == len(T) - 2

    #S matrix and y
    matrix=[]
    vector=[]
    for el in  samples:
        n = len(T) - 2 
        row=[]
        for i in range(n):
            row.append(B(el, 1, i, T))
        matrix.append(row)
        vector.append(func(el,f_i))

    if scaled:
        vector = vector / np.linalg.norm(vector)

    matrix[n_steps-1][n_steps-1]=1.0 
    matrix = np.array(matrix)
    #print(matrix)
    #print(matrix.shape)

    #print('vector',vector)
    v_norm = vector/np.linalg.norm(vector)
    #print('v-norm',v_norm)    

    return matrix,vector,v_norm