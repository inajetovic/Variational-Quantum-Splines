#############################################################################################
######################################### Libraries #########################################
#############################################################################################

from utils import *
from vqls import *
import time

#############################################################################################
####################################### Initialization ######################################
#############################################################################################
# K = 2^n = dim(T)-2
# with dim(S) = KxK
# n = num qubits
# T = set of knots
def train_eval(nq, n_step, label, MAX_ITER = 100, scaled=False):
  func_dict, func_out, lower, upper = get_func('gqs')

  func = func_out[label]
  f_i = func_dict[label]

  #############################################################################################
  ###################################### System Preparatio and _sin_#################
  #############################################################################################


  x = np.arange(lower, upper + .03, (upper-lower)/n_step).tolist() 
  xx = np.linspace(lower, upper, n_step) ##inputs sampling in the interval 0,1
  y = [func(value,f_i) for value in x]

  if scaled:
    norm = np.linalg.norm(y)
    y = y / norm

  matrix,vector,v_norm = GeneralizedVQS_System(n_step,label,x,xx,scaled=scaled)


  #############################################################################################
  ################################# VQLS and Linear Prob. Solving #############################
  #############################################################################################

  result = {}

  k_numb=np.linalg.cond(np.array(matrix))
  result['Condition number']=k_numb
  result['norm(yk)']=np.linalg.norm(vector)

  vqls_circuit = VQLS(matrix,v_norm,nq,opt='COBYLA') 

  start = time.time()
  weights = vqls_circuit.train(max_iter=MAX_ITER)  #########################################
  end = time.time()
  result["training_cost"]=vqls_circuit.cost_vals
  result["exe_time"]= end-start
  result["in_train_weights"] = vqls_circuit.weight_history
  #Classic beta coefficients
  c = np.linalg.solve(matrix,vector)
  q = vqls_circuit.solution(weights).real
  #############################################################################################
  ######################################## Inner Product ######################################
  #############################################################################################

  y_c=np.dot(matrix,c) #classic


  y_q=np.dot(matrix,q) #hybrid
                      #Quantum

  y_fq=[]
  for el in matrix:
    y_fq.append(vqls_circuit.direct_prod2(weights,el,visualize=False))



  rss_full = np.sum(np.square(np.array(y_c) - np.array(y_fq)))
  rss_hybr = np.sum(np.square(np.array(y_c) - np.array(y_q)))

  result['RSS_q']=rss_full.item()
  result['RSS_h']=rss_hybr.item()
  result['seed']=vqls_circuit.rng_seed
  return result

if __name__=='__main__':
  train_eval(nq=3, n_step=8, label="sigmoid", MAX_ITER = 100, scaled=False)