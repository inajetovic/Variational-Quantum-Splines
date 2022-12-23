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
def train_eval(nq, n_step, label, MAX_ITER = 100, lower = 0., upper = 1. , scaled=False):
  func_out = {'sigmoid': sigmoid_t,'tanh': tanh_t,'elu': elu_t, 'relu': relu_t, 'sin':sin_m}
  func = func_out[label]
  func_dict = {'sigmoid': .0,'tanh': 1.0,'elu':.12, 'relu':.0, 'sin':2}
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

  tck=splrep(x,y,k=1) #coeffs
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
  ##############################################################################################
  ######################################### Visualization ######################################
  ##############################################################################################
  #import matplotlib
  #matplotlib.use('TKAgg')
  #
  #fig, ax = plt.subplots()
  #ax.plot(xx, y_q, color='steelblue', lw=2, alpha=0.7, label='Hybrid Spline ')
  #ax.plot(xx, y_c, color='orange', lw=2, alpha=0.7, label='Classic Spline')
  #ax.scatter(x,y,color='sienna')
  #
  #ax.grid(True)
  #
  #ax.legend(loc='best')
  #
  #plt.show()
  #
  #fig, ax = plt.subplots()
  #
  #ax.plot(xx, y_fq, color='steelblue', lw=2, alpha=0.7, label='Full Quantum Spline')
  #ax.plot(xx, y_c, color='orange', lw=2, alpha=0.7, label='Classic Spline')
  ##ax.plot(x, y, color='orange', lw=2, alpha=0.7, label='Classic Spline')
  #ax.scatter(x,y,color='sienna')
  ##ax.plot(xx,v_norm,color='yellow')
  #
  #ax.grid(True)
  #
  #ax.legend(loc='best')
  #plt.show()
  #
  #

if __name__=='__main__':
  train_eval(nq=3, n_step=8, label="sigmoid", MAX_ITER = 100, lower = 0., upper = 1. ,  scaled=False)