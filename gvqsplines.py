#############################################################################################
######################################### Libraries #########################################
#############################################################################################

from utils import *
from vqls import *
import time

#############################################################################################
####################################### Initialization ######################################
#############################################################################################

lower = 0.
upper = 1. 
n_step = 8
f_i = 0.0  #.12elu 1. tanh

scaled=False

label = 'sigmoid_t'
func = sigmoid_t



x = np.arange(lower, upper + .03, (upper-lower)/n_step).tolist() 
print('x',x)
xx = np.linspace(lower, upper, n_step) ##inputs sampling in the interval 0,1
y = [func(value,f_i) for value in x]

if scaled:
  norm = np.linalg.norm(y)
  y = y / norm

tck=splrep(x,y,k=1) #coeffs
print('scipy knots',tck[0])
print('scipy coeffs',tck[1])
print('xx',xx)

#############################################################################################
############################### Knots and Basis Expansion Matrix ############################
#############################################################################################

#Knots List
T = [x[0]]
for el in x:
    T.append(el)

print('Knots list',T)
print('T dim',len(T))
print('x_dim',len(x))

#S matrix and y
matrix=[]
vector=[]
for j,el in enumerate(xx):
   n = len(T) - 2 
   row=[]
   for i in range(n):
       row.append(B(el, 1, i, T))
   
   print(j)
   print(row)
   matrix.append(row)
   n_row = row/ np.linalg.norm(row)
   print('Norm of the row',n_row)
   vector.append(func(el,f_i))
if scaled:
  vector = vector / np.linalg.norm(vector)

matrix[n_step-1][n_step-1]=1.0 
matrix = np.array(matrix)
print(matrix)
print(matrix.shape)

print('vector',vector)
v_norm = vector/np.linalg.norm(vector)
print('v-norm',v_norm)    

#############################################################################################
################################# VQLS and Linear Prob. Solving #############################
#############################################################################################
nq=3 

k_numb=np.linalg.cond(np.array(matrix))
print('Condition number\n',k_numb)
print('norm(yk)',np.linalg.norm(vector))

vqls_circuit = VQLS(matrix,v_norm,nq,opt='COBYLA') 

print('Optimizing variational params..')
start = time.time()
weights = vqls_circuit.train(max_iter=200)  #########################################
end = time.time()
print("The time of execution of above program is :", end-start)
q = vqls_circuit.solution(weights)
print('Quantum coefficients         :',q)
print("Variational Circuit's weights:",weights)

#Classic beta coefficients
c = np.linalg.solve(matrix,vector)
print('beta_classic                 :',c)

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

print('RSS_q:',rss_full)
print('RSS_h:',rss_hybr)

#############################################################################################
######################################## Visualization ######################################
#############################################################################################

fig, ax = plt.subplots()
ax.plot(xx, y_q, color='steelblue', lw=2, alpha=0.7, label='Hybrid Spline ')
ax.plot(xx, y_c, color='orange', lw=2, alpha=0.7, label='Classic Spline')
ax.scatter(x,y,color='sienna')

ax.grid(True)

ax.legend(loc='best')

plt.show()

fig, ax = plt.subplots()
ax.plot(xx, y_fq, color='steelblue', lw=2, alpha=0.7, label='Full Quantum Spline')
ax.plot(xx, y_c, color='orange', lw=2, alpha=0.7, label='Classic Spline')
#ax.plot(x, y, color='orange', lw=2, alpha=0.7, label='Classic Spline')
ax.scatter(x,y,color='sienna')
#ax.plot(xx,v_norm,color='yellow')

ax.grid(True)

ax.legend(loc='best')

plt.show()
