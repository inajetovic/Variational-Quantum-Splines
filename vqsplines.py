#############################################################################################
######################################### Libraries #########################################
#############################################################################################

from utils import *
from vqls import *

#############################################################################################
####################################### Initialization ######################################
#############################################################################################

lower = -1.
upper = 1. 
step = .1

y_norm=False
point_norm=False

both = False

visualize_k_numb=False

#elu 0.1,0.0, relu 0.0 0.0, tanh 1. 1.0
f_i = .0
f_e = .0

optimizer='COBYLA'
label = 'sigmoid'
func = sigmoid


x = np.arange(lower, upper + .03, step).tolist()
y = [func(value,f_i) for value in x]


#############################################################################################
################################# SubSystemMatrices Procedure ###############################
#############################################################################################

M = []
Y = []

for i in range(1, len(x)):
    eq1 = pd.Series([1, x[i - 1]])
    eq2 = pd.Series([1, x[i]])
    M_c = pd.concat([eq1, eq2], axis=1).transpose()
    Y.append([y[i - 1], y[i]])
    M.append(M_c)


#############################################################################################
################################# VQLS and Linear Prob. Solving #############################
#############################################################################################

q_weights=[]
c_coeff=[]
beta_q=[]
v_norms=[]
k_list=[]
for i in range(len(M)):
    print('index ',i)
    m = M[i]
    y = Y[i]
    if y == [0.0, 0.0]:
        #y = [el + 10 ** -4 for el in y]
        y = [.000001, 0.00001] # the relu case
    matrix = m
    vector = y
    
    #normalization for the linear problem solving with VQLS
    v_norm = vector/np.linalg.norm(vector)
    v_norms.append(np.linalg.norm(vector))
    
    nq=1 #number of qubits, 1 for 2x2 matrix inversion
    
    #VQLS
    print('Matrix\n',matrix)
    print('matrixnp\n',np.array(matrix))
    k_numb=np.linalg.cond(np.array(matrix))
    print('Condition number\n',k_numb)
    k_list.append(k_numb)
    print('yk     :',vector)
    print('norm(yk)',np.linalg.norm(vector))
    print('yk_norm:',v_norm)

    vqls_circuit = VQLS(matrix,v_norm,nq,opt=optimizer) 

    print('Optimizing variational params..')
    weights = vqls_circuit.train(max_iter=200) 
    q_weights.append(weights)
    q = vqls_circuit.solution(weights)
    print('Quantum coefficients         :',q)
    beta_q.append(q)
    print("Variational Circuit's weights:",weights)
    
    #Classic beta coefficients
    c = np.linalg.solve(matrix,vector)
    print('beta_classic                 :',c)
    c_coeff.append(c)

df = pd.DataFrame(columns=['lower', 'upper'])

for m in M:
    row = [m[1][0], m[1][1]]
    row = pd.Series(row, index=df.columns)
    df = df.append(row, ignore_index=True)

interval = df.lower.tolist() + df.upper.tolist()
interval = list(dict.fromkeys(interval))

# Sampling points within intervals
X = []
for i in range(1, len(interval)):
    X.append(np.arange(interval[i - 1], interval[i], step - 0.01).tolist())

#############################################################################################
######################################## Inner Product ######################################
#############################################################################################

qc_full = []
classic_prod = []
print('Classic and Quantum Product estimation..')
for i in range(len(X)):
    print('Index',i)
    for x in X[i]:
        point = [1,x]
        #classic_prod
        classic_prod.append(c_coeff[i][0]+x*c_coeff[i][1])
        #quantum_prod with/without norm
        if y_norm:
            norm=v_norms[i]
        elif point_norm:
            norm=np.linalg.norm(point)
            print('point',point)
            print('norm(point)',norm)
            print('point_norm',point/norm)
        else:
            norm = 1
        if both:
            print('v_norms[i]',v_norms[i])
            print('point',np.linalg.norm(point))
            norm= v_norms[i]*np.linalg.norm(point)
            print('norm',norm)

        qc_full.append(vqls_circuit.direct_prod2(q_weights[i],point,visualize=True)*norm)  

x = [item for sublist in X for item in sublist]
y = [func(value,f_e) for value in x]

if label == 'tanh':
    qc_full = [q - 0.2 for q in qc_full] 
if label == 'elu':
    classic_prod = [c -f_i for c in classic_prod]
    qc_full = [c-f_i for c in qc_full]

rss_full = np.sum(np.square(np.array(y) - np.array(qc_full)))
print('RSS:',rss_full)

print('condition_numbers',k_list)


#############################################################################################
######################################## Visualization ######################################
#############################################################################################
                                                    
type = 'Full'
x_cond = (df.lower + df.upper) / 2

x1=[i for j,i in enumerate(x) if j%2==0]
y1=[i for j,i in enumerate(y) if j%2==0]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, classic_prod, color='orange', label='Classic spline',zorder=1)  
ax.plot(x, qc_full, color='steelblue',label=type + ' Qspline') 
ax.scatter(x1,y1,color='sienna',s=10)

ax.grid(alpha=0.3)
plt.legend(loc='best')
plt.show()
plt.close()

if visualize_k_numb:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_cond, k_list, color='cornflowerblue', label='ConditionNumber', s=10)
    ax.grid(alpha=0.3)
    plt.legend(loc='best')
    plt.show()
    plt.close()