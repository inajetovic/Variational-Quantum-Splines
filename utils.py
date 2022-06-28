from vqls import *
import math
from scipy.linalg import block_diag
from scipy.interpolate import splrep


def fidelty(vqls,ref):
    
    vqls_normed = vqls/np.linalg.norm(vqls)
    ref_normed = ref/np.linalg.norm(ref)
    return np.abs(vqls_normed.conj().dot(ref_normed)) ** 2

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

def vqls_estimation(x,y,label,saving=False):

    M = []
    Y = []

    for i in range(1, len(x)):
        eq1 = pd.Series([1, x[i - 1]])
        eq2 = pd.Series([1, x[i]])
        M_c = pd.concat([eq1, eq2], axis=1).transpose()
        Y.append([y[i - 1], y[i]])
        M.append(M_c)

    ## Solving B-Spline using diagonal block matrix
    q_beta = []
    c_beta = []
    fid = []

    for i in range(len(M)):
        print('index ',i)
        m = M[i]
        y = Y[i]
        if y == [0.0, 0.0]:
            y = [el + 10 ** -4 for el in y]
        matrix = m
        vector = y

        #normalization = np.sqrt(np.sum(np.array(vector) ** 2, -1))
        #v_norm = (np.array(vector).T / normalization).T

        #v_norm,v_min,v_max = NormalizeData(vector)

        v_norm = vector/np.linalg.norm(vector)

        print(vector,v_norm)
        nq=1
        
        #VQLS
        print('matrix',matrix)
        print('v_norm',v_norm)

        vqls_circuit = VQLS(matrix,v_norm,nq) #Mottonen
        #vqls_circuit = VQLS(matrix,vector,nq) #AE
        weights = vqls_circuit.train()
        q  = vqls_circuit.solution(weights)
    
        #Classic
        c = np.linalg.solve(matrix,vector)

        #denorm
        #q = q.real * (np.linalg.norm(vector))**2 
        #print("vqls error for q",np.linalg.norm(np.linalg.solve(matrix,v_norm)))
        #q = q.real * (np.linalg.norm(vector)) * np.linalg.norm(np.linalg.solve(matrix,v_norm))
        #q = q.real * np.linalg.norm(np.linalg.solve(matrix,vector)) 
        print('pre denorm',q.real)
        
        
        q = q.real  * (np.linalg.norm(vector)) 
        
        print('after denorm',q)

        f = fidelty(q,c)
        
        print('Fidelty between classic and quantum solutions:',f)
        
        q_beta.append(q)
        c_beta.append(c)
        fid.append(f)

    df = pd.DataFrame(columns=['lower', 'upper', 'q_beta0', 'q_beta1', 'c_beta0', 'c_beta1'])

    for m, q, c in zip(M, q_beta, c_beta):
        row = [m[1][0], m[1][1], q[0], q[1], c[0], c[1]]
        row = pd.Series(row, index=df.columns)
        df = df.append(row, ignore_index=True)
    df['fidelity'] = fid
    if saving:
        df.to_csv('./' + label + '_full.csv', index=False)
    return df

def estimate_function(data, function, label, c=0, step=0.05,qprod='swap'):

    interval = data.lower.tolist() + data.upper.tolist()
    interval = list(dict.fromkeys(interval))

    if qprod == 'swap':
        dot_prod = swt_product
    elif qprod == 'new':
        dot_prod = simple_inner_product

    # Sampling points within intervals
    X = []
    for i in range(1, len(interval)):
        X.append(np.arange(interval[i - 1], interval[i], step - 0.01).tolist())


    # Function estimation - quantum and classical
    q_beta = [[b0, b1] for b0, b1 in zip(data.q_beta0, data.q_beta1)]
    c_beta = [[b0, b1] for b0, b1 in zip(data.c_beta0, data.c_beta1)]

    full_qy = []
    hybrid_qy = []
    cy = []

    for i in range(len(X)):
        for x in X[i]:
            point = [1, x]
            coeffs = q_beta[i]

            #point = point / np.linalg.norm(point)
            #coeffs = coeffs / np.linalg.norm(coeffs)
            full_qy.append(dot_prod(coeffs,point)) #*np.linalg.norm(coeffs)*np.linalg.norm(point))
            cy.append(c_beta[i][0] + x * c_beta[i][1])
            hybrid_qy.append(point[0]*coeffs[0] + point[1] * coeffs[1]) 

            #print('coeffs in estimate',coeffs)
            #print('point',[1,x])
            #print('full',dot_prod(coeffs,point))
            #print('hybrid',coeffs[0] + x * coeffs[1])
            #print('DIff btw classic and ip', (coeffs[0] + x * coeffs[1]) - (dot_prod(coeffs,point)))
 
    full_qy = [0 if math.isnan(x) else x for x in full_qy]

    #print('Hybrid',hybrid_qy)
    #print('full',full_qy)

    #fnq,_,_ = NormalizeData(full_qy)
    #print('fnq',fnq)

    #print('X',X)

    x = [item for sublist in X for item in sublist]
    y = [function(value, c) for value in x]

    data_est = pd.DataFrame()
    data_est['x'] = x
    data_est['y'] = y
    data_est['full_quantum'] = full_qy
    data_est['hybrid_quantum'] = hybrid_qy
    data_est['classical_spline'] = cy

    return data_est

def plot_activation(label, data, data_coef, full=True):
    '''Plot the specified function with quantum and classical estimates overimposed.'''
    x = data.x
    y = data.y
    cy = data.classical_spline

    if full:
        qy = data.full_quantum
        type = 'Full'
    else:
        qy = data.hybrid_quantum
        type = 'Hybrid'

    x_fid = (data_coef.lower + data_coef.upper) / 2
    fid = data_coef.fidelity

    fig, ax = plt.subplots(figsize=(6, 5))
    # Full Qspline
    ax.plot(x, cy, color='orange', label='Classic spline',
            zorder=1)  # , dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
    ax.plot(x, qy, color='steelblue',
            label=type + ' Qspline')  # , dashes=(5, 7),  linestyle='dashed',linewidth=1.3)
    ax.plot(x, y, label='Activation', color='sienna', linestyle='dotted', dashes=(1, 1.5), zorder=2,
            linewidth=3)
    ax.scatter(x_fid, fid, color='cornflowerblue', label='Fidelity', s=10)
    #ax.set_xlim(-1.1, 1.1)
    ax.grid(alpha=0.3)
    #ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
    ax.text(0.65, 0.1, label,
            transform=ax.transAxes, ha="left")
    plt.legend(loc='best')
    #plt.savefig('results/' + label + '_' + type + '.png', dpi=300)
    plt.show()
    plt.close()

def simple_inner_product(q,x):
    dev_i = qml.device("default.qubit", wires=1, shots=10 ** 6)
    @qml.qnode(dev_i)
    def inner_prod(q,x):
        #print ('x',x)
        #print ('q pre inner norm',q)
        #x_norm = x #/ np.linalg.norm(x)
        #q_norm = q #/ np.linalg.norm(q)
        #print ('x_norm',x_norm)
        #print ('q_norm inner',q_norm)

        #print('classic_prod',x_norm[0]*q_norm[0] + x_norm[1]*q_norm[1])
        
        #state preparation for the inputs
        #qml.templates.state_preparations.MottonenStatePreparation(q_norm,wires=0)
        #qml.adjoint(qml.templates.state_preparations.MottonenStatePreparation)(x_norm,wires=0)
        flag=False
        if x[1]<0:
            flag=True
        qml.AmplitudeEmbedding(q,wires=0,pad_with=1.0)
        qml.adjoint(qml.AmplitudeEmbedding)(x,wires=0,pad_with=1.0)
        if flag:
                qml.RZ(pi,wires=0)

        #return qml.expval(qml.PauliZ(wires=0))
        return qml.state()
    
    res = inner_prod(q,x)
    
    #print('ip',res)
    return res[0].real 

def B(x, k, i, t): #from DeBoor/scipy
   #print('x',x)
   #print('k',k) 
   #print('index',i)
   if k == 0:
      #print('k==0')
      return 1.0 if t[i] <= x < t[i+1] else 0.0

   if t[i+k] == t[i]:
      #print('t[i+k] == t[i]')
      c1 = 0.0
   else:
      #print('t[i+k] != t[i]')
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)

   if t[i+k+1] == t[i+1]:  
      #print('t[i+k+1] == t[i+1]')
      c2 = 0.0
   else:
      #print('t[i+k+1] != t[i+1]')   
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)

   #print('c1',c1)
   #print('c2',c2)
   return c1 + c2

def GeneralizedVQS_System(n_steps,func,f_i,inputs,samples,scaled=False):
        
    #Knots List
    T = [inputs[0]]
    for el in inputs:
        T.append(el)

    print('Knots list',T)
    print('T dim',len(T))
    print('x_dim',len(inputs))

    #S matrix and y
    matrix=[]
    vector=[]
    for el in samples:
        n = len(T) - 2 
        row=[]
        for i in range(n):
            row.append(B(el, 1, i, T))
        matrix.append(row)
        if scaled:
            vector.append(func(el,f_i)/1.88)
            # vector = vector / np.linalg.norm(vector)
        else:
            vector.append(func(el,f_i))

    matrix[n_steps-1][n_steps-1]=1.0 
    matrix = np.array(matrix)
    print(matrix)
    print(matrix.shape)

    print('vector',vector)
    v_norm = vector/np.linalg.norm(vector)
    print('v-norm',v_norm)    

    return matrix,vector,v_norm