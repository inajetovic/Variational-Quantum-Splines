#############################################################################################
######################################### Libraries #########################################
#############################################################################################

from utils import *
from vqls import *
import time


def train_eval(nq, n_step, label, MAX_ITER = 100, lower = -1, upper = 1. , scaled=False, verbose = False):
    optimizer='COBYLA'
    func_out = {'sigmoid': sigmoid,'tanh': tanh,'elu': elu, 'relu': relu, 'sin':sin_m}
    func_dict = {'sigmoid': .0,'tanh': 1.0,'elu':.12, 'relu':.0, 'sin':.0}
    step = (upper-lower)/n_step
    result = {}
    print(nq, n_step, label, MAX_ITER)

    #TODO capisci se tenerlo
    # "denormalizations" options. only for visualize the penalties
    y_norm=False
    point_norm=False
    both = False

    #to visualize the condition number plot
    visualize_k_numb=False  

    # function specific 
    f_i = func_dict[label]

    #TODO THE FUCK IS FE
    f_e = f_i

    func = func_out[label]

    #################################
    #########   FUNCTION  ###########    
    #################################
    x = np.arange(lower, upper + .03, step).tolist()
    y = [func(value,f_i) for value in x]

    M = []
    Y = []
    q_weights=[]
    c_coeff=[]
    beta_q=[]
    v_norms=[]
    k_list=[]
    tr_cost = []

    for i in range(1, len(x)):
        eq1 = pd.Series([1, x[i - 1]])
        eq2 = pd.Series([1, x[i]])
        M_c = pd.concat([eq1, eq2], axis=1).transpose()
        Y.append([y[i - 1], y[i]])
        M.append(M_c)

    #################################
    ######### VQLS TRAINING #########
    #################################
    start = time.time()

    for i in range(len(M)):
        print(i, end=' ')
        matrix = M[i]
        vector = Y[i]
        if vector == [0.0, 0.0]:
            #y = [el + 10 ** -4 for el in y]
            vector = [.000001, 0.00001] # the relu case
        v_norm = vector/np.linalg.norm(vector)
        v_norms.append(np.linalg.norm(vector))
        k_numb=np.linalg.cond(np.array(matrix))
        k_list.append(k_numb)

        if verbose:
            print('Matrix\n',matrix)
            print('Condition number\n',k_numb)
            print('yk     :',vector)
            print('norm(yk)',np.linalg.norm(vector))
            print('yk_norm:',v_norm)
            print('Optimizing variational params..')

        vqls_circuit = VQLS(matrix,v_norm,nq,opt=optimizer) 
        weights = vqls_circuit.train(max_iter=MAX_ITER) 
        tr_cost = vqls_circuit.cost_vals
        q_weights.append(vqls_circuit.weight_history[-1])
        c = np.linalg.solve(matrix,vector)
        c_coeff.append(c)

        if verbose:
            print("Variational Circuit's weights:",weights)
            print('beta_classic                 :',c)
    X = []
    for i in range(1, len(x)):
        X.append(np.arange(x[i - 1], x[i], step - 0.01).tolist())
        
    end = time.time()

    result["exe_time"]= end-start
    #################################
    ######### Inner Product #########
    #################################

    qc_full = []
    classic_prod = []

    for i in range(len(X)):
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
            elif both:
                print('v_norms[i]',v_norms[i])
                print('point',np.linalg.norm(point))
                norm= v_norms[i]*np.linalg.norm(point)
                print('norm',norm)
            else:
                norm = 1

            qc_full.append(vqls_circuit.direct_prod2(q_weights[i],point,visualize=False)*norm)  

    x = [item for sublist in X for item in sublist]
    y = [func(value,f_e) for value in x]


    ##### TODO: ???????????????????????????????????????????????????????????????????
    '''
    if label == 'tanh':
        qc_full = [q - 0.2 for q in qc_full] 
    if label == 'elu':
        classic_prod = [c -f_i for c in classic_prod]
        qc_full = [c-f_i for c in qc_full]
    '''
    result['rmse'] = math.sqrt(np.square(np.subtract(y,qc_full)).mean())
    result['RSS_q']= np.sum(np.square(np.array(y) - np.array(qc_full))).item()
    result['weights'] = q_weights
    result["training_cost"] = tr_cost
    result['seed'] = vqls_circuit.rng_seed


    x1=[i for j,i in enumerate(x) if j%2==0]
    y1=[i for j,i in enumerate(y) if j%2==0]



    #TMP
    #fig, ax = plt.subplots(figsize=(6, 5))
    #ax.plot(x, classic_prod, color='orange', label='Classic spline',zorder=1)  
    #ax.plot(x, qc_full, color='steelblue',label=' Qspline') 
    #ax.scatter(x1,y1,color='sienna',s=10)
    #ax.grid(alpha=0.3)
    #plt.legend(loc='best')
    #plt.show()
    #plt.close()

    return result


if __name__=='__main__':
  train_eval(nq=1, n_step=20, label="sigmoid", MAX_ITER = 100)