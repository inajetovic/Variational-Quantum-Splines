#############################################################################################
######################################### Libraries #########################################
#############################################################################################

from utils import *
from vqls import *
import time


def train_eval(nq, n_step, label, MAX_ITER = 100, verbose = False):
    optimizer='COBYLA'
    func_dict, func_out, lower, upper = get_func('vqs')
    step = (upper-lower)/n_step
    result = {}

    
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
    k_list=[]
    tr_costs =[]

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

        matrix = M[i]
        vector = Y[i]
        if vector == [0.0, 0.0]:
            #y = [el + 10 ** -4 for el in y]
            vector = [.000001, 0.00001] # the relu case
        v_norm = vector/np.linalg.norm(vector)
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
        tr_costs.append(vqls_circuit.cost_vals)
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
            norm = 1

            qc_full.append(vqls_circuit.direct_prod2(q_weights[i],point,visualize=False)*norm)  

    x = [item for sublist in X for item in sublist]
    y = [func(value,f_e) for value in x]

    result['rmse'] = math.sqrt(np.square(np.subtract(y,qc_full)).mean())
    result['RSS_q']= np.sum(np.square(np.array(y) - np.array(qc_full))).item()
    result['weights'] = q_weights
    result["training_cost"] = tr_costs
    result['seed'] = vqls_circuit.rng_seed
    fig, ax = plt.subplots()
    #print(f"X has len {len(X)} and is {X}")
    #print(f"y has len {len(y)} and is {y}")
    #print(f"x has len {len(x)} and is {x}")
    #print(f"tr_costs has len {len(tr_costs)} and is {tr_costs}")
    #ax.plot(x,qc_full)
    #ax.plot(x, y )
    #ax1 = ax.twinx()
    #ax1.scatter([b[0] for b in X], [u[-1] for u in tr_costs])
    return result


if __name__=='__main__':
  train_eval(nq=1, n_step=20, label="sin", MAX_ITER = 100)
