from utils import *

lower = -1.
upper = 1.
step = .1


## Definition of the interval of B-Spline
c=.5

x = np.arange(lower, upper + .03, step).tolist()
y = [ relu(value, c) for value in x]

data_coef = vqls_estimation(x, y,label='relu') # data_coef = pd.read_csv('results/relu_full.csv')
data_est = estimate_function(data_coef, relu, 'relu', c=0, step=step)

data_est.hybrid_quantum = data_est.hybrid_quantum - c
data_est.classical_spline = data_est.classical_spline - c
#data_est.full_quantum,_,_=NormalizeData(data_est.full_quantum)

plot_activation('relu', data_est, data_coef, full = True)
plot_activation('relu', data_est, data_coef, full = False)