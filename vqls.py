import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from math import pi,exp
import pandas as pd
import scipy
from scipy.optimize import minimize


#print(qml.__version__)

class VQLS:
    def __init__(self,matrix,vector, n_qubits,opt ="COBYLA", seed = None):
        self.matrix = matrix
        self.vector = vector
        self.n_qubits = n_qubits
        self.tot_qubits = n_qubits + 1  
        self.ancilla_idx = n_qubits  
        self.q_delta = 0.01
        self.n_shots = 512#10 ** 6
        if seed is None:
            np.random.seed()
            self.rng_seed = np.random.randint(0, 10000)
        else:
            self.rng_seed = seed
        #print(self.rng_seed)
        self.iterations = 0
        self.opt = opt
        self.cost_vals = []
        self.weight_history=[]

    # circuit, and his adjoint, to prepare the state |b> = |yk> from b = yk = v_norm 
    def U_b(self, adjoint=False):
        lines=[e for e in range(self.n_qubits)]
        
        if adjoint:
            qml.adjoint(qml.templates.state_preparations.MottonenStatePreparation)(self.vector,wires=lines)
        else:
            qml.templates.state_preparations.MottonenStatePreparation(self.vector,wires=lines)

    def U_b_full(self, adjoint=False):
        lines=[e for e in range(self.n_qubits)]
        if adjoint:
            qml.adjoint(qml.templates.state_preparations.MottonenStatePreparation)(self.vector,wires=lines)
            #qml.adjoint(qml.AmplitudeEmbedding)(self.vector,wires=lines,pad_with=0.0,normalize=True)
        else:
            qml.templates.state_preparations.MottonenStatePreparation(self.vector,wires=lines)
            #qml.AmplitudeEmbedding(self.vector,wires=lines,pad_with=0.0,normalize=True)
        
    #circuits for the Sk matrix
    def A_c(self, idx, adjoint=False):
        for q in range(self.n_qubits):
            if idx == 4 * q:
                # Identity operation
                None
            elif idx == 4 * q + 1:
                qml.CNOT(wires=[self.ancilla_idx, q])

            elif idx == 4 * q + 2:
                qml.CZ(wires=[self.ancilla_idx, q])
            
            elif idx == 4 * q + 3:
                if adjoint:
                    qml.adjoint(qml.CRY)(3*pi,wires=[self.ancilla_idx,q])
                else:
                    qml.CRY(3*pi,wires=[self.ancilla_idx,q])

    #Ansatz from Pennylane
    def variational_block(self,weights): 
        """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
        # We first prepare an equal superposition of all the states of the computational basis.

        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)

        # A very minimal variational circuit.
        for idx, element in enumerate(weights):
            qml.RY(element, wires=idx)

    #These funcitons can be refactored following the code at https://qiskit.org/textbook/ch-paper-implementations/vqls.html
    def three_ansatz(self,weights):
        qml.RY(weights[0],wires=0)
        qml.RY(weights[1],wires=1)
        qml.RY(weights[2],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[3],wires=0)
        qml.RY(weights[4],wires=1)
        qml.RY(weights[5],wires=2)

        qml.CZ(wires=[1,2])
        qml.CZ(wires=[2,0])

        qml.RY(weights[6],wires=0)
        qml.RY(weights[7],wires=1)
        qml.RY(weights[8],wires=2)

    def five_ansatz(self,weights): 

        qml.RY(weights[0],wires=0)
        qml.RY(weights[1],wires=1)
        qml.RY(weights[2],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[3],wires=1)
        qml.RY(weights[4],wires=2)
        qml.RY(weights[5],wires=3)

        qml.CZ(wires=[1,2])
        qml.CZ(wires=[3,1])

        qml.RY(weights[6],wires=2)
        qml.RY(weights[7],wires=3)
        qml.RY(weights[8],wires=4)

        qml.CZ(wires=[2,3])
        qml.CZ(wires=[4,2])

        qml.RY(weights[9],wires=0)
        qml.RY(weights[10],wires=1)
        qml.RY(weights[11],wires=2)
        qml.RY(weights[12],wires=3)
        qml.RY(weights[13],wires=4)

    def four_ansatz(self,weights):

        qml.RY(weights[0],wires=0)
        qml.RY(weights[1],wires=1)
        qml.RY(weights[2],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[3],wires=1)
        qml.RY(weights[4],wires=2)
        qml.RY(weights[5],wires=3)

        qml.CZ(wires=[1,2])
        qml.CZ(wires=[3,1])

        qml.RY(weights[6],wires=0)
        qml.RY(weights[7],wires=1)
        qml.RY(weights[8],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[9],wires=0)
        qml.RY(weights[10],wires=1)
        qml.RY(weights[11],wires=2)
        qml.RY(weights[12],wires=3)


    def vqls_circuit(self,params,output = False):
        dev_mu = qml.device("default.qubit", wires=self.tot_qubits)
        @qml.qnode(dev_mu)
        def local_hadamard_test():
            """
            params=[weights, l, lp, j, part]
            """
            # First Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if params[4] == "Im" or params[4] == "im":
                qml.PhaseShift(-np.pi / 2, wires=self.ancilla_idx)

            # Variational circuit generating a guess for the solution vector |x>
            if self.n_qubits==5:
                self.five_ansatz(params[0])
            elif self.n_qubits==4:
                self.four_ansatz(params[0])
            elif self.n_qubits==3:
                self.three_ansatz(params[0])
            else:
                self.variational_block(params[0])

            # Controlled application of the unitary component A_l of the problem matrix A.
            self.A_c(params[1],adjoint=False)

            # Adjoint of the unitary U_b associated to the problem vector |b>. 
            if self.n_qubits>1:
                self.U_b_full(adjoint=True)
            else:
                self.U_b(adjoint=True)

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if params[3] != -1:
                qml.CZ(wires=[self.ancilla_idx, params[3]])

            # Unitary U_b associated to the problem vector |b>.
            if self.n_qubits>1:
                self.U_b_full(adjoint=False)
            else:
                self.U_b(adjoint=False)

            # Controlled application of Adjoint(A_lp).
            self.A_c(params[2],adjoint=True)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=self.ancilla_idx))
        if output:
            print(params)
            print()
            print(qml.draw_mpl(local_hadamard_test, expansion_strategy="gradient")())
            raise Exception
        return local_hadamard_test()

    def mu(self,weights, l=None, lp=None, j=None):
        """Generates the coefficients to compute the "local" cost function C_L."""
        re_params=[weights, l, lp, j, "Re"]
        mu_real = self.vqls_circuit(re_params)
        im_params=[weights, l, lp, j, "Im"]
        mu_imag = self.vqls_circuit(im_params)
        return mu_real + 1.0j * mu_imag

    def psi_norm(self,c,weights):
        """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
        norm = 0.0

        for l in range(0, len(c)):
            for lp in range(0, len(c)):
                norm = norm + c[l] * np.conj(c[lp]) * self.mu(weights, l, lp, -1)

        return abs(norm)

    def cost_loc(self,c,weights):
        """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
        mu_sum = 0.0

        for l in range(0, len(c)):
            for lp in range(0, len(c)):
                for j in range(0, self.n_qubits):
                    mu_sum = mu_sum + c[l] * np.conj(c[lp]) * self.mu(weights, l, lp, j)

        mu_sum = abs(mu_sum)
        
        # Cost function C_L
        return 0.5 - 0.5 * mu_sum / (self.n_qubits * self.psi_norm(c,weights))

    def Sk_coefficients(self,m):
        """
        Finds the coefficients for the VQS 2x2 matrices
        input m: 2x2 matrix
        output c: coefficients needed to build the matrix Sk by means of quantum circuits
        with the function A_c.
        
        """
        #let's take the elements from the column of the matrix m
        # Sk = | 1 a |
        #      | 1 b |
        

        # Sk = c_0*I + c_1*Pauli_X + c_2*Pauli_Z + c_3*RY(3*pi) 
        if self.n_qubits!=1:
            a=m[0][1]
            b=m[1][1]
        else:
            a=m[1][0]   
            b=m[1][1]

        c=[]
        c.append((b+1)/2)
        c.append((a+1)/2)
        c.append((1-b)/2)
        c.append((a-1)/2)
        return np.array(c)

    def Sk_coefficients_v2(self,m):
        """
        Finds the coefficients for the GVQS 2x2 matrices

        """
        #let's take the elements from the column of the matrix m
        # Sk = | 1-a  a |
        #      | 0  1-b |

        # Sk = c_0*I + c_1*Pauli_X + c_2*Pauli_Z + c_3*RY(3*pi)

        a=m[0][1]  
        b=1.- m[1][1]

        c=[]
        c.append(1.-a/2 -b/2)
        c.append(a/2)
        c.append((b-a)/2)
        c.append(a/2)
        return np.array(c)

    def full_matrix_coeff(self):
        c_list=[]
        if self.n_qubits != 1:
            for i in range(0,len(self.matrix)-1,1):
                c_list.append(self.Sk_coefficients_v2(self.matrix[i:i+2,i:i+2]))
            
            return [float(item) for sublist in c_list for item in sublist] 
        else:
            return self.Sk_coefficients(self.matrix)

    def cost_execution(self,params):
        c = self.full_matrix_coeff()
        cost = self.cost_loc(c, params)
        #print('current solution',self.solution(params,visualize=False))
        #print("\rCost at Step {}: {:9.7f}".format(self.iterations, cost))
        self.cost_vals.append(cost.item())
        self.iterations += 1
        #print(params)
        #print(type(params))
        self.weight_history.append(params)
        return cost
    
    def __minmaxrand(self,nel,min, max):
        return np.random.rand(nel)*(max-min)+min

    def train(self,max_iter, warm_start= None):
        #init

        np.random.seed(self.rng_seed)
        if warm_start is not None:
            w = warm_start
        elif self.n_qubits==3:
            #w = np.full(9, pi/2, requires_grad=True)
            w = self.__minmaxrand(9, 0, np.pi)

        elif self.n_qubits==4:
            #w = np.full(13, np.random., requires_grad=True)
            w = self.__minmaxrand(13, 0, np.pi)
        else:
            #w = self.q_delta * np.random.randn(self.n_qubits, requires_grad=True)
            w = self.__minmaxrand(self.n_qubits, 0, np.pi)
        #
        out = minimize(self.cost_execution, x0=w, method=self.opt, options={"maxiter": max_iter, "tol":0.01})
        out_params = out["x"]
        #print('Final cost function',self.cost_execution(out_params))
        #print('Number of steps',self.iterations)
        return out_params
    
    def solution(self,params,visualize=False, depth = False):
        dev_v = qml.device("default.qubit", wires=self.n_qubits, shots=self.n_shots)
        @qml.qnode(dev_v)
        def state_vector(weights):
            if self.n_qubits==5:
                self.five_ansatz(weights)
            elif self.n_qubits==4:
                self.four_ansatz(weights)
            elif self.n_qubits==3:
                self.three_ansatz(weights)
            else:
                self.variational_block(weights)
            return qml.state()

        if depth:
            specs_func = qml.specs(state_vector)
            print(specs_func(params)["depth"])

        if visualize==True:
            print(qml.draw_mpl(state_vector)(params))

        return state_vector(params)
        
    def direct_prod2(self,params,x,visualize=False, depth=False):     
        #Variational + Inner Prod Circuit
        dev_v = qml.device("default.qubit", wires=self.n_qubits, shots=self.n_shots)
        @qml.qnode(dev_v)
        def prod(weights,x):
            if self.n_qubits==1:
                self.variational_block(weights) #variational block to estimate coefficients
                qml.adjoint(qml.AmplitudeEmbedding)(x,wires=0,pad_with=1.0) #points encoding

            if self.n_qubits==3:
                x = x / np.linalg.norm(x)
                self.three_ansatz(weights)
                qml.adjoint(qml.MottonenStatePreparation)(x,wires=[0,1,2])
    
            if self.n_qubits==4:
                x = x / np.linalg.norm(x)
                self.four_ansatz(weights)
                
                qml.adjoint(qml.MottonenStatePreparation)(x,wires=[0,1,2,3])

            if x[1]<0 and self.n_qubits==1:
                qml.RZ(pi,wires=0)
            return qml.state()
        res = prod(params,x)

        if depth:
            specs_func = qml.specs(prod)
            print(specs_func(params,x)["depth"])

        #visualization
        if visualize:
            print('Quantum State',res[0])
            #print(qml.draw_mpl(prod)(params,x))
        #print(res)
        return res[0].real  


class qProduct():
    def __init__(self,n_qubits=3):
        self.n_qubits = n_qubits
        self.n_shots = 1028
    
    def variational_block(self,weights): 
        """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
        # We first prepare an equal superposition of all the states of the computational basis.

        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)

        # A very minimal variational circuit.
        for idx, element in enumerate(weights):
            qml.RY(element, wires=idx)


    def three_ansatz(self,weights):
        qml.RY(weights[0],wires=0)
        qml.RY(weights[1],wires=1)
        qml.RY(weights[2],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[3],wires=0)
        qml.RY(weights[4],wires=1)
        qml.RY(weights[5],wires=2)

        qml.CZ(wires=[1,2])
        qml.CZ(wires=[2,0])

        qml.RY(weights[6],wires=0)
        qml.RY(weights[7],wires=1)
        qml.RY(weights[8],wires=2)

    def five_ansatz(self,weights): 

        qml.RY(weights[0],wires=0)
        qml.RY(weights[1],wires=1)
        qml.RY(weights[2],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[3],wires=1)
        qml.RY(weights[4],wires=2)
        qml.RY(weights[5],wires=3)

        qml.CZ(wires=[1,2])
        qml.CZ(wires=[3,1])

        qml.RY(weights[6],wires=2)
        qml.RY(weights[7],wires=3)
        qml.RY(weights[8],wires=4)

        qml.CZ(wires=[2,3])
        qml.CZ(wires=[4,2])

        qml.RY(weights[9],wires=0)
        qml.RY(weights[10],wires=1)
        qml.RY(weights[11],wires=2)
        qml.RY(weights[12],wires=3)
        qml.RY(weights[13],wires=4)

    def four_ansatz(self,weights):

        qml.RY(weights[0],wires=0)
        qml.RY(weights[1],wires=1)
        qml.RY(weights[2],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[3],wires=1)
        qml.RY(weights[4],wires=2)
        qml.RY(weights[5],wires=3)

        qml.CZ(wires=[1,2])
        qml.CZ(wires=[3,1])

        qml.RY(weights[6],wires=0)
        qml.RY(weights[7],wires=1)
        qml.RY(weights[8],wires=2)

        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,0])

        qml.RY(weights[9],wires=0)
        qml.RY(weights[10],wires=1)
        qml.RY(weights[11],wires=2)
        qml.RY(weights[12],wires=3)

    def direct_prod2(self,params,x,visualize=False, depth=False):     
        #Variational + Inner Prod Circuit
        dev_v = qml.device("default.qubit", wires=self.n_qubits, shots=self.n_shots)
        @qml.qnode(dev_v)
        def prod(weights,x):
            if self.n_qubits==1:
                self.variational_block(weights) #variational block to estimate coefficients
                qml.adjoint(qml.AmplitudeEmbedding)(x,wires=0,pad_with=1.0) #points encoding

            if self.n_qubits==3:
                x = x / np.linalg.norm(x)
                self.three_ansatz(weights)
                qml.adjoint(qml.MottonenStatePreparation)(x,wires=[0,1,2])
    
            if self.n_qubits==4:
                x = x / np.linalg.norm(x)
                self.four_ansatz(weights)
                
                qml.adjoint(qml.MottonenStatePreparation)(x,wires=[0,1,2,3])

            if x[1]<0 and self.n_qubits==1:
                qml.RZ(pi,wires=0)
            return qml.state()
        res = prod(params,x)

        if depth:
            specs_func = qml.specs(prod)
            print(specs_func(params,x)["depth"])

        #visualization
        if visualize:
            print('Quantum State',res[0])
            print(qml.draw_mpl(prod)(params,x));
        #print(res)
        return res[0].real  
