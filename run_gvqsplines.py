from gvqsplines_v2 import train_eval
import pandas as pd

def multiple_experiment(nq, path, max_iter, experiment_number, func):
    import os 
    pnq = {'3':(3,8),
          '4':(4,16)}
    for i in range(int(experiment_number)):
        print(f"Experiment number {i}")
        el = pnq[nq]
        if not os.path.isfile(path):
            df = pd.DataFrame([train_eval(el[0],el[1], func, max_iter )])
        else:
            df = pd.read_json(path)
            df = pd.concat([df, pd.DataFrame([train_eval(el[0],el[1], func, max_iter )])], ignore_index = True)
        
        df.to_json(path)



if __name__=='__main__':   
    import sys
    if '-h' in sys.argv:
        print('The script accepts only the fololowing paramters: \n\
        - -sp\t\tname of the saving file and path (NOTE the results will be saved with json formatting)\n\
        - -mi\t\tmaximum number of iteration for the COBYLA optimizator\n\
        - -en\t\tnumber of experiments to launch\n\
        - -func\t\tname of the function to approximate, canbe choosen between _sigmoid_ _tanh_, _elu_,_relu_, and _sin_.\n\
        - -nq\t\tnumber of qubits\n\
        - -h \t\tOutputs list of possible parameters\n\
        \n \
        Running the script without any parameter is the same as running:\n \
        python run_gvqsplines.py -sp results.json -mi 300 -en 25 -func sigmoid -nq 3')
        exit()
    if '-mi' in sys.argv:
        mi = sys.argv[sys.argv.index("-mi")+1]
    else:
        mi = 300
    if '-en' in sys.argv:
        en = sys.argv[sys.argv.index("-en")+1]
    else:
        en = 25
    if '-func' in sys.argv:
        func = sys.argv[sys.argv.index("-func")+1]
    else:
        func = 'sigmoid'
    if '-nq' in sys.argv:
        nq = sys.argv[sys.argv.index("-nq")+1]
    else:
        nq = '3'
    if '-sp' in sys.argv:
        path = sys.argv[sys.argv.index("-sp")+1]
    else:
        path = f"results_gvqs_{func}_{nq}.json"

    multiple_experiment(nq, path, mi, en, func)