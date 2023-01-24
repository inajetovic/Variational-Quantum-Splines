from vqsplines_v2 import train_eval
import pandas as pd

def multiple_experiment(nq, path, stp, max_iter, experiment_number, func):
    res = []
    import os 
    for i in range(int(experiment_number)):
        print(f"Experiment number {i}")
        if not os.path.isfile(path):
            df = pd.DataFrame([train_eval(int(nq),int(stp), func, int(max_iter) )])
        else:
            df = pd.read_json(path)
            df = pd.concat([df, pd.DataFrame([train_eval(int(nq),int(stp), func, int(max_iter) )])], ignore_index = True)

        #import os
        #if os.path.isfile(path):
        #    os.remove(path)
        print(df.dtypes)
        df.to_json(path)
        #res.append(train_eval(el[0],el[1], func, max_iter ), ignore_index = True)


    #df = pd.DataFrame(res)
    

    #df.to_json(path)


if __name__=='__main__':   
    import sys
    if '-h' in sys.argv:
        print('The script accepts only the fololowing paramters: \n\
        - -sp\t\tname of the saving file and path (NOTE the results will be saved with json formatting)\n\
        - -mi\t\tmaximum number of iteration for the COBYLA optimizator\n\
        - -en\t\tnumber of experiments to launch\n\
        - -func\t\tname of the function to approximate, canbe choosen between _sigmoid_ _tanh_, _elu_,_relu_, and _sin_.\n\
        - -stp\t\tnumber of steps\n\
        - -h \t\tOutputs list of possible parameters\n\
        \n \
        Running the script without any parameter is the same as running:\n \
        python run_vqsplines.py -sp results_vqs_sigmoid_20.json -mi 300 -en 25 -func sigmoid -stp 20')
        exit()
    if '-mi' in sys.argv:
        mi = sys.argv[sys.argv.index("-mi")+1]
    else:
        mi = 300
    if '-en' in sys.argv:
        en = sys.argv[sys.argv.index("-en")+1]
    else:
        en = 25
    if '-stp' in sys.argv:
        stp = sys.argv[sys.argv.index("-stp")+1]
    else:
        stp = 20
    if '-func' in sys.argv:
        func = sys.argv[sys.argv.index("-func")+1]
    else:
        func = 'sigmoid'
    if '-sp' in sys.argv:
        path = sys.argv[sys.argv.index("-sp")+1]
    else:
        path = f"results_vqs_{func}_1_{stp}.json"

    multiple_experiment('1', path, stp, mi, en, func)