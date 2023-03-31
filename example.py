import numpy as np
import teg_get_best_n

# Tests
nObs = 1000
nVar = 50
nL_max = 10
noise = 1
nSims = 100
nL_true = []
nL_est = []
for iSim in range(nSims):
    if iSim % 10 == 0:
        print(iSim)
    nL = np.random.randint(0, nL_max)
    X = teg_get_best_n.make_sim_data(nObs, nVar, nL, noise)
    O = teg_get_best_n.get_n_components(X)
    nComp = O['nComponents']
    nL_true.append(nL)
    nL_est.append(nComp)
nL_true = np.array(nL_true)
nL_est = np.array(nL_est)
nL_err = np.abs(nL_true - nL_est)
for z in zip(nL_true, nL_est, nL_err):
    print(z)
