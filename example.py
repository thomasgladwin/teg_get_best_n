import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
import teg_get_best_n
import importlib
importlib.reload(teg_get_best_n)

# Tests
nL_v = []
per_nL = []
for nL_loop in [0, 1, 2, 3, 4] + list(range(5, 11)) + list(range(12, 20, 4)):
    nObs = 2000
    nVar = 40
    nL_max = 15
    noise = 0.1
    nSims = 100
    nL_true = []
    nL_est = []
    nL_knee = []
    for iSim in range(nSims):
        #if iSim % 10 == 0:
        #    print(iSim)
        #nL = np.random.randint(0, nL_max)
        nL = nL_loop
        X = teg_get_best_n.make_sim_data(nObs, nVar, nL, noise)

        O = teg_get_best_n.get_n_components(X)
        nComp = O['nComponents']

        w = O['eigenvalues']
        KL = KneeLocator(range(0, len(w)), w, curve="convex", direction="decreasing")
        KL_split = KL.elbow
        nComp_knee = KL_split

        nL_true.append(nL)
        nL_est.append(nComp)
        nL_knee.append(nComp_knee)
    nL_true = np.array(nL_true)
    nL_est = np.array(nL_est)
    nL_knee = np.abs(nL_knee)
    nL_correct = nL_est == nL_true
    nL_err = np.abs(nL_est - nL_true)
    nL_err_knee = np.abs(nL_knee - nL_true)
    nL_correct_knee = nL_knee == nL_true
    nL_err_bias = nL_est - nL_true
    nL_err_knee_bias = nL_knee - nL_true
    nL0 = nL_est == 0
    M = []
    for z in zip(nL_true, nL_est, nL_correct, nL_err, nL_err_bias, nL0):
        #print(z)
        M.append(z)
    M = np.array(M)
    print(np.mean(M, axis = 0))
    nL_v.append(nL_loop)
    per_nL.append(np.mean(M, axis = 0))

for z in zip(nL_v, per_nL):
    print(z)

