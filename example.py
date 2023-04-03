import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
import teg_get_best_n
import importlib
importlib.reload(teg_get_best_n)

# Tests
nObs = 1000
nVar = 30
nL_max = 5
noise = 0.1
nSims = 50
nL_true = []
nL_est = []
nL_knee = []
for iSim in range(nSims):
    if iSim % 10 == 0:
        print(iSim)
    nL = np.random.randint(0, nL_max)
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
nL_err = np.abs(nL_est - nL_true)
nL_err_knee = np.abs(nL_knee - nL_true)
nL_err_bias = nL_est - nL_true
nL_err_knee_bias = nL_knee - nL_true
M = []
for z in zip(nL_true, nL_est, nL_err, nL_err_bias):
    print(z)
    M.append(z)
M = np.array(M)
print(np.mean(M, axis = 0))
M = []
for z in zip(nL_true, nL_knee, nL_err_knee, nL_err_knee_bias):
    #print(z)
    M.append(z)
M = np.array(M)
print(np.mean(M, axis = 0))

# Checks of KneeLocator
if False:
    from kneed import KneeLocator, DataGenerator as dg
    import matplotlib.pyplot
    wx, w = dg.convex_decreasing()
    KL = KneeLocator(wx, w, curve="convex", direction="decreasing")
    KL_split = KL.elbow
    nComp_knee = KL_split
    print(nComp_knee)
    KL.plot_knee()
