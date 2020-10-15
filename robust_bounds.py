from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import torch
from torch.autograd import Variable
from torch import autograd
from sklearn import linear_model
from scipy.linalg import solve_lyapunov
from scipy.integrate import ode, odeint
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matlab.engine
import time
import pandas as pd



# helpers
def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return math.ceil(n * multiplier) / multiplier
def print_matrix(M, name):
    m, n = M.shape
    print(name,'= [ ', end='')
    for i in range(m):
        for j in range(n):
            print(M[i,j], end=' ')
        if i<m-1:
            print('; ', end='\n')
    print('];')
def volume(S, rho):
    # only up to constants, should be this vol*Vol(B(0,1))
    d = S.shape[0]
    return rho**(d/2)/np.linalg.det(S)**(1/2)




def one_bound(rho_upper, eta, p, cfg):
	# Order one bound
	d, m, Q, R, x0, u0, A0, B0, Rinv, S0, S0inv, S0invs, S0sq, K0, f, jacobian, hessian, bound_hessians, bound_jacobian = cfg
	rho_init = rho_upper
	eng = matlab.engine.start_matlab()
	succ = False
	cpu_time = []
	while not succ:
		ts = time.process_time()
		A, bounds = bound_jacobian(x0, S0invs, rho_upper, p=p)
		ti = time.process_time() - ts
		tj = time.process_time()
		B = np.zeros((d, d**2))
		for k in range(d):
			for l in range(k*d, (k+1)*d):
				B[k,l] = 1
		C = np.zeros((d**2, d))
		for k in range(d):
			for l in range(d):
				C[k*d+l, l] = bounds[k, l]
		A_arg = [list(A[i]) for i in range(A.shape[0]) ]
		B_arg = [list(B[i]) for i in range(B.shape[0]) ]
		C_arg = [list(C[i]) for i in range(C.shape[0]) ]
		S_arg = [list(S0[i]) for i in range(S0.shape[0]) ]
		L_shp = [list([1, 1]) for i in range(d**2)]
		tmin, L = eng.nbldi(matlab.double(L_shp), matlab.double(A_arg),  matlab.double(B_arg),  matlab.double(C_arg),
			      matlab.double(S_arg), nargout=2)
		succ = tmin < 0
		dt = time.process_time() - tj
		cpu_time.append([ti, dt])
		if not succ: rho_upper = eta*rho_upper
	vol = volume(S0, rho_upper)
	print('Ended robust bound =', rho_upper)
	return [str(x0), str(R), 'robust', p, rho_init, len(cpu_time), rho_upper, vol, cpu_time]





def twocs_bound(rho_upper, eta, p, cfg):
	# CS bound
	d, m, Q, R, x0, u0, A0, B0, Rinv, S0, S0inv, S0invs, S0sq, K0, f, jacobian, hessian, bound_hessians, bound_jacobian = cfg
	rho_init = rho_upper
	P0 = Q + S0 @ B0 @ K0
	P0inv = np.linalg.inv(P0)
	P0invs = scipy.linalg.sqrtm(P0inv)
	cpu_time = []
	rhoCS = 0.
	while rhoCS < rho_upper:
		ts = time.process_time()
		bd = bound_hessians(x0, S0invs, P0invs, rho_upper, p=p)
		ti = time.process_time() - ts
		tj = time.process_time()
		M = np.zeros((d, d))
		for i in range(d):
		    M = M + np.sqrt(S0[i,:] @ S0inv @ S0[i,:].T) * bd[i, :, :]
		lbdaCS = np.max(np.linalg.eigvals(M))
		nrhoCS = 1./lbdaCS**2
		dt = time.process_time() - tj
		cpu_time.append([ti, dt])
		print(nrhoCS, rho_upper)
		if nrhoCS <= rho_upper:
		    rhoCS = nrhoCS
		else: break
		rho_upper = eta*rho_upper
	vol = volume(S0, rhoCS)
	print('Ended CS bound =', rhoCS)
	return [str(x0), str(R), 'CS', p, rho_init, len(cpu_time), rhoCS, vol, cpu_time]




def twob_bound(rho_upper, eta, p, cfg):
	# B bound
	d, m, Q, R, x0, u0, A0, B0, Rinv, S0, S0inv, S0invs, S0sq, K0, f, jacobian, hessian, bound_hessians, bound_jacobian = cfg
	rho_init = rho_upper
	P0 = Q + S0 @ B0 @ K0
	P0inv = np.linalg.inv(P0)
	P0invs = scipy.linalg.sqrtm(P0inv)
	rhoB = 0.
	cpu_time = []
	while rhoB < rho_upper:
		ts = time.process_time()
		bd = bound_hessians(x0, S0invs, P0invs, rho_upper, p=p)
		ti = time.process_time() - ts
		tj = time.process_time()
		D = np.eye(d)
		for i in range(d):
		    D[i, i] = np.max(np.linalg.eigvals(bd[i, :, :]))
		lbdaB = np.sqrt(d)*np.max(np.linalg.eigvals(D @ S0sq))
		nrhoB = 1./lbdaB**2
		dt = time.process_time() - tj
		cpu_time.append([ti, dt])
		print(nrhoB, rho_upper)
		if nrhoB <= rho_upper:
		    rhoB = nrhoB
		else: break
		rho_upper = eta*rho_upper
	vol = volume(S0, rhoB)
	print('Ended B bound =', rhoB)
	return [str(x0), str(R), 'B', p, rho_init, len(cpu_time), rhoB, vol, cpu_time]
