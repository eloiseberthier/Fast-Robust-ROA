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
import robust_bounds as rb

# prepare the output file
cols = ['x_0', 'r_ctrl', 'method', 'p_spl', 'rho_up', 'n_itr', 'rho', 'vol', 'cpu_t']
df = pd.DataFrame(columns=cols)

exp_name = 'vanderpol'
cfg_name = exp_name + '_config'
import_str = 'from ' + cfg_name + ' import *'
exec(import_str)
cfg = d, m, Q, R, x0, u0, A0, B0, Rinv, S0, S0inv, S0invs, S0sq, K0, f, jacobian, hessian, bound_hessians, bound_jacobian
print('Prepared for x=', str(x0))
print('R=', str(R)) 

for p in [0] + list(np.logspace(1, 4, 10)):
	p = int(p)
	# Order one bound
	eta = 0.99
	rho_upper = 1. ##########
	#p = 10000 ########## do the sampling if p>0, explicit comp if p=0
	res = rb.one_bound(rho_upper, eta, p, cfg)
	df = df.append(pd.DataFrame([res], columns=cols))

# CS bound
eta = 0.99
rho_upper = 1. ##########
p = 0 ##########
res = rb.twocs_bound(rho_upper, eta, p, cfg)
df = df.append(pd.DataFrame([res], columns=cols))

# B bound
eta = 0.99
rho_upper = 1. ##########
p = 0 ##########
res = rb.twob_bound(rho_upper, eta, p, cfg)
df = df.append(pd.DataFrame([res], columns=cols))

# write in file
file_name = exp_name + '/' + time.strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
df.to_csv(file_name)

print('Results written in file')
