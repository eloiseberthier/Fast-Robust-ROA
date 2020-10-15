from __future__ import print_function
import numpy as np
import scipy
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42;
matplotlib.rcParams['ps.fonttype'] = 42
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
import time
import robust_bounds as rb

exp_name = 'vanderpol'
cfg_name = exp_name + '_config'
import_str = 'from ' + cfg_name + ' import *'
exec(import_str)
cfg = d, m, Q, R, x0, u0, A0, B0, Rinv, S0, S0inv, S0invs, S0sq, K0, f, jacobian, hessian, bound_hessians, bound_jacobian
print('Prepared for x=', str(x0))
print('R=', str(R)) 

# target point
xT = np.array([-1, -1])
# target region around xT
Starget = np.array([[9.5616 ,  2.2650],
               [2.2650 ,  3.6483]])  
t1 = 2
dt = 0.05

# Compute the xs
def ff(t, y):
    return -f(y)
r = ode(ff).set_integrator('vode', method='bdf', with_jacobian=False)
r.set_initial_value(xT, 0)
ts = [r.t]
xs = [r.y]
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    ts.append(r.t)
    xs.append(r.y)
xs.reverse()

# Compute the As
N = len(ts)
As = []
Bs = []
Ks = []
for t in range(N):
    As.append(jacobian(xs[t], None)[0])
    Bs.append(np.zeros((d, m)))
    Ks.append(np.zeros((m, d)))

# Compute the Ss
def g(S, t, As, Q, N, t1):
    Sm = S.reshape((d, d))
    idx = int( (N - 1)/t1 * (t1 - t) )
    return ( As[idx].T @ Sm + Sm @ As[idx] + Q).flatten()
sol = odeint(g, Starget.flatten(), ts, args=(As, Q, N, t1))
Ss = []
for t in range(N):
    Ss.append(sol[t].reshape((d, d)))
Ss.reverse()

# helpers for adaptive version
def sample_hessians(p=10):
    hessians = np.zeros((p, d, d, d))
    points = np.zeros((p, d+1))
    for i in range(p):
        x = 2*np.random.rand(d) - np.ones(d)
        points[i, :] = np.concatenate([np.ones(1), x])
        hessians[i,:,:,:] = hessian(x)
    return points, hessians
def fit_hessians(points, hessians, Pinvs=np.eye(d)):
    # fit degree one model
    p = points.shape[0]
    coefs = np.zeros((d, d, d, d+1))
    for idx in range(d):
        ys = Pinvs @ hessians[:,idx,:,:] @ Pinvs # to be checked
        for i in range(d):
            for j in range(d):
                regression = linear_model.LinearRegression(fit_intercept=False)
                model = regression.fit(points, ys[:,i,j])
                coefs[idx,i,j,:] = model.coef_
    return coefs
def ada_bound_hessians(x0, Sinv, rho0, coefs):
    # solve sup cT x on xTS0x<1 and give bound on S0, rho=1, around x0
    bd = np.zeros((d, d, d))
    for idx in range(d):
        for i in range(d):
            for j in range(d):
                c = coefs[idx,i,j,1:]
                intercept = coefs[idx,i,j,0] + c @ x0.T
                ray = np.sqrt(rho0 * c @ Sinv @ c.T)
                bd[idx,i,j] = max(abs(intercept+ray), abs(intercept-ray))
    return bd
points, hessians = sample_hessians() # used everywhere

# adaptive version
def rho_dynamics(rv, t, Ss, xs, Q, N, t1, rd0=10., eta=0.9):
    # iterative, P-1/2 rescaling
    idx = int( (N - 1)/t1 * (t1 - t) )
    S0 = Ss[idx]
    S0inv = np.linalg.inv(S0) 
    if rv<=0: return 0.
    flag = True
    rdot = rd0
    lastrdot = rdot
    while flag:
        Uis = scipy.linalg.sqrtm(np.linalg.inv(Q + rdot/rv*S0)).real #Q+SBK, B=0
        coefs = fit_hessians(points, hessians, Uis)
        bd = ada_bound_hessians(xs[idx], S0inv, rv, coefs)
        M = np.zeros((d, d))
        for i in range(d):
            M = M + np.sqrt(S0[i,:] @ S0inv @ S0[i,:].T) * bd[i, :, :]
        lbdaCS = np.max(np.linalg.eigvals(M))
        flag = math.sqrt(rv) * lbdaCS < 1.
        if flag:
            lastrdot = rdot
            rdot = eta*rdot
    return lastrdot



init_t2 = time.process_time()

rho0 = 1.
eta = 0.9
rd0 = 2.
rhos = [rho0]
for t in range(N-1):
    rhos.append(rhos[-1] - dt*rho_dynamics(rhos[-1], ts[t], Ss, xs, Q, N, t1, rd0=rd0, eta=eta))
    print(rhos[-1], t)
rhos.reverse()
print(rhos)

t2 = time.process_time() - init_t2

### Order one bound


eng = matlab.engine.start_matlab()


def rho_dynamics(rv, t, Ss, xs, As, Q, N, t1, rd0=10., eta=0.9):
	# iterative, P-1/2 rescaling
	idx = int( (N - 1)/t1 * (t1 - t) )
	S0 = Ss[idx]
	A0 = As[idx]
	S0inv = np.linalg.inv(S0) 
	S0invs = scipy.linalg.sqrtm(S0inv) 
	if rv<=0: return 0.
	flag = True
	rdot = rd0
	lastrdot = rdot
	while flag:
		A, bounds = bound_jacobian(xs[idx], S0invs, rv, p=0)
		Sdot = - Q - S0 @ A0 - A0.T @ S0 # Sdot with K=0. (+SBK omitted)
		A = A - .5*rdot/rv*np.eye(d) + .5* S0inv @ Sdot
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
		flag = tmin<0 #tmin >= 0
		if flag:
			lastrdot = rdot
			rdot = eta * rdot - (1. - eta)* 1.
	return lastrdot


init_t1 = time.process_time()

rho0 = 1.
eta = 0.95
rd0 = 2.#0.01
rhos1 = [rho0]
for t in range(N-1):
    rhos1.append(rhos1[-1] - dt*rho_dynamics(rhos1[-1], ts[t], Ss, xs, As, Q, N, t1, rd0=rd0, eta=eta))
    print(rhos1[-1], t)
rhos1.reverse()
print(rhos1)

t1 = time.process_time() - init_t1

### Import results for SOS bound
file0 = open('ts.txt', 'r') 
Lines = file0.readlines() 
tsbis = []
for line in Lines: 
    tsbis.append(float(line.strip()))

file1 = open('itr1.txt', 'r') 
Lines = file1.readlines() 
itr1 = []
for line in Lines: 
    itr1.append(float(line.strip()))
file2 = open('itr2.txt', 'r') 
Lines = file2.readlines() 
itr2 = []
for line in Lines: 
    itr2.append(float(line.strip()))
print(itr1)
print(itr2)

print('Times:')
print('1st order', t1)
print('2nd order', t2)

plt.figure(figsize=(7, 4.2))
plt.xlabel(r'$t$', size=15)
plt.ylabel(r'$\rho(t)$', size=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.plot(ts, rhos1, linewidth=3, color='tab:red', label=r'$\mathcal{C}_1$')
plt.plot(ts, rhos, linewidth=3, color='tab:green', label=r'$\mathcal{C}_2^a$')
plt.plot(tsbis, itr1, linewidth=3, linestyle=':', color='black', label='SOS itr 1')
plt.plot(tsbis, itr2, linewidth=3, linestyle='--', color='black', label='SOS itr 2')
plt.legend(prop={"size":15})
#plt.title('Time-varying Vanderpol')
plt.savefig('./tv_vanderpol.pdf')
plt.show()
 

plt.figure(figsize=(10, 6))

# le fond
plt.fill(xlim[:, 0], xlim[:, 1], facecolor='chartreuse', alpha=0.4, edgecolor='black', linewidth=1)

# ROA of 0: R
ROA = np.array([[ 1.5, -0.5],
       [-0.5,  1. ]])
th = np.linspace(-math.pi, math.pi, 100)
E = np.linalg.inv(scipy.linalg.sqrtm(ROA/2.3))
ell = E @ np.array([np.cos(th), np.sin(th)])
plt.fill(ell[0,:], ell[1,:],  facecolor=[0.9, 0.9, 0.9],   linewidth=1)

# B_f
th = np.linspace(-math.pi, math.pi, 100)
E = np.linalg.inv(scipy.linalg.sqrtm(Starget))#S0/rhoCS))
ell = E @ np.array([np.cos(th), np.sin(th)]) + np.array([xT for i in range(100)]).T
plt.fill(ell[0,:], ell[1,:],  facecolor=[1, 0., 0.], edgecolor='black', linewidth=1)

# B(t)
for idx in range(N-1, -1, -1):
    E = np.linalg.inv(scipy.linalg.sqrtm(Ss[idx]/rhos1[idx]))
    ell = E @ np.array([np.cos(th), np.sin(th)]) + np.repeat([xs[idx]], 100, axis=0).T
    plt.fill(ell[0,:], ell[1,:], facecolor=[0.8, 0.8, 0.8], edgecolor='black', linewidth=1, alpha=0.5)

# trajectory
xs = np.array(xs)
x = xs[:,0]
y = xs[:,1]
plt.plot(x, y, color='black', linewidth=1.5, linestyle='--')
for i in range(0, len(x)-1, 4):
    theta = np.arctan( (y[i+1] - y[i]) /( x[i+1] - x[i]) )
    if ( x[i+1] < x[i]): theta = theta - math.pi
    plt.arrow(x[i], y[i], 0.05*np.cos(theta), 0.05 * np.sin(theta), width= 0.01,
    head_width=0.06, head_length=0.15, facecolor='black')

plt.annotate(r'$\mathcal{B}_f$', xy=(-0.6, -1.7), size=30, color='red' )
plt.annotate(r'$\mathcal{B}(t)$', xy=(-0.3, 1.2), size=30, color='black' )
plt.annotate(r'$\mathcal{R}$', xy=(1, 0.2), size=30, color=[0.5, 0.5, 0.5] )
plt.axis('off')
plt.savefig('./funnel.pdf')
plt.show()



