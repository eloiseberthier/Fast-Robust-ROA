from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.autograd import Variable
from torch import autograd
import scipy
from scipy.linalg import solve_lyapunov
from scipy.integrate import ode, odeint
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matlab.engine
import time
import pandas as pd

# Dynamics, can be replaced by anything else
d = 4
m = 1
G = 9.8
Mass = 1.
Length = 0.5

def f(x, u):
    f = np.zeros(4)
    t1, t2, p1, p2 = x
    den = 16. - 9.*np.cos(t1 - t2)**2
    f[0] = 6./(Mass*Length**2)*(2.*p1 - 3.*np.cos(t1 - t2)*p2)/den
    f[1] = 6./(Mass*Length**2)*(8.*p2 - 3.*np.cos(t1 - t2)*p1)/den
    f[2] = -.5*(Mass*Length**2)*(f[0]*f[1]*np.sin(t1 - t2) + 3.*G/Length*np.sin(t1))
    f[3] = -.5*(Mass*Length**2)*(-f[0]*f[1]*np.sin(t1 - t2) + G/Length*np.sin(t2))
    f[3] += u
    return f

def jacobian(x, u):
    jacx = np.zeros((d, d))
    jacu = np.zeros((d, m))
    f = torch.zeros(4)
    u = Variable(torch.FloatTensor(u), requires_grad = True)
    x = Variable(torch.FloatTensor(x), requires_grad = True)
    t1, t2, p1, p2 = x
    den = 16. - 9.*torch.cos(t1 - t2)**2
    f0 = 6./(Mass*Length**2)*(2.*p1 - 3.*torch.cos(t1 - t2)*p2)/den
    f[0] = f0
    f1 = 6./(Mass*Length**2)*(8.*p2 - 3.*torch.cos(t1 - t2)*p1)/den
    f[1] = f1
    f[2] = -.5*(Mass*Length**2)*(f0*f1*torch.sin(t1 - t2) + 3.*G/Length*torch.sin(t1))
    f[3] = -.5*(Mass*Length**2)*(-f0*f1*torch.sin(t1 - t2) + G/Length*torch.sin(t2))
    f[3:] += u
    for idx in range(d):
        f[idx].backward(retain_graph=True)
        jacx[idx,:] = np.array(x.grad)
        jacu[idx,:] = np.array(u.grad)
        x.grad = None
        u.grad = None
    return jacx, jacu

def hessian(x):
    hess = np.zeros((d, d, d))
    f = torch.zeros(4)
    u = torch.zeros(1)
    x = Variable(torch.FloatTensor(x), requires_grad = True)
    t1, t2, p1, p2 = x
    den = 16. - 9.*torch.cos(t1 - t2)**2
    f0 = 6./(Mass*Length**2)*(2.*p1 - 3.*torch.cos(t1 - t2)*p2)/den
    f[0] = f0
    f1 = 6./(Mass*Length**2)*(8.*p2 - 3.*torch.cos(t1 - t2)*p1)/den
    f[1] = f1
    f[2] = -.5*(Mass*Length**2)*(f0*f1*torch.sin(t1 - t2) + 3.*G/Length*torch.sin(t1))
    f[3] = -.5*(Mass*Length**2)*(-f0*f1*torch.sin(t1 - t2) + G/Length*torch.sin(t2))
    f[3:] += u
    for idx in range(d):
        g = autograd.grad(f[idx], x, create_graph=True)[0].view(-1)
        for n in range(d):
            g[n].backward(retain_graph=True)
            hess[idx,n,:] = np.array(x.grad)
            x.grad = None
    return hess

def bound_hessians(x0, S0invs, P0invs, rho_upper, p=1000):
    hess = np.zeros((p, d, d, d))
    for i in range(p):
        y = (2*np.random.rand(d) - np.ones(d))
        y = y/np.linalg.norm(y, 2) * np.random.rand()
        x = np.sqrt(rho_upper)*S0invs @ y
        h = hessian(x0+x)
        for idx in range(d):
            hess[i,idx,:,:] = P0invs @ h[idx,:,:] @ P0invs
    bounds = np.max(abs(hess), axis=0)
    return bounds

def bound_jacobian(x0, S0invs, rho_upper, p=500):
	cst_jac = - B0 @ K0	
	jacob = np.zeros((p, d, d))
	for i in range(p):
		y = (2*np.random.rand(d) - np.ones(d))
		y = y/np.linalg.norm(y, 2) * np.random.rand()
		x = np.sqrt(rho_upper)*S0invs @ y 
		jacob[i,:,:] = cst_jac + jacobian(x0 + x, u0)[0]
	A = (np.min(jacob, axis=0) + np.max(jacob, axis=0))/2.
	bounds = np.max(abs(jacob - A), axis=0)
	return A, bounds


# parameters of the LQR
Q = np.eye(d)
R = 1.*np.eye(m)
x0 = np.zeros(d)#np.array([np.pi, np.pi, 0., 0.]) # bottom / top
u0 = np.zeros(m)

# compute S0, K0
A0, B0 = jacobian(x0, u0)
Rinv = np.linalg.inv(R)
S0 = scipy.linalg.solve_continuous_are(A0, B0, Q, R)
S0inv = np.linalg.inv(S0)
S0invs = scipy.linalg.sqrtm(S0inv)
S0sq = scipy.linalg.sqrtm(S0)
K0 = Rinv @ B0.T @ S0
