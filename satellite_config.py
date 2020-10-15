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

#global d, m, Q, R, x0, u0, A0, B0, Rinv, S0, S0inv, S0invs, S0sq, K0, f, jacobian, hessian, bound_hessians, bound_jacobian

# dynamics, can be replaced by anything else
d = 6
m = 3
Is = np.array([5., 3., 2.])
I = np.diag(Is)
Iinv = np.diag(np.array([1./5., 1./3., 1./2.]))
l = np.array([1, 2, 0])
r = np.array([2, 0, 1])
Diag = np.array([[-1.,  0.,  0.],
                 [ 0.,  3.,  0.],
                 [ 0.,  0., -2.]])
Mat = Diag @ Iinv

def f(x, u):
    omega = x[:3]
    sigma = x[3:]
    f = np.zeros(6)
    f[:3] = Mat @ np.multiply(omega[l], omega[r])
    f[3:] = .25*((1-sigma @ sigma.T)*np.eye(3) + 2*np.outer(sigma, sigma) - 2*np.array([[0, sigma[2], sigma[1]],
                                                                                [sigma[2], 0, sigma[0]],
                                                                                [sigma[1], sigma[0], 0]]) ) @ omega
    f[:3] += Iinv @ u
    return f

def jacobian(x, u):
    omega = x[:3]
    sigma = x[3:]
    jac = np.zeros((6, 6))
    jacu = np.zeros((6, 3))
    jac[:3, 0] = Mat @ np.array([0., omega[2], omega[1]])
    jac[:3, 1] = Mat @ np.array([omega[2], 0., omega[0]])
    jac[:3, 2] = Mat @ np.array([omega[1], omega[0], 0.])
    jac[3:, :3] = .25*((1-sigma @ sigma.T)*np.eye(3)+2*np.outer(sigma, sigma) - 2*np.array([[0, sigma[2], sigma[1]],
                                                                                [sigma[2], 0, sigma[0]],
                                                                                [sigma[1], sigma[0], 0]]) )
    jac[3:, 3:] = - 0.5 * np.outer(omega, sigma)
    jac[3:, 3] += 0.5* np.array([2*sigma[0]*omega[0]+sigma[1]*omega[1]+sigma[2]*omega[2],
                                 sigma[1]*omega[0],
                                 sigma[2]*omega[0]])
    jac[3:, 4] += 0.5* np.array([sigma[0]*omega[1],
                                 2*sigma[1]*omega[1]+sigma[0]*omega[0]+sigma[2]*omega[2],
                                 sigma[2]*omega[1]])
    jac[3:, 5] += 0.5* np.array([sigma[0]*omega[2], 
                                 sigma[1]*omega[2],
                                 2*sigma[2]*omega[2]+sigma[0]*omega[0]+sigma[1]*omega[1]])
    jac[3:, 3] -= 0.5* np.array([0., omega[2], omega[1]])
    jac[3:, 4] -= 0.5* np.array([omega[2], 0., omega[0]])
    jac[3:, 5] -= 0.5* np.array([omega[1], omega[0], 0.])
    jacu[:3] = Iinv
    return jac, jacu

def hessian(x):
    hess = np.zeros((d, d, d))
    Is = torch.FloatTensor([5., 3., 2.])
    I = torch.diag(Is)
    Iinv = torch.diag(torch.FloatTensor([1./5., 1./3., 1./2.]))
    l = np.array([1, 2, 0])
    r = np.array([2, 0, 1])
    Diag = torch.FloatTensor([[-1.,  0.,  0.],
                              [ 0.,  3.,  0.],
                              [ 0.,  0., -2.]])
    Mat = Diag @ Iinv
    x = Variable(torch.FloatTensor(x), requires_grad = True)
    omega = x[:3]
    sigma = x[3:]
    f = torch.zeros(6)
    u = torch.ones(3)
    f[:3] = Mat @ torch.mul(omega[l], omega[r])
    f[3:] = .25*((1-sigma @ sigma.T)*torch.eye(3)+2*torch.ger(sigma, sigma) - 2*torch.FloatTensor([[0, sigma[2], sigma[1]],
                                                                                [sigma[2], 0, sigma[0]],
                                                                                [sigma[1], sigma[0], 0]]) ) @ omega
    f[:3] += Iinv @ u 
    for idx in range(6):
        g = autograd.grad(f[idx], x, create_graph=True)[0].view(-1)
        for n in range(6):
            g[n].backward(retain_graph=True)
            hess[idx,n,:] = np.array(x.grad)
            x.grad = None
    return hess

def bound_hessians(x0, S0invs, P0invs, rho_upper, p=1000):
	if p>0:
		hess = np.zeros((p, d, d, d))
		for i in range(p):
		    y = (2*np.random.rand(d) - np.ones(d))
		    y = y/np.linalg.norm(y, 2) * np.random.rand()
		    x = np.sqrt(rho_upper)*S0invs @ y
		    h = hessian(x0+x)
		    for idx in range(d):
		        hess[i,idx,:,:] = P0invs @ h[idx,:,:] @ P0invs
		bounds = np.max(abs(hess), axis=0)
	elif p==0:
		S0inv = S0invs @ S0invs
		# sample a few hessians
		ploc = 10
		hessians = np.zeros((ploc, d, d, d))
		points = np.zeros((ploc, d+1))
		for i in range(ploc):
			x = 2.*np.random.rand(d) - np.ones(d)
			points[i] = np.concatenate([np.ones(1), x])
			hessians[i] = hessian(x)
		# fit a linear model
		coefs = np.zeros((d, d, d, d+1)) # coord 0 of x in intercept
		for idx in range(d):
			ys = P0invs @ hessians[:,idx,:,:] @ P0invs
			for i in range(d):
				for j in range(d):
					regression = linear_model.LinearRegression(fit_intercept=False)
					model = regression.fit(points, ys[:,i,j])
					coefs[idx,i,j,:] = model.coef_
		# compute the bound
		bounds = np.zeros((d, d, d))
		for idx in range(d):
		    for i in range(d):
		        for j in range(d):
		            c = coefs[idx,i,j,1:]
		            intercept = coefs[idx,i,j,0] + c @ x0.T
		            ray = np.sqrt(rho_upper * c @ S0inv @ c.T)
		            bounds[idx,i,j] = max(abs(intercept+ray), abs(intercept-ray))
	return bounds

def bound_jacobian(x0, S0invs, rho_upper, p=100000):
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
R = 10*np.eye(m)
x0 = np.zeros(d)
u0 = np.zeros(m)

# compute S0, K0
A0, B0 = jacobian(x0, u0)
Rinv = np.linalg.inv(R)
S0 = scipy.linalg.solve_continuous_are(A0, B0, Q, R)
S0inv = np.linalg.inv(S0)
S0invs = scipy.linalg.sqrtm(S0inv)
S0sq = scipy.linalg.sqrtm(S0)
K0 = Rinv @ B0.T @ S0
