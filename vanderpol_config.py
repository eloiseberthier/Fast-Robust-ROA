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
d = 2
m = 1 # unactuated, B = 0
xlim = pd.read_csv('xlim.csv', header=None)
xlim = np.array(xlim)

f = lambda x : np.array([-x[1], x[0] + x[1]*(x[0]**2-1)])

def jacobian(x, u):
	return np.array([[0., -1.], [1. + 2.*x[0]*x[1], x[0]**2 - 1.]]), np.zeros((d, m))

def hessian(x):
    hess = np.zeros((2, 2, 2))
    hess[1,:,:] = np.array([[2.*x[1], 2.*x[0]], [2.*x[0], 0.]])
    return hess

def bound_hessians(x0, S0invs, P0invs, rho_upper, p=0):
    # Since P0 = I for Q=I, the coefficients are computed manually and fixed
    coefs = np.zeros((2, 2, 2, 3)) # coord 0 of x is intercept of affine model
    coefs[1, 0, 0, 2] = 2
    coefs[1, 0, 1, 1] = 2
    coefs[1, 1, 0, 1] = 2
    # solve sup cT x on xTS0x<1 and give bound on S0
    S0inv = S0invs @ S0invs
    bd = np.zeros((2, 2, 2))
    for idx in range(2):
        for i in range(2):
            for j in range(2):
                c = coefs[idx,i,j,1:]
                intercept = coefs[idx,i,j,0] + c @ x0.T
                ray = np.sqrt(rho_upper * c @ S0inv @ c.T)
                bd[idx,i,j] = max(abs(intercept+ray), abs(intercept-ray))
    return bd

def bound_jacobian(x0, S0invs, rho_upper, p=500):
	cst_jac = - B0 @ K0
	if p>0:
		jacob = np.zeros((p, d, d))
		for i in range(p):
			y = (2*np.random.rand(d) - np.ones(d))
			y = y/np.linalg.norm(y, 2) * np.random.rand()
			x = np.sqrt(rho_upper)*S0invs @ y 
			jacob[i,:,:] = cst_jac + jacobian(x0 + x, u0)[0]
		A = (np.min(jacob, axis=0) + np.max(jacob, axis=0))/2.
		bounds = np.max(abs(jacob - A), axis=0)
	elif p==0:
		# a no-sampling version for this simple quadratic jacobian
		J1 = np.array([[0, 1], [1, 0]])
		bd1d = rho_upper * np.min(np.linalg.eigvals( S0invs @ J1 @ S0invs))
		bd1u = rho_upper * np.max(np.linalg.eigvals( S0invs @ J1 @ S0invs))
		m1 = (bd1d + bd1u)/2.
		bd1 = abs(bd1u - bd1d)/2.
		J2 = np.array([[1, 0], [0, 0]])
		bd2d = rho_upper * np.min(np.linalg.eigvals( S0invs @ J2 @ S0invs))
		bd2u = rho_upper * np.max(np.linalg.eigvals( S0invs @ J2 @ S0invs))
		m2 = (bd2d + bd2u)/2.
		bd2 = abs(bd2u - bd2d)/2.
		A = np.array([[0, -1], [1+m1, -1+m2]])
		bounds = np.zeros((d, d))
		bounds[1, 0] = bd1
		bounds[1, 1] = bd2
	return A, bounds

# parameters of the LQR
x0 = np.zeros(d)
u0 = np.zeros(m)
Q = np.eye(d)
R = np.eye(m) # unused

# compute S0, K0
A0, B0 = jacobian(x0, None)
Rinv = np.linalg.inv(R)
S0 = solve_lyapunov(A0.T, -Q)
S0inv = np.linalg.inv(S0)
S0invs = scipy.linalg.sqrtm(S0inv)
S0sq = scipy.linalg.sqrtm(S0)
K0 = Rinv @ B0.T @ S0
