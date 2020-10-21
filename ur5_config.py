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
import pinocchio
from pinocchio.utils import *
import example_robot_data

# dynamics, can be replaced by anything else
d = 12
m = 6

robot = example_robot_data.loadUR()
q0 = np.array([0., -0.2*math.pi, -0.6*math.pi, 0, 0, 0])
v = pinocchio.utils.zero(robot.model.nv)
a = pinocchio.utils.zero(robot.model.nv)
u0 = pinocchio.rnea(robot.model, robot.data, q0, v, a) # recursive Newton-Euler
x0 = np.zeros(12)
x0[:6] = q0 # u0 has been defined above

def f(x, u):
    q = x[:6]
    dq = x[6:]
    a = pinocchio.aba(robot.model, robot.data, q, dq, u)
    f = np.zeros(12)
    f[:6] = dq.copy()
    f[6:] = a
    return f


def jacobian(x, u):
    jacx = np.zeros((12, 12))
    jacu = np.zeros((12, 6))
    q = x[:6]
    dq = x[6:]
    a = pinocchio.computeABADerivatives(robot.model, robot.data, q, dq, u)
    jacx[:6, 6:] = np.eye(6)
    jacx[6:, :6] = robot.data.ddq_dq
    jacx[6:, 6:] = robot.data.ddq_dv
    jacu[6:, :] = robot.data.Minv
    return jacx, jacu

def hessian(x):
    eps = 1e-6
    hess = np.zeros((12, 12, 12))
    for i in range(12):
        dx = np.zeros(12)
        dx[i] = eps
        u = u0 - K0 @ (x + dx - x0)
        jacx, jacu = jacobian(x + dx, u)
        jacp = jacx -  jacu @ K0
        u = u0 - K0 @ (x - dx - x0)
        jacx, jacu = jacobian(x - dx, u)
        jacm = jacx -  jacu @ K0
        hess[:, :, i] = (jacp - jacm)/(2.*eps)
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
    jacob = np.zeros((p, d, d))
    for i in range(p):
        y = (2*np.random.rand(d) - np.ones(d))
        y = y/np.linalg.norm(y, 2) * np.random.rand()
        x = np.sqrt(rho_upper)*S0invs @ y
        jacx, jacu = jacobian(x0 + x, u0 - K0 @ x)
        jacob[i,:,:] = jacx -  jacu @ K0
    A = (np.min(jacob, axis=0) + np.max(jacob, axis=0))/2.
    bounds = np.max(abs(jacob - A), axis=0)
    return A, bounds


# parameters of the LQR
Q = np.eye(d)
R = 1.*np.eye(m)

# compute S0, K0
A0, B0 = jacobian(x0, u0)
Rinv = np.linalg.inv(R)
S0 = scipy.linalg.solve_continuous_are(A0, B0, Q, R)
S0inv = np.linalg.inv(S0)
S0invs = scipy.linalg.sqrtm(S0inv)
S0sq = scipy.linalg.sqrtm(S0)
K0 = Rinv @ B0.T @ S0
