import numpy as np
from robotics import Rotx, Roty, Rotz, linear
from numpy import pi, sin, cos

MAX_SIMULATION_TIME = 100

def desPosition(t:float) -> np.ndarray:
    if t < 15:
        z = 0.5 - 0.25*t/15
        return np.array([[0,0,z]]).T
    if t < 20:
        z = 0.25 + 0.1*(t-15)/5
        return np.array([[0,0,z]]).T
    return np.array([[0,0,0.35]]).T

def desOrientation(t:float) -> np.ndarray:
    if t < 20:
        return Rotz(-np.pi/2)
    if t < 30:
        a = ((t-20)/10)*np.pi/20
        return Rotx(a) @ Rotz(-np.pi/2)
    if t < 50:
        a = np.pi/20 - ((t-30)/10)*np.pi/20
        return Rotx(a) @ Rotz(-np.pi/2)
    if t < 60:
        a = -np.pi/20 + ((t-50)/10)*np.pi/20
        return Rotx(a) @ Rotz(-np.pi/2)
    if t < 70:
        a = ((t-60)/10)*np.pi/12
        return Roty(a) @ Rotz(-np.pi/2)
    if t < 90:
        a = np.pi/12 - ((t-70)/10)*np.pi/12
        return Roty(a) @ Rotz(-np.pi/2)
    if t < 100:
        a = -np.pi/12 + ((t-90)/10)*np.pi/12
        return Roty(a) @ Rotz(-np.pi/2)
    return Rotz(-np.pi/2)

Q = [[-0.9, 0.8, 0, 0, 0.4, -1.1, -0.870796, pi/2],
    [0.9, 0.8, pi, 0, 0.4, -1.1, -0.870796, -pi/2],
    [0.9, -0.8, pi, 0, 0.4, -1.1, -0.870796, pi/2],
    [-0.9, -0.8, 0, 0, 0.4, -1.1, -0.870796, -pi/2]]

PARAMETERS = dict(KphiC  = 5.0,
                  KphiE  = 5.0,
                  KphiB  = 2.0,
                  KpB    = 3.0,
                  kS     = 20.0,
                  desQ   = np.array([[0, 0, 0, 0, -0.4, -0.8, -0.8, 0]]).T,
                  KGains = np.array([0,0,0,0,1,1,1,0]),
                  base_o = -1.0,
                  gamma  = 10.0,
                  AF = linear)