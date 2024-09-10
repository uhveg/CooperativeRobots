import numpy as np
from robotics import Rotx, Roty, Rotz, sgnbi, linear, power_sigmoid, hyperbolic_sine
from numpy import pi, sin, cos

MAX_SIMULATION_TIME = 120

def desPosition(t:float) -> np.ndarray:
    return np.array([[0,0,0.35]]).T

def desOrientation(t:float) -> np.ndarray:
    omega = 0.1*t
    alpha = (np.pi/20)*sin(omega)
    beta = (np.pi/20)*cos(omega)
    return Rotx(alpha) @ Roty(beta) @ Rotz(-np.pi/2)

Q = [[-1.113, 0.8, 0, 0, -0.7, -1.0, 0.135, pi/2],
    [1.113, 0.8, pi, 0, -0.7, -1.0, 0.135, -pi/2],
    [1.113, -0.8, pi, 0, -0.7, -1.0, 0.135, pi/2],
    [-1.113, -0.8, 0, 0, -0.7, -1.0, 0.135, -pi/2]]

PARAMETERS = dict(KphiC  = 5.0,
                  KphiE  = 5.0,
                  KphiB  = 2.0,
                  KpB    = 3.0,
                  kS     = 20.0,
                  desQ   = np.array([[0,0,0,0,-0.7, -1.0, 0.135,0]]).T,
                  KGains = np.array([0,0,0,0,1,1,1,0]),
                  base_o = -1.0,
                  gamma  = 10.0,
                  AF = linear)