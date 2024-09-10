import numpy as np
from robotics import sgnbi, linear, power_sigmoid, hyperbolic_sine
from numpy import pi, sin, cos, sqrt

MAX_SIMULATION_TIME = 20

def desPosition(t:float) -> np.ndarray:
    return np.array([[0,0,0.5]]).T

def desOrientation(t:float) -> np.ndarray:
    return np.array([[0.5*sqrt(2)*cos(pi/18),0.5*sqrt(2)*cos(pi/18),-sin(pi/18)]]).T

Q = [[0.383, -0.587, -1.571, 1.565, -0.5, 0.2, -0.3, -1.570],
    [0.366, 0.583, 1.570, -1.575, -0.5, 0.2, -0.3, 1.571]]

PARAMETERS = dict(KphiC  = 5.0,
                  KphiE  = 5.0,
                  KphiB  = 2.0,
                  KpB    = 1.0,
                  kS     = 1.0,
                  desQ   = np.array([[0,0,0,0,0.5, -0.6, -0.4,0]]).T,
                  KGains = 5*np.array([0,0,0,0,1,1,1,0]),
                  gamma  = 5.0,
                  AF = linear,
                  updKl = False)