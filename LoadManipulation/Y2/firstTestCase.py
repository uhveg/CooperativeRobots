from numpy import pi, sin, cos
import numpy as np
from robotics import sgnbi, linear

MAX_SIMULATION_TIME = 30

def desPosition(t:float) -> np.ndarray:
    return np.array([[0,1,0.5]]).T

def desOrientation(t:float) -> np.ndarray:
    return np.array([[0,1,0]]).T

Q = [[1,  0, pi/2, 0, 0.3, -1.3, -0.8, 0],
    [-1,  0, pi/2, 0, 0.3, -1.3, -0.8, 0]]

PARAMETERS = dict(KphiC  = 1.0,
                  KphiE  = 1.0,
                  KphiB  = 1.0,
                  KpB    = 1.0,
                  kS     = 1.0,                                             # not used
                  desQ   = np.array([[0,0,0,0,0.3,-1.3,0.8,0]]).T,          # not used
                  KGains = np.array([0,0,0,0,1,1,1,0]),                     # not used
                  gamma  = 1.0,                                             # not used
                  AF = linear,                                              # not used
                  updKl = False)