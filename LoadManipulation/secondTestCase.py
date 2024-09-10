import numpy as np
from robotics import Rotx, Roty, Rotz
from numpy import pi, sin, cos

def desPosition(t:float) -> np.ndarray:
    return np.array([[0,0,0.35]]).T
    if t < 15:
        z = 0.5 - 0.25*t/15
        return np.array([[0,0,z]]).T
    if t < 20:
        z = 0.25 + 0.1*(t-15)/5
        return np.array([[0,0,z]]).T
    return np.array([[0,0,0.35]]).T

def desOrientation(t:float) -> np.ndarray:
    # if t < 10*pi/2:
    #     return Rotz(-np.pi/2)
    omega = 0.1*t
    alpha = (np.pi/20)*sin(omega)
    beta = (np.pi/20)*cos(omega)
    return Rotx(alpha) @ Roty(beta) @ Rotz(-np.pi/2)

    if t < 10*(3*pi/2):
        a = (np.pi/20)*sin(0.1*(t-10*pi/2))
        return Rotx(a) @ Rotz(-np.pi/2)
    a = (np.pi/14)*sin(0.1*(t-10*(3*pi/2)))
    return Roty(a) @ Rotz(-np.pi/2)
    if t < 20:
        return Rotz(-np.pi/2)
    if t < 30:
        a = ((t-20)/10)*np.pi/20
        # a = np.pi/20
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
    # if t < 30:
    #     return Rotx(np.pi/60) @ Rotz(-np.pi/2)
    # if t < 40:
    #     return Rotz(-np.pi/2)
    # if t < 50:
    #     return Rotx(-np.pi/60) @ Rotz(-np.pi/2)
    # if t < 60:
    #     return Rotz(-np.pi/2)
    # if t < 70:
    #     return Roty(np.pi/30) @ Rotz(-np.pi/2)
    # if t < 80:
    #     return Rotz(-np.pi/2)
    # if t < 90:
    #     return Roty(-np.pi/30) @ Rotz(-np.pi/2)
    return Rotx(-np.pi/20) @ Rotz(-np.pi/2)

Q = [[3+0.212,  0.92, 3*pi/4, 0, -0.4, 1.1, 0.8708, -pi/2],
    [3+0.92,  0.212, -pi/4, 0, -0.4, 1.1, 0.8708, pi/2],
    [3-0.212, -0.92, -pi/4, 0, -0.4, 1.1, 0.8708, -pi/2],
    [3-0.92, -0.212, 3*pi/4, 0, -0.4, 1.1, 0.8708, pi/2]]