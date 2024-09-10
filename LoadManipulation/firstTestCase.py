from numpy import pi, sin, cos
import numpy as np

def desPosition(t:float) -> np.ndarray:
    t /= 40
    P0 = np.array([3, 0])
    P1 = np.array([0, -3])
    P2 = np.array([0, 3])
    P3 = np.array([-3, 0])
    if t > 1:
        return np.vstack(( P3[:, np.newaxis], 0.5 ))
    line = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
    pos = np.vstack(( line[:, np.newaxis], 0.5 ))
    return pos

def desOrientation(t:float) -> np.ndarray:
    if t > 40:
        t = 40 - 0.05
    dx = desPosition(t+0.05) - desPosition(t)
    dx = dx / np.linalg.norm(dx)
    dz = np.array([[0,0,1]]).T
    dy = np.cross(dz.squeeze(), dx.squeeze())[:,np.newaxis]
    dy = dy / np.linalg.norm(dy)
    return np.hstack(( dx, dy, dz ))

Q = [[3+0.212,  0.92, 3*pi/4, 0, -0.4, 1.1, 0.8708, -pi/2],
    [3+0.92,  0.212, -pi/4, 0, -0.4, 1.1, 0.8708, pi/2],
    [3-0.212, -0.92, -pi/4, 0, -0.4, 1.1, 0.8708, -pi/2],
    [3-0.92, -0.212, 3*pi/4, 0, -0.4, 1.1, 0.8708, pi/2]]