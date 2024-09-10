from Coppelia import *
import numpy as np
from numpy import pi, sin, cos
from logger import *
import traceback

np.set_printoptions(precision=4, suppress=True)

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

# def desPosition(t:float) -> np.ndarray:
#     t /= 40
#     P0 = np.array([3, 0])
#     P1 = np.array([0, -3])
#     P2 = np.array([0, 3])
#     P3 = np.array([-3, 0])
#     if t > 1:
#         return np.vstack(( P3[:, np.newaxis], 0.5 ))
#     line = (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
#     pos = np.vstack(( line[:, np.newaxis], 0.5 ))
#     return pos

# def desOrientation(t:float) -> np.ndarray:
#     if t > 40:
#         t = 40 - 0.05
#     dx = desPosition(t+0.05) - desPosition(t)
#     dx = dx / np.linalg.norm(dx)
#     dz = np.array([[0,0,1]]).T
#     dy = np.cross(dz.squeeze(), dx.squeeze())[:,np.newaxis]
#     dy = dy / np.linalg.norm(dy)
#     return np.hstack(( dx, dy, dz ))

def getLoadLog(load:CoppeliaPanel, dR:np.ndarray) -> list[np.ndarray]:
    p = load.p
    R = load.R
    cosimx = np.dot(R[:,0], dR[:,0])
    cosimy = np.dot(R[:,1], dR[:,1])
    return [p, R, cosimx, cosimy]

# N, k = 4, 5

youbots = [CoppeliaYoubot(f"youBot[{i}]") for i in range(4)]
sim.setFloatSignal('y_gripper_opening', 0.0)

# youbots[0].setPose([-0.75, 0.63, pi/2, -pi/2, 0.4, -1.1, -0.870796, pi/2])
# youbots[1].setPose([0.75, 0.63, pi/2, pi/2, 0.4, -1.1, -0.870796, -pi/2])
# youbots[2].setPose([0.75, -0.63, -pi/2, -pi/2, 0.4, -1.1, -0.870796, pi/2])
# youbots[3].setPose([-0.75, -0.63, -pi/2, pi/2, 0.4, -1.1, -0.870796, -pi/2])

# youbots[0].setPose([-0.9, 0.8, 0, 0, 0.4, -1.1, -0.870796, pi/2])
# youbots[1].setPose([0.9, 0.8, pi, 0, 0.4, -1.1, -0.870796, -pi/2])
# youbots[2].setPose([0.9, -0.8, pi, 0, 0.4, -1.1, -0.870796, pi/2])
# youbots[3].setPose([-0.9, -0.8, 0, 0, 0.4, -1.1, -0.870796, -pi/2])

# youbots[0].setPose([3+0.212,  0.92, 3*pi/4, 0, -0.4, 1.1, 0.8708, -pi/2])
# youbots[1].setPose([3+0.92,  0.212, -pi/4, 0, -0.4, 1.1, 0.8708, pi/2])
# youbots[2].setPose([3-0.212, -0.92, -pi/4, 0, -0.4, 1.1, 0.8708, -pi/2])
# youbots[3].setPose([3-0.92, -0.212, 3*pi/4, 0, -0.4, 1.1, 0.8708, pi/2])

youbots[0].setPose([-1.113, 0.8, 0, 0, -0.7, -1.0, 0.135, pi/2])
youbots[1].setPose([1.113, 0.8, pi, 0, -0.7, -1.0, 0.135, -pi/2])
youbots[2].setPose([1.113, -0.8, pi, 0, -0.7, -1.0, 0.135, pi/2])
youbots[3].setPose([-1.113, -0.8, 0, 0, -0.7, -1.0, 0.135, -pi/2])

for ybot in youbots:
    ybot.updateValues()
    # print(ybot.getEndEffector())
# exit()
youbots[0].addNeighbor(youbots[1], 0.9)
youbots[0].addNeighbor(youbots[3], 3.8)
youbots[1].addNeighbor(youbots[0], 0.9)
youbots[1].addNeighbor(youbots[2], 3.8)
youbots[2].addNeighbor(youbots[3], 0.9)
youbots[2].addNeighbor(youbots[1], 3.8)
youbots[3].addNeighbor(youbots[2], 0.9)
youbots[3].addNeighbor(youbots[0], 3.8)



KNeurons = [[0, 1, znn(0.9, 0.05)], [2, 3, znn(0.9, 0.05)], [0, 3, znn(3.8, 0.05)], [1, 2, znn(3.8, 0.05)]]

solar_panel = CoppeliaPanel('SolarPanel', [0, 0, 0.5], [f"/youBot[{i}]/Rectangle7" for i in range(4)],
                            [youbots[i].getEndEffector().squeeze() for i in range(4)])
# exit()
database = Log("Comparative", replace=True)

pb = lambda t: desPosition(t)
db = lambda t: desOrientation(t)

try:
    setStepping(True)
    startSimulation()
    while (t := getSimulationTime()) < 10*2*2*pi:
        colOrder = ['time']
        values = [t]
        for k in KNeurons:
            id_i, id_j, Kz = k
            youbots[id_i].updateK(id_j, Kz.k)
        for ybot in youbots:
            logs = ybot.applyNeuralControl(solar_panel, pb(t), db(t))
            colOrder += [f'q{ybot.id}', f'dq{ybot.id}', f'ee{ybot.id}', f'rho{ybot.id}', f'zeta{ybot.id}', 
                         f'EQ{ybot.id}', f'IQ{ybot.id}', f'SC_eex{ybot.id}', f'SC_eez{ybot.id}']
            values += logs
            if np.max(np.abs(logs[0])) > 200:
                stopSimulation()
                setStepping(False)
                exit(-1)
        for i, k in enumerate(KNeurons):
            id_i, id_j, Kz = k
            Kz:znn
            lij = youbots[id_i].getEndEffector() - youbots[id_j].getEndEffector()
            nlij = np.linalg.norm(lij)
            colOrder += [f'nl{id_i}{id_j}', f'K{id_i}{id_j}']
            values += [nlij, Kz.k]
            # print(f"{id_i}-{id_j} : {nlij}")
            if i < 2:
                Kz.updateK(nlij - 0.95)
            else:
                Kz.updateK(nlij - 1.6)
        colOrder += ['loadP', 'loadR', 'SC_loadx', 'SC_loady']
        values += getLoadLog(solar_panel, db(t))
        colOrder += ['desiredP', 'desiredR']
        values += [pb(t), db(t)]
        database.write(colOrder, values)

        step()
        solar_panel.updatePose([youbots[i].getEndEffector().squeeze() for i in range(4)])
    print(db(t), end='\n\n')
    print(solar_panel.R)
    print(solar_panel.p)
except Exception as e:
    print(e)
    traceback.print_exc()
finally:
    stopSimulation()
    setStepping(False)
    database.close()

