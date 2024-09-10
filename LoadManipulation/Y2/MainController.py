from Coppelia import *
import numpy as np
from numpy import pi, sin, cos
from logger import *


## UNCOMMENT THE TEST CASE TO IMPORT:
#   - Initial values
#   - Parameters
#   - Desired values

# from firstTestCase import *
from secondTestCase import *
# from thirdTestCase import *


np.set_printoptions(precision=4, suppress=True)
STEP_SIZE = 0.05 # For approximations in ZNN

def getLoadLog(load:CoppeliaPanel, dR:np.ndarray) -> list[np.ndarray]:
    p = load.p
    R = load.R
    cosimx = np.dot(R[:,0], dR[:,0])
    cosimy = np.dot(R[:,1], dR[:,1])
    return [p, R, cosimx, cosimy]

youbots = [CoppeliaYoubot(f"youBot[{i}]", (STEP_SIZE, ) + tuple(PARAMETERS.values())) for i in range(4)]
sim.setFloatSignal('y_gripper_opening', 0.0)

for i in range(len(youbots)):
    youbots[i].setPose(Q[i])

for ybot in youbots:
    ybot.updateValues()

youbots[0].addNeighbor(youbots[1], 0.9)
youbots[0].addNeighbor(youbots[3], 3.8)
youbots[1].addNeighbor(youbots[0], 0.9)
youbots[1].addNeighbor(youbots[2], 3.8)
youbots[2].addNeighbor(youbots[3], 0.9)
youbots[2].addNeighbor(youbots[1], 3.8)
youbots[3].addNeighbor(youbots[2], 0.9)
youbots[3].addNeighbor(youbots[0], 3.8)

KNeurons = [[0, 1, znn(0.9, STEP_SIZE)], [2, 3, znn(0.9, STEP_SIZE)], [0, 3, znn(3.8, STEP_SIZE)], [1, 2, znn(3.8, STEP_SIZE)]]

solar_panel = CoppeliaPanel('SolarPanel', [0, 0, 0.5], [f"/youBot[{i}]/Rectangle7" for i in range(4)],
                            [youbots[i].getEndEffector().squeeze() for i in range(4)])
# exit(-1)
database = Log("test", replace=True) # Comment if not need to log for plot latter

pb = lambda t: desPosition(t)
db = lambda t: desOrientation(t)

try:
    setStepping(True)
    startSimulation()
    while (t := getSimulationTime()) < MAX_SIMULATION_TIME:
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
                print("ERROR OCURRED: check the parameters, initial conditions, and that the desired values are within the workspace")
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
except Exception as e:
    print(e)
finally:
    stopSimulation()
    setStepping(False)
    database.close()

