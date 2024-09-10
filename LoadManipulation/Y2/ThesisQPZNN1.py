from Coppelia import *
import numpy as np
from numpy import pi, sin, cos
from logger import Log, DEFAULT_LOG_COLUMNS

from comparativeTestCase import *

STEP_SIZE = 0.05

np.set_printoptions(precision=2, suppress=True)
logger = Log("test_tvqpei", columns=DEFAULT_LOG_COLUMNS + 
             ['EqConstraints1', 'EqConstraints2', 'rho1', 'rho2', 'zeta1', 'zeta2', 'Ineq1', 'Ineq2'] +
             ['cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2', 'dpb', 'dde'], replace=True)

youbot1 = CoppeliaYoubot('youBot1', (STEP_SIZE, ) + tuple(PARAMETERS.values()))
youbot2 = CoppeliaYoubot('youBot2', (STEP_SIZE, ) + tuple(PARAMETERS.values()))


youbot1.openGrip(1.0)
youbot2.openGrip(1.0)
tube = sim.getObject('/tube')

youbot1.setPose(Q[0])
youbot2.setPose(Q[1])

youbot1.updateValues()
youbot2.updateValues()

updateTubePose(tube, np.squeeze(youbot1.getEndEffector()), np.squeeze(youbot2.getEndEffector()))
setStepping(True)
startSimulation()
step()
t = getSimulationTime()

try:
    while t < MAX_SIMULATION_TIME:
        logy1 = youbot1.applyNeuralControl(youbot2, desPosition(t), desOrientation(t))
        logy2 = youbot2.applyNeuralControl(youbot1, desPosition(t), desOrientation(t))
        step()
        t = getSimulationTime()
        updateTubePose(tube, np.squeeze(youbot1.getEndEffector()), np.squeeze(youbot2.getEndEffector()))
        logger.write(['time', 'ee1', 'ee2', 'Youbot1', 'Youbot2', 'dpb', 'dde',
                    'EqConstraints1', 'Youbot1_d', 'rho1', 'zeta1', 'nl12', 'de1', 'cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1', 'Ineq1',
                    'EqConstraints2', 'Youbot2_d', 'rho2', 'zeta2', 'nl21', 'de2', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2', 'Ineq2'], 
                    (t, youbot1.getEndEffector(), youbot2.getEndEffector(), youbot1.getQ(), youbot2.getQ(), desPosition(t), desOrientation(t)) + logy1 + logy2)
        if (np.max(np.abs(logy1[1])) > 100 or  np.max(np.abs(logy2[1])) > 100):
            break
    print("DESIRED V:")
    print(desOrientation(MAX_SIMULATION_TIME))
    print("TUBE POSITION:")
    print((youbot1.getEndEffector() + youbot2.getEndEffector())/2)
    print("DE1 & ACTUAL:")
    print(logy1[5])
    print("LIJ")
    print(logy1[4])
except Exception as e:
    print(e)
finally:
    stopSimulation()
    setStepping(False)
    logger.close()