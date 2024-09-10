from Coppelia import *
import numpy as np
from numpy import pi, sin, cos, exp
from logger import Log, DEFAULT_LOG_COLUMNS

## UNCOMMENT THE TEST CASE TO IMPORT:
#   - Initial values
#   - Parameters
#   - Desired values

# from firstTestCase import *
# from secondTestCase import *
from comparativeTestCase import *

STEP_SIZE = 0.05

np.set_printoptions(precision=4, suppress=True)
logger = Log("test", columns=DEFAULT_LOG_COLUMNS + 
             ['cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2', 'dpb', 'dde', 'K1', 'K2'], replace=True)

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
        logy1 = youbot1.applyFixedControl(youbot2, desPosition(t), desOrientation(t))
        logy2 = youbot2.applyFixedControl(youbot1, desPosition(t), desOrientation(t))
        step()
        t = getSimulationTime()
        updateTubePose(tube, np.squeeze(youbot1.getEndEffector()), np.squeeze(youbot2.getEndEffector()))
        logger.write(['time', 'ee1', 'ee2', 'Youbot1', 'Youbot2', 'dpb', 'dde', 'K1', 'K2',
                    'Youbot1_d', 'nl12', 'de1', 'cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1',
                    'Youbot2_d', 'nl21', 'de2', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2'], 
                    (t, youbot1.getEndEffector(), youbot2.getEndEffector(), 
                    youbot1.getQ(), youbot2.getQ(), desPosition(t), desOrientation(t), youbot1.KL, youbot2.KL) + logy1 + logy2)
    print("DESIRED V:")
    print(desOrientation(MAX_SIMULATION_TIME))
    print("TUBE POSITION:")
    print((youbot1.getEndEffector() + youbot2.getEndEffector())/2)
    print("DE1 & ACTUAL:")
    print(logy1[5])
    print("LIJ")
    print(logy1[1])
except Exception as e:
    print(e)
finally:
    stopSimulation()
    setStepping(False)
    logger.close()