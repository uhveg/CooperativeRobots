from Coppelia import *
import numpy as np
from numpy import pi, sin, cos
from logger import Log, DEFAULT_LOG_COLUMNS

np.set_printoptions(precision=2, suppress=True)
logger = Log("QP_5_THESIS_LINEAR_2", columns=DEFAULT_LOG_COLUMNS + 
             ['EqConstraints1', 'EqConstraints2', 'rho1', 'rho2', 'zeta1', 'zeta2', 'Ineq1', 'Ineq2'] +
             ['cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2', 'dpb', 'dde'], replace=True)

youbot1 = CoppeliaYoubot('youBot1')
youbot2 = CoppeliaYoubot('youBot2')


youbot1.openGrip(1.0)
youbot2.openGrip(1.0)
tube = sim.getObject('/tube')

# youbot1.setPose([0.383, -0.587, -1.571, 1.565, 0.5, -0.6, -0.3, -1.570,])
# youbot2.setPose([0.366, 0.583, 1.570, -1.575, 0.5, -0.6, -0.3, 1.571,])
youbot1.setPose([0.383, -0.587, -1.571, 1.565, -0.5, 0.2, -0.3, -1.570,])
youbot2.setPose([0.366, 0.583, 1.570, -1.575, -0.5, 0.2, -0.3, 1.571])
# youbot1.setPose([0, -0.6, -1.57, 1.57, -0.5, 0.2, -0.3, -1.57])
# youbot2.setPose([0, 0.6, 1.57, -1.57, -0.5, 0.2, -0.3, 1.57,])
# youbot1.setPose([1, -0.6, -1.57, 1.57, 0.5, -0.8, -0.7, -1.57])
# youbot2.setPose([1, 0.6, 1.57, -1.57, 0.5, -0.8, -0.7, 1.57])
# youbot1.setPose([1, -0.6, -1.57, -1.57-0.2, 0.9, -0.8, -0.7, 1.57])
# youbot2.setPose([0.4, 0.6, 1.57, -1.57+0.2, 0.5, -0.8, -0.7, 0.5*1.57])
# youbot1.setPose([-0.17, -0.6, -1.57, 1.57, 0.9, -1.3, -1.2, -1.57])
# youbot2.setPose([-0.17, 0.6, 1.57, -1.57, 0.9, -1.3, -1.2, 1.57])

# exit(-1)
youbot1.updateValues()
youbot2.updateValues()

# pi = youbot1.getEndEffector()
# pj = youbot2.getEndEffector()
# de = (pj - pi) / np.linalg.norm(pj - pi)
# print(f"Pb = {0.5*(pi + pj)}")
# print(f"nlij = {np.linalg.norm(pj - pi)}")
# print(f"{de=}")
# exit()

youbot1.setArmVelocity([0,0,0,0,0])
youbot2.setArmVelocity([0,0,0,0,0])
# exit()

updateTubePose(tube, np.squeeze(youbot1.getEndEffector()), np.squeeze(youbot2.getEndEffector()))
setStepping(True)
startSimulation()
step()
t = getSimulationTime()

TS = 0.1
alpha, beta = np.pi/18, np.pi/4
pB = lambda t: np.array([[0,0, 0.5+0*0.05*sin(0.5*t)]]).T
dirB = lambda t: np.array([[np.cos(alpha)*np.cos(beta), np.sin(beta)*np.cos(alpha), -np.sin(alpha)]]).T

# R, v, c = 1, 0.2, 0.002
# pB = lambda t: np.array([[R * np.cos(v*t/R), 
#                           R * np.sin(v*t/R), 
#                           0.45 + c*t]]).T
# dirB = lambda t: np.array([[1,1,0]]).T * (pB(t+0.05) - pB(t))/np.linalg.norm(pB(t+0.05)[:2] - pB(t)[:2])

# a, b, c = 1, 0.05, 0.02
# pB = lambda t: np.array([[(a + b * t) * np.cos(0.1*t), 
#                           (a + b * t) * np.sin(0.1*t), 
#                           0.45 + c * (0.1*t)]]).T
# dirB = lambda t: np.array([[1,1,0]]).T * (pB(t+0.05) - pB(t))/np.linalg.norm(pB(t+0.05)[:2] - pB(t)[:2])

# pB = lambda t: np.array([[1, 0, 0.65-0.2*(sin(0.2*t)**2)]]).T

# alpha = lambda t: (0.5 + 0.5*np.tanh(t-60))*(np.pi/18)*np.sin(0.3*(t-60))
# beta = lambda t: pi/2 + 0.1*t
# pB = lambda t: np.array([[0,0,0.95 + 0.5*np.sin(0.2*t)*np.exp(-0.1*t) - 0.8*(1 - np.exp(-0.1*(t+10))) + (0.17 + 0.17*np.tanh(0.1*(t-40)))]]).T
# dirB = lambda t: np.array([[np.cos(alpha(t))*np.cos(beta(t)), np.sin(beta(t))*np.cos(alpha(t)), -np.sin(alpha(t))]]).T


while t < 20:
    logy1 = youbot1.applyNeuralControl(youbot2, pB(t), dirB(t))
    logy2 = youbot2.applyNeuralControl(youbot1, pB(t), dirB(t))
    step()
    t = getSimulationTime()
    updateTubePose(tube, np.squeeze(youbot1.getEndEffector()), np.squeeze(youbot2.getEndEffector()))
    logger.write(['time', 'ee1', 'ee2', 'Youbot1', 'Youbot2', 'dpb', 'dde',
                  'EqConstraints1', 'Youbot1_d', 'rho1', 'zeta1', 'nl12', 'de1', 'cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1', 'Ineq1',
                  'EqConstraints2', 'Youbot2_d', 'rho2', 'zeta2', 'nl21', 'de2', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2', 'Ineq2'], 
                 (t, youbot1.getEndEffector(), youbot2.getEndEffector(), youbot1.getQ(), youbot2.getQ(), pB(t), dirB(t)) + logy1 + logy2)
    if (np.max(np.abs(logy1[1])) > 100 or  np.max(np.abs(logy2[1])) > 100):
        break
print("DESIRED V:")
print(dirB(0))
print("TUBE POSITION:")
print((youbot1.getEndEffector() + youbot2.getEndEffector())/2, end='\n\n\n')
print("DE1 & ACTUAL:")
print(logy1[5])
print(youbot1.getEndEffectorR(axis=0), end='\n\n\n')
print("DE2 & ACTUAL:")
print(logy1[5])
print(youbot1.getEndEffectorR(axis=0))
youbot1.setArmVelocity([0,0,0,0,0])
youbot1.setArmVelocity([0,0,0,0,0])
stopSimulation()
setStepping(False)

logger.close()