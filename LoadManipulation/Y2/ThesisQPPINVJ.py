from Coppelia import *
import numpy as np
from numpy import pi, sin, cos, exp
from logger import Log, DEFAULT_LOG_COLUMNS

np.set_printoptions(precision=2, suppress=True)
logger = Log("PINVJ_12_THESIS_znnK_linear", columns=DEFAULT_LOG_COLUMNS + 
             ['cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2', 'dpb', 'dde', 'K1', 'K2'], replace=True)

youbot1 = CoppeliaYoubot('youBot1')
youbot2 = CoppeliaYoubot('youBot2')

youbot1.openGrip(1.0)
youbot2.openGrip(1.0)
tube = sim.getObject('/tube')

# youbot1.setPose([0.383, -0.587, -1.571, 1.565, 0.798, 0.919, -0.720, -1.570,])
# youbot2.setPose([0.366, 0.583, 1.570, -1.575, 0.610, 1.167, -0.718, 1.571,])
# youbot1.setPose([0.383, -0.587, -1.571, 1.565, -0.5, 0.2, -0.3, -1.570,])
# youbot2.setPose([0.366, 0.583, 1.570, -1.575, -0.5, 0.2, -0.3, 1.571,])
# youbot1.setPose([1, -0.6, -1.57, -1.57-0.2, 0.9, -0.8, -0.7, 1.57])
# youbot2.setPose([0.4, 0.6, 1.57, -1.57+0.2, 0.5, -0.8, -0.7, 0.5*1.57])
# youbot1.setPose([1, -0.6, -1.57, 1.57, 0.9, -1.3, -1.2, -1.57])
# youbot2.setPose([1, 0.6, 1.57, -1.57, 0.9, -1.3, -1.2, 1.57])
# youbot1.setPose([-0.17, -0.6, -1.57, 1.57, 0.9, -1.3, -1.2, -1.57])
# youbot2.setPose([-0.17, 0.6, 1.57, -1.57, 0.9, -1.3, -1.2, 1.57])

youbot1.setPose([2.3,0.6, np.pi/2, np.pi/2, 0.3, -1, -0.8, -np.pi/2])
youbot2.setPose([2.3,-0.6, -np.pi/2, -np.pi/2, 0.3, -1, -0.8, np.pi/2])

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

# R, v, c = 1, 0.2, 0.002
# pB = lambda t: np.array([[R * np.cos(v*t/R), 
#                           R * np.sin(v*t/R), 
#                           0.45 + c*t]]).T
# dirB = lambda t: np.array([[1,1,0]]).T * (pB(t+0.05) - pB(t))/np.linalg.norm(pB(t+0.05)[:2] - pB(t)[:2])
# alpha, beta = 0, np.pi/2
# alpha = lambda t: 0 + 0*(np.pi/18)#*np.sin(0.3*t)
# beta = lambda t: pi/2
# pB = lambda t: np.array([[1, 0, 0.65-0.2*(sin(0.2*t)**2)]]).T
# dirB = lambda t: np.array([[np.cos(alpha(t))*np.cos(beta(t)), np.sin(beta(t))*np.cos(alpha(t)), -np.sin(alpha(t))]]).T
# alpha = lambda t: (0.5 + 0.5*np.tanh(t-60))*(np.pi/18)*np.sin(0.3*(t-60))
# beta = lambda t: pi/2 + 0.1*t
# pB = lambda t: np.array([[0,0,0.95 + 0.5*np.sin(0.2*t)*np.exp(-0.1*t) - 0.8*(1 - np.exp(-0.1*(t+10))) + (0.17 + 0.17*np.tanh(0.1*(t-40)))]]).T
# dirB = lambda t: np.array([[np.cos(alpha(t))*np.cos(beta(t)), np.sin(beta(t))*np.cos(alpha(t)), -np.sin(alpha(t))]]).T

alpha = lambda t: 0.3*t + 0.3*t*np.sin(0.1*t)**2
beta = lambda t: pi/2
pB = lambda t: np.array([[2*cos(0.1*t),2*sin(0.1*t),0.45]]).T
dirB = lambda t: np.array([[sin(0.1*t), -cos(0.1*t), 0]]).T
# dirB = lambda t: np.array([[np.cos(alpha(t))*np.cos(beta(t)), np.sin(beta(t))*np.cos(alpha(t)), -np.sin(alpha(t))]]).T

while t < 60:
    logy1 = youbot1.applyFixedControl(youbot2, pB(t), dirB(t))
    logy2 = youbot2.applyFixedControl(youbot1, pB(t), dirB(t))
    step()
    t = getSimulationTime()
    updateTubePose(tube, np.squeeze(youbot1.getEndEffector()), np.squeeze(youbot2.getEndEffector()))
    logger.write(['time', 'ee1', 'ee2', 'Youbot1', 'Youbot2', 'dpb', 'dde', 'K1', 'K2',
                  'Youbot1_d', 'nl12', 'de1', 'cosim_ddede1', 'cosim_dcrxc1', 'cosim_derxe1',
                  'Youbot2_d', 'nl21', 'de2', 'cosim_ddede2', 'cosim_dcrxc2', 'cosim_derxe2'], 
                 (t, youbot1.getEndEffector(), youbot2.getEndEffector(), 
                  youbot1.getQ(), youbot2.getQ(), pB(t), dirB(t), youbot1.KL, youbot2.KL) + logy1 + logy2)
print("DESIRED V:")
print(dirB(0))
print("TUBE POSITION:")
print((youbot1.getEndEffector() + youbot2.getEndEffector())/2)
print("DE1 & ACTUAL:")
print(logy1[2])
print("LIJ")
print(logy1[1])
youbot1.setArmVelocity([0,0,0,0,0])
youbot1.setArmVelocity([0,0,0,0,0])
stopSimulation()
setStepping(False)

logger.close()