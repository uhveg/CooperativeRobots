from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robotics import *
import numpy as np
import sys

time = 0

try:
    client = RemoteAPIClient(host="localhost", port=23000)
    sim = client.require('sim')
    sim.getObject('/DefaultCamera')
    print("Connected to CoppeliaSim successfully.")
except Exception as e:
    # Handle the exception if something goes wrong
    print(f"Failed to connect to CoppeliaSim: {e}")
    sys.exit(1)  # Exit the program with an error code, or handle it as needed

def getSimulationTime() -> float:
    global time
    time = sim.getSimulationTime()
    return time

def stopSimulation() -> None:
    sim.stopSimulation()

def setStepping(value:bool) -> None:
    sim.setStepping(value)

def step() -> None:
    sim.step()

def startSimulation() -> None:
    sim.startSimulation()

class CoppeliaYoubot:
    def __init__(self, name:str, parameters) -> None:
        self.id = int(name[-1])
        self.robot = sim.getObject(f"/{name}")
        self.yScript = sim.getScript(sim.scripttype_childscript, self.robot)
        self.yFuncs = client.getScriptFunctions(self.yScript)
        self.joints = [sim.getObject(f"/{name}/youBotArmJoint{i}") for i in range(5)]

        self.x = np.zeros((3,))
        self.theta = np.zeros((5,))
        self.T = Youbot.fkine(*np.hstack((self.x,self.theta)))

        self.KL, self.alphaL, self.deltaL = 3.15, 1.0, 0.1
        # self.KphiC, self.KphiE, self.KphiB = 5.0, 5.0, 5.0
        # self.KpB = 5.0

        step_sz, self.KphiC, self.KphiE, \
            self.KphiB, self.KpB, self.kS, \
                self.desQ, self.KGains, \
                    gamma, AF, self.UPDATE_KL = parameters

        self.Kznn = znn(3.15, step_sz)

        n, m, q = 8, 7, 16
        y0 = 0.0*np.ones((m+n+q,1))
        G0 = 0.5*np.ones((m+n+q,m+n+q))
        u0 = 0.0*np.ones((m+n+q,1))
        self.NN = TVQPEIZNN(y0, G0, u0, (n,m,q), gamma=gamma, tau=step_sz, AF=AF)
    
    def applyFixedControl(self, neighbor:'CoppeliaYoubot', posB_d:np.ndarray, dir_d: np.ndarray) -> tuple[np.ndarray]:
        self.updateValues()
        pi  = self.getEndEffector()
        pj  = neighbor.getEndEffector()
        de  = (pj - pi) / np.linalg.norm(pj - pi)
        dc  = (self.x - neighbor.x)
        dc[2] = 0
        dc = dc / np.linalg.norm(dc)
        rxE = self.getEndEffectorR(axis=0)

        phi_E = 0.5 * S_operator(de).T @ rxE
        phi_C = 0.5 * S_operator(dc).T @ self.getCarRx()
        phi_B = 0.5 * S_operator(dir_d).T @ de
        
        sgnID = 1 if self.id == 1 else -1
        l12 = pi - pj
        nl12 = np.linalg.norm(l12)
        
        if self.UPDATE_KL:
            self.Kznn.updateK((nl12 - 2.0))
            self.KL = self.Kznn.k if self.id == 1 else neighbor.KL
        
        UL = -self.alphaL*(1 - (1/(self.KL*nl12))*(csch((nl12-self.deltaL)/self.KL)**2))*(l12)
        Up = self.KpB*(posB_d - 0.5*(pi+pj))
        JB = S_operator(de)
        phi_Aux = sgnID * self.KphiB * (nl12**2) * JB @ phi_B
        b = np.vstack(( UL + Up + phi_Aux, self.KphiE*phi_E))

        J = Youbot.jacob0(*np.hstack((self.x, self.theta)))
        mu = np.linalg.det(J @ J.T)
        lmax, epsilon = 0.2, 0.02
        if mu > epsilon:
            lb = 0
        else:
            lb = (1-(mu/epsilon)**2)*(lmax**2)
        pJ = J.T @ np.linalg.inv(J @ J.T + (lb**2) * np.eye(6))
        dq = pJ @ b
        dq[2,0] = self.KphiC * phi_C[2,0]

        self.setArmVelocity(np.squeeze(dq[3:]))
        self.setOmnidirectionalGlobalSpeed(np.squeeze(dq[:3]))

        cosim_derxe = (de.T @ rxE).item()
        cosim_dcrxc = (dc.T @ self.getCarRx()).item()
        cosim_ddede = (dir_d.T @ de).item()
        return (dq, nl12, de, cosim_ddede, cosim_dcrxc, cosim_derxe)

    def applyNeuralControl(self, neighbor:'CoppeliaYoubot', posB_d:np.ndarray, dir_d: np.ndarray) -> tuple[np.ndarray]:
        self.updateValues()
        # self.NN.gamma = 0.5 + 15*time/15
        q   = np.hstack((self.x, self.theta))[:,np.newaxis]
        pi  = self.getEndEffector()
        pj  = neighbor.getEndEffector()
        de  = (pj - pi) / np.linalg.norm(pj - pi)
        dc  = (self.x - neighbor.x)
        dc[2] = 0
        dc = dc / np.linalg.norm(dc)
        rxE = self.getEndEffectorR(axis=0)
        sgnID = 1 if self.id == 1 else -1

        phi_E = 0.5 * S_operator(de).T @ rxE
        phi_C = 0.5 * S_operator(dc).T @ self.getCarRx()
        phi_B = 0.5 * S_operator(dir_d).T @ de

        l12 = pi - pj
        nl12 = np.linalg.norm(l12)
        UL = -self.alphaL*(1 - (1/(self.KL*nl12))*(csch((nl12-self.deltaL)/self.KL)**2))*(l12)
        Up = self.KpB*(posB_d - 0.5*(pi+pj))

        JB = S_operator(de)
        phi_Aux = sgnID * self.KphiB * (nl12**2) * JB @ phi_B
        E2 = np.array([[0,0,1,0,0,0,0,0]])
        S = self.kS*np.eye(8)
        A = np.vstack(( E2, Youbot.jacobnn(*np.hstack((self.x, self.theta)))))
        C = np.vstack(( np.eye(8), -np.eye(8) ))
        p = np.diag(self.KGains) @ (q - self.desQ)
        
        b = np.vstack(( phi_C[2], UL + Up + phi_Aux, self.KphiE*phi_E ))
        d = calculateEta(self.theta)
        
        dq, rho, lam = self.NN.update(S, A, C, p, b, d)
        self.setArmVelocity(np.squeeze(dq[3:]))
        self.setOmnidirectionalGlobalSpeed(np.squeeze(dq[:3]))

        cosim_derxe = (de.T @ rxE).item()
        cosim_dcrxc = (dc.T @ self.getCarRx()).item()
        cosim_ddede = (dir_d.T @ de).item()

        return (A @ dq - b, dq, rho, lam, nl12, de, cosim_ddede, cosim_dcrxc, cosim_derxe, d - C @ dq)

    def getTheta(self) -> np.ndarray:
        return self.theta[:,np.newaxis]

    def getX(self) -> np.ndarray:
        return self.x[:,np.newaxis]
    
    def getQ(self) -> np.ndarray:
        return np.vstack((self.getX(), self.getTheta()))
    
    def getEndEffector(self) -> np.ndarray:
        return self.T[:3, 3][:,np.newaxis]
    
    def getEndEffectorR(self, axis:int = -1) -> np.ndarray:
        if axis == -1:
            return self.T[:3, :3]
        assert axis in [0,1,2]
        return self.T[:3, axis][:,np.newaxis]

    def getCarRx(self) -> np.ndarray:
        return np.array([[np.cos(self.x[2])],[np.sin(self.x[2])],[0]])

    def updateValues(self) -> None:
        self.getArmPosition()
        self.getRobotPosition()
        self.T = Youbot.fkine(*np.hstack((self.x,self.theta)))

    def openGrip(self, value:float) -> None:
        sim.setFloatSignal(f'y{self.id}_gripper_opening', value)

    def setOmnidirectionalSpeed(self, dxyth: list[float,float,float]) -> None:
        # dxyth = [dx,dy,dtheta] local
        r, l, d = 0.05, 0.2355, 0.15023
        w = -(1/r)*np.array([[1,-1,-l-d],\
                            [1, 1,-l-d],\
                            [1,-1, l+d],\
                            [1, 1, l+d]]) @ (np.array(dxyth).reshape((3,1)))
        w = w.squeeze()
        self.yFuncs.setWheelSpeed(w[0],w[1],w[2],w[3])
    
    def setOmnidirectionalGlobalSpeed(self, dxyth: list[float,float,float]):
        # dxyth = [dx,dy,dtheta]
        thc = self.x[2]
        x_axis = np.array([[np.cos(thc), np.sin(thc)]]).T
        y_axis = np.array([[-np.sin(thc), np.cos(thc)]]).T
        global_vector = np.array([[dxyth[0], dxyth[1]]]).reshape((2,1))
        projection_x = global_vector.T @ x_axis
        projection_y = global_vector.T @ y_axis
        dxyth[0] = projection_x.item()
        dxyth[1] = projection_y.item()
        r, l, d = 0.05, 0.2355, 0.15023
        w = -(1/r)*np.array([[1,-1,-l-d],\
                            [1, 1,-l-d],\
                            [1,-1, l+d],\
                            [1, 1, l+d]]) @ (np.array(dxyth).reshape((3,1)))
        w = w.squeeze()
        self.yFuncs.setWheelSpeed(w[0],w[1],w[2],w[3])
    
    def setArmPosition(self, q: list[float]) -> None:
        self.theta = np.array(q)
        sim.setJointPosition(self.joints[0], q[0])
        sim.setJointPosition(self.joints[1], q[1])
        sim.setJointPosition(self.joints[2], q[2])
        sim.setJointPosition(self.joints[3], q[3])
        sim.setJointPosition(self.joints[4], q[4])

    def getArmPosition(self) -> np.ndarray:
        self.theta = np.array([sim.getJointPosition(self.joints[0]),\
                        sim.getJointPosition(self.joints[1]),\
                        sim.getJointPosition(self.joints[2]),\
                        sim.getJointPosition(self.joints[3]),\
                        sim.getJointPosition(self.joints[4])])
        return self.theta

    def setArmVelocity(self, dq:list[float]) -> None:
        sim.setJointTargetVelocity(self.joints[0], dq[0])
        sim.setJointTargetVelocity(self.joints[1], dq[1])
        sim.setJointTargetVelocity(self.joints[2], dq[2])
        sim.setJointTargetVelocity(self.joints[3], dq[3])
        sim.setJointTargetVelocity(self.joints[4], dq[4])

    def getRobotPosition(self) -> np.ndarray:
        # theta = pi + theta
        # q = [sin(pi/4)*sin(x/2), -cos(x/2)*sin(pi/4), cos(pi/4)*sin(x/2), cos(pi/4)*cos(x/2)]
        pos = sim.getObjectPosition(self.robot)
        quater = sim.getObjectQuaternion(self.robot)
        theta = 2*np.arctan2(quater[2], quater[3])+np.pi
        while abs(theta) > np.pi:
            theta -= 2*np.pi*np.sign(theta)
        # theta = np.arcsin(np.sin(theta))
        self.x = np.array([pos[0], pos[1], theta])
        return self.x

    def setRobotPosition(self, pos:list[float]) -> None:
        self.x = np.array(pos)
        x = pos[2] + np.pi
        q = [np.sin(np.pi/4)*np.sin(x/2),\
            -np.cos(x/2)*np.sin(np.pi/4),\
            np.cos(np.pi/4)*np.sin(x/2),\
            np.cos(np.pi/4)*np.cos(x/2)]
        sim.setObjectPosition(self.robot, [pos[0],pos[1],0.1])
        sim.setObjectQuaternion(self.robot,q)

    def setPose(self, q:list[float]) -> None:
        self.setRobotPosition(q[:3])
        self.setArmPosition(q[3:])

    def setRandomPose(self) -> np.ndarray:
        lw = np.hstack((np.array([-4.5,-4.5,-np.pi]), Youbot.q_lower))
        up = np.hstack((np.array([4.5,4.5,np.pi]), Youbot.q_upper))
        q = np.random.rand(8,)*(up-lw) + lw
        self.setRobotPosition(q[:3])
        self.setArmPosition(q[3:])
        return q

class znn:
    def __init__(self, k0:float, tau:float) -> None:
        self.tau = tau
        self.gamma = 5.0
        self.fk = 0
        self.fk_1 = 0
        self.k = k0

    def updateK(self, f:float) -> float:
        self.fk_1 = self.fk
        self.fk = f
        pDpK = self.partialD_partialK()
        kp = (self.gamma*self.tau*self.sigma(self.fk) + self.fk - self.fk_1)
        self.k += - kp / pDpK

    def partialD_partialK(self) -> float:
        return arccsch(np.sqrt(self.k)) - 1 / (2*np.sqrt(self.k+1))

    def sigma(self, x:float) -> float:
        return x

def updateTubePose(objectID:int, p1:np.ndarray, p2:np.ndarray) -> None:
    p:np.ndarray = (p1+p2)/2
    z = np.array([0,0,1])
    z_o = (p2-p1)/np.linalg.norm(p2-p1)
    axis = np.cross(z, z_o)
    angle = np.arccos(np.dot(z, z_o))
    v = np.sin(angle/2)*axis
    sim.setObjectPosition(objectID, p.tolist())
    sim.setObjectQuaternion(objectID, [v[0] ,v[1], v[2], np.cos(angle/2)])

def calculateEta(theta):
    # first three inf
    # thetal = Youbot.th_lower + 0.3
    # thetar = Youbot.th_upper - 0.3
    # phi1 = Youbot.th_upper - Youbot.dth_upp * ((theta - thetar)**2/(Youbot.th_upper - thetar)**2)
    # phi2 = Youbot.th_lower - Youbot.dth_low * ((theta - thetal)**2/(Youbot.th_lower - thetal)**2)
    # eta_plus = np.where(theta>thetar, phi1, Youbot.dth_upp)
    # eta_minus = np.where(theta<thetal, phi2, Youbot.dth_low)

    xi_p = np.minimum(0.5*Youbot.dth_upp, 0.1*(Youbot.th_upper-theta))
    xi_m = np.maximum(0.5*Youbot.dth_low, 0.1*(Youbot.th_lower-theta))


    d = np.vstack((
        0.5*np.ones((3, 1)),
        xi_p[:,np.newaxis],
        0.5*np.ones((3, 1)),
        -xi_m[:,np.newaxis]
    ))
    # d = 1e8*np.ones((16, 1))
    return d


def perturbation(t:float, id:int) -> np.ndarray:
    # if id == 1:
    return np.zeros((3,1))
    # return np.array([[
    #     3*((0.5 + 0.5*np.tanh(t-10)) - (0.5 + 0.5*np.tanh(t-12))),
    #     0,
    #     0
    # ]]).T