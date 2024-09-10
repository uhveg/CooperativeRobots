from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robotics import *
import numpy as np
import sys

time = 0
BASE_ORIENTATION = 1

try:
    # Use localhost if Coppelia is running on the same machine
    client = RemoteAPIClient(host="192.168.100.51", port=23000)
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
    def __init__(self, name:str) -> None:
        self.id = int(name[-2])
        self.robot = sim.getObject(f"/{name}")
        self.yScript = sim.getScript(sim.scripttype_childscript, self.robot)
        self.yFuncs = client.getScriptFunctions(self.yScript)
        self.joints = [sim.getObject(f"/{name}/youBotArmJoint{i}") for i in range(5)]

        self.grippers = [sim.getObject(f"/{name}/youBotGripperJoint1"), \
                         sim.getObject(f"/{name}/youBotGripperJoint2")]
        sim.setJointPosition(self.grippers[1], -0.015)
        sim.setJointPosition(self.grippers[0], 0.003)

        self.neighbors: dict[str, list['CoppeliaYoubot'] | list[float]] = dict(youbot=[], K=[])
        self.orientation_index = dict(rxE=1, rzE=0, rxC=0)
        self.error_pos = deque(maxlen=2)

        self.x = np.zeros((3,))
        self.theta = np.zeros((5,))
        self.T = Youbot.fkine(*np.hstack((self.x,self.theta)))

        self.KphiC, self.KphiE, self.KphiB = 5.0, 5.0, 5.0
        self.KpB = 3.0
        self.kS = 20

        self.Kznn = znn(3.15, 0.05)
        n, m, q = 8, 7, 16
        y0 = 0.0*np.ones((m+n+q,1))
        G0 = 0.5*np.ones((m+n+q,m+n+q))
        u0 = 0.0*np.ones((m+n+q,1))
        self.NN = TVQPEIZNN(y0, G0, u0, (n,m,q), gamma=10.0, tau=0.05)

    def applyNeuralControl(self, load:'CoppeliaPanel', posB_d:np.ndarray, RB_d: np.ndarray) -> list[np.ndarray]:
        self.updateValues()
        # self.NN.gamma = 1 + 15*time/15
        q   = np.hstack((self.x, self.theta))[:,np.newaxis]
        phi_E, cosimx, cosimz = self.endEffectorOrientation()
        phi_C = self.baseOrientation()

        UL = self.edgeWeighted()
        Up = self.pd_position_error(posB_d, load.p[:, np.newaxis])

        phi_Aux = self.loadOrientation(load.p[:,np.newaxis], load.R, RB_d)
        E2 = np.array([[0,0,1,0,0,0,0,0]])
        Sdiag = self.kS*np.array([1,1,1,1,1,1,1,1])
        S = np.diag(Sdiag)
        # S[:3,:3] *= 0.05
        A = np.vstack(( E2, Youbot.jacobnn(*np.hstack((self.x, self.theta)))))
        # A[3,:] *= 10
        C = np.vstack(( np.eye(8), -np.eye(8) ))
        # p = -5*(np.array([[0.1,0.1,0.1,0.1,2,2,0.1,0.1]]).T)*np.array([[0,0,0,0,-0.4 - q[4,0],1.1 - q[5,0],0.8708 - q[6,0],0]]).T
        desQ = np.array([[0,0,0,0,0.4,0.8,0.8708,0]]).T
        KGains = np.array([0,0,0,0,1,1,1,0])
        p = np.diag(KGains) @ (q - desQ)
        
        b = np.vstack(( phi_C[2], UL + Up + phi_Aux, self.KphiE*phi_E ))
        d = calculateEta(self.theta)

        # dq = np.zeros((8,1))
        # dq = np.linalg.pinv(A[1:,:]) @ b[1:]

        dq, rho, lam = self.NN.update(S, A, C, p, b, d)
        self.setArmVelocity(np.squeeze(dq[3:]))
        self.setOmnidirectionalGlobalSpeed(np.squeeze(dq[:3]))


        return [q, dq, self.getEndEffector(), rho, lam, A @ dq - b, d - C @ dq, cosimx, cosimz]

    def loadOrientation(self, panel: np.ndarray, R:np.ndarray, Rd:np.ndarray) -> None:
        pi = self.getEndEffector()
        dpb = (panel - pi) / np.linalg.norm(panel - pi)
        phi  = 0
        phi += (np.linalg.norm(panel - pi)**2) * S_operator(dpb) @ S_operator(Rd[:,0]).T @ R[:,[0]]
        phi += (np.linalg.norm(panel - pi)**2) * S_operator(dpb) @ S_operator(Rd[:,1]).T @ R[:,[1]]
        return 2*phi

    def pd_position_error(self, pd, p) -> None:
        e = self.KpB*(pd - p)
        self.error_pos.append(e.copy())
        return e

    def baseOrientation(self) -> np.ndarray:
        dc = BASE_ORIENTATION * (self.x - self.neighbors['youbot'][self.orientation_index['rxC']].x)
        dc[2] = 0
        dc = dc / np.linalg.norm(dc)
        return 0.5 * S_operator(dc).T @ self.getCarRx()

    def endEffectorOrientation(self) -> np.ndarray:
        pi = self.getEndEffector()
        pjx  = self.neighbors['youbot'][self.orientation_index['rxE']].getEndEffector()
        pjz  = self.neighbors['youbot'][self.orientation_index['rzE']].getEndEffector()
        dex  = (pjx - pi) / np.linalg.norm(pjx - pi)
        dez  = (pjz - pi) / np.linalg.norm(pjz - pi)
        rxE = self.getEndEffectorR(axis=0)
        rzE = self.getEndEffectorR(axis=2)
        return (0.5 * ( S_operator(dex).T @ rxE + S_operator(dez).T @ rzE ), (dex.T @ rxE).item(), (dez.T @ rzE).item())

    def edgeWeighted(self) -> np.ndarray:
        UL = np.zeros((3,1))
        pi = self.getEndEffector()
        alpha, delta = 1.0, 0.1
        for youbot, K in zip(self.neighbors['youbot'], self.neighbors['K']):
            pj = youbot.getEndEffector()
            l12 = pi - pj
            nl12 = np.linalg.norm(l12)
            UL -= alpha*(1 - (1/(K*nl12))*(csch((nl12-delta)/K)**2))*(l12)
        return UL

    def addNeighbor(self, bot:'CoppeliaYoubot', K:float) -> None:
        if bot not in self.neighbors['youbot']:
            self.neighbors['youbot'].append(bot)
            self.neighbors['K'].append(K)

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

    def getNeighbor(self, id:int):
        for index, n in enumerate(self.neighbors['youbot']):
            if n.id == id:
                return index, n
        return None
    
    def updateK(self, id:int, K:float) -> None:
        if (tup := self.getNeighbor(id)) is None:
            return
        j, n = tup
        n: 'CoppeliaYoubot'
        self.neighbors['K'][j] = K
        i, _ = n.getNeighbor(self.id)
        n.neighbors['K'][i] = K

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
    xi_p = np.minimum(Youbot.dth_upp, 0.1*(Youbot.th_upper-theta))
    xi_m = np.maximum(Youbot.dth_low, 0.1*(Youbot.th_lower-theta))

    d = np.vstack((
        0.8*np.ones((3, 1)),
        xi_p[:,np.newaxis],
        0.8*np.ones((3, 1)),
        -xi_m[:,np.newaxis]
    ))
    # uncomment next line in order to ignore inequality constraints
    # d = 1e8*np.ones((16, 1))
    return d


# Function not used (ignore)
def perturbation(t:float, id:int) -> np.ndarray:
    # if id == 1:
    return np.zeros((3,1))
    # return np.array([[
    #     3*((0.5 + 0.5*np.tanh(t-10)) - (0.5 + 0.5*np.tanh(t-12))),
    #     0,
    #     0
    # ]]).T

class CoppeliaPanel:
    def __init__(self, name:str, p0:list[float], points:list[str], P) -> None:
        self.id = sim.getObject(f'/{name}')
        self.object_points = [sim.getObject(p) for p in points]
        sim.setObjectPosition(self.id, p0)
        self.p = np.array(p0)
        self.quaternion = None
        self.R = None
        
        self.updatePose(P)

    def updatePose(self, P:list[np.ndarray]) -> None:
        p:np.ndarray = (P[0]+P[1]+P[2]+P[3])/4
        dirx1 = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
        dirx2 = (P[2] - P[3]) / np.linalg.norm(P[2] - P[3])
        diry1 = (P[0] - P[3]) / np.linalg.norm(P[3] - P[0])
        diry2 = (P[1] - P[2]) / np.linalg.norm(P[2] - P[1])
        dirx = 0.5 * (dirx1 + dirx2).squeeze()
        diry = 0.5 * (diry1 + diry2).squeeze()
        dirz = np.cross(dirx, diry)
        dirz = dirz / np.linalg.norm(dirz)

        otherz = np.cross(-diry, dirx)
        otherz = otherz / np.linalg.norm(otherz)

        R = np.hstack(( dirx[:,np.newaxis], diry[:,np.newaxis], dirz[:,np.newaxis] ))
        self.R = np.hstack(( -diry[:,np.newaxis], dirx[:,np.newaxis], otherz[:,np.newaxis] ))
        self.p = p.copy()
        self.quaternion = rotation_matrix_to_quaternion(R)
        sim.setObjectPosition(self.id, p.tolist())
        sim.setObjectQuaternion(self.id, self.quaternion.tolist())
    
    def updatePoseBck(self) -> None:
        P = []
        for ee in self.object_points:
            pos = sim.getObjectPosition(ee)
            P.append(np.array(pos))
        p:np.ndarray = (P[0]+P[1]+P[2]+P[3])/4
        p[2] -= 0.02
        Normal = np.cross(P[3] - P[0], P[1] - P[0])
        Normal = Normal / np.linalg.norm(Normal)
        z = np.array([0,0,1])
        z_o = (P[3]-P[0])/np.linalg.norm(P[3]-P[0])
        axis = np.cross(z, Normal)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z, Normal))
        v = np.sin(angle/2)*axis
        self.p = p.copy()
        self.quaternion = np.array([v[0] ,v[1], v[2], np.cos(angle/2)])
        sim.setObjectPosition(self.id, p.tolist())
        sim.setObjectQuaternion(self.id, [v[0] ,v[1], v[2], np.cos(angle/2)])