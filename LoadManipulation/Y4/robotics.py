import numpy as np
from numpy import sin, cos, sinh
from collections import deque

def sgnbi(x:float) -> float:
    # return x
    r1, r2 = 2, 0.5
    abs_x = abs(x)
    sgn = 1 if x > 0 else -1
    if (abs_x > 1):
        return sgn*(abs_x**r1)
    return (r1/r2)*sgn*(abs_x**r2)

def linear(x:float) -> float:
    return x

def power_sigmoid(x:float) -> float:
    zeta = 3
    if abs(x) >= 1:
        return x**3
    return (1 + np.exp(-zeta) - np.exp(-zeta*x))/(1 - np.exp(-zeta) + np.exp(-zeta*x))

def power_activation(x:float) -> float:
    zeta = 3
    return x**zeta

def hyperbolic_sine(x:float) -> float:
    zeta = 2
    return 0.5*(np.exp(zeta*x) - np.exp(-zeta*x))


class TVQPEIZNN:
    def __init__(self, 
                 y0:np.ndarray, 
                 G0:np.ndarray, 
                 u0:np.ndarray, 
                 shape:tuple[int],
                 gamma:float = 1.0,
                 tau:float = 0.05,
                 AF = linear) -> None:
        self.n, self.m, self.q = shape
        self.gamma = gamma
        self.tau = tau
        self.G = deque(maxlen=2)
        self.u = deque(maxlen=2)
        self.y = deque(maxlen=2)

        self.deltaplus = 1e-8*np.ones((self.q, 1))
        
        # G0[-self.m-self.q:, -self.m-self.q:] = np.zeros((self.m + self.q))
        A0 = np.hstack(( np.eye(self.m), np.zeros((self.m, self.n-self.m)) ))
        C0 = np.vstack(( np.eye(self.n), -np.eye(self.n) ))
        G0 = np.vstack((
            np.hstack(( np.eye(self.n), A0.T, C0.T )),
            np.hstack(( A0, np.zeros((self.m,self.m)), np.zeros((self.m,self.q)) )),
            np.hstack((-C0, np.zeros((self.q,self.m)), np.eye(self.q) ))
        ))
        self.u.append(u0.copy())
        self.G.append(G0.copy())
        self.y.append(y0.copy())

        self.PSI = np.vectorize(AF)


    def update(self, 
               Sk:np.ndarray, 
               Ak:np.ndarray, 
               Ck:np.ndarray, 
               pk:np.ndarray, 
               bk:np.ndarray, 
               dk:np.ndarray) -> np.ndarray:
        yk = self.y[-1]
        xk = yk[:self.n]
        lambdak = yk[self.n+self.m:]
        Gk = np.vstack((
            np.hstack(( Sk, Ak.T, Ck.T )),
            np.hstack(( Ak, np.zeros((self.m,self.m)), np.zeros((self.m,self.q)) )),
            np.hstack((-Ck, np.zeros((self.q,self.m)), np.eye(self.q) ))
        ))
        hk = dk - Ck @ xk
        uk = np.vstack((
            pk,
            -bk,
            dk - np.sqrt(hk*hk + lambdak*lambdak + self.deltaplus)
        ))

        delta_uk = uk - self.u[-1]
        delta_Gk = Gk - self.G[-1]

        mu = np.linalg.det(Gk @ Gk.T)
        lmax, epsilon = 0.2, 0.05
        if mu > epsilon:
            lb = 0
        else:
            lb = (1-(mu/epsilon)**2)*(lmax**2)
        iGk = Gk.T @ np.linalg.inv(Gk @ Gk.T + (lb**2) * np.eye(self.m+self.n+self.q))
        yk_1:np.ndarray = yk - iGk @ (delta_Gk @ yk + self.gamma*self.tau*self.PSI(Gk @ yk + uk) + delta_uk)
        # if(len(self.y) != self.y.maxlen):
        #      yk_1[:self.n] = np.linalg.pinv(Ak) @ bk
        #      yk_1[self.n:] = np.zeros((self.m+self.q, 1))
             

        # print("="*10 + "ITERATION" + "="*10)
        # print(f"{yk.T=}")
        # print(f"{np.linalg.matrix_rank(Gk)=}, G[{Gk.shape}]")
        # print(f"{Ak=}")
        # print(f"{np.linalg.pinv(Ak)=}")
        
        # print(f"{delta_uk=}")
        # print(f"{xk=}")
        # print(f"{bk=}")
        # print(f"{pk=}")
        # print(f"{(Gk @ yk + uk)=}")
        # print(f"{(Ak @ xk - bk)=}")
        # print('-'*20)
        # print(f"{dk=}")
        # print(f"{hk=}")
        # print('-'*20)
        # print(f"{yk_1=}")
        # print(f"{np.linalg.inv(Gk)=}", end='\n\n\n\n')
        
        self.y.append(yk_1.copy())
        self.u.append(uk.copy())
        self.G.append(Gk.copy())
        
        return (yk_1[:self.n].copy(), yk_1[self.n:self.n+self.m].copy(), yk_1[self.n+self.m:].copy())


class Youbot:
    # This are for CoppeliaSim, as precise as it can gets
    dx0 = 0.16618
    dz0 = 0.250
    a1 = 0.03301
    a2 = 0.155
    a3 = 0.13485
    d5 = 0.205

    # a1, a2, a3 = 0.033, 0.155, 0.135
    # d5 = 0.210
    # dx0, dz0 = 0.167, 0.245
    th_lower = np.deg2rad(np.array([-169,-90,-131,-102,-160]))
    th_upper = np.deg2rad(np.array([169,65,131,102,160]))
    #                               -2.94, -1.57, -2.28, -1.78, -2.79
    #                                2.94,  1.13,  2.28,  1.78,  2.79
    dth_upp = np.deg2rad(90*np.ones((5,)))
    dth_low = -(np.pi/2)*np.ones((5,))

    @staticmethod
    def fkine(x_c: float, y_c: float, theta_c: float, \
              theta1: float, theta2: float, theta3: float,\
                  theta4: float, theta5: float) -> np.ndarray:
        s1 = sin(theta2+theta3+theta4)
        s2 = cos(theta2+theta3+theta4)
        s3 = Youbot.a1 - Youbot.a2*sin(theta2) - Youbot.a3*sin(theta2+theta3) - Youbot.d5*s1
        s4 = - Youbot.a2*cos(theta2) - Youbot.a3*cos(theta2+theta3) - Youbot.d5*s2
        s5 = cos(theta1)*cos(theta5) - sin(theta5)*sin(theta1)*s2
        s6 = cos(theta1)*sin(theta5) + cos(theta5)*sin(theta1)*s2
        s7 = -sin(theta1)*cos(theta5) - sin(theta5)*cos(theta1)*s2
        s8 = sin(theta1)*sin(theta5) - cos(theta5)*cos(theta1)*s2
        s9 = sin(theta1 + theta_c)
        s10 = cos(theta1 + theta_c)

        R = [[s8*cos(theta_c) + s6*sin(theta_c), s5*sin(theta_c) - s7*cos(theta_c), -s10*s1],\
             [s8*sin(theta_c) - s6*cos(theta_c), -s5*cos(theta_c) - s7*sin(theta_c), -s9*s1],\
             [-s1*cos(theta5), s1*sin(theta5), s2]]
        t = [[x_c + Youbot.dx0*cos(theta_c) + s10*s3],\
             [y_c + Youbot.dx0*sin(theta_c) + s9*s3],\
             [Youbot.dz0 - s4]]
        return np.vstack((np.hstack((np.array(R), np.array(t))),np.array([[0,0,0,1]])))
    
    @staticmethod
    def jacob0(x_c: float, y_c: float, theta_c: float, \
              theta1: float, theta2: float, theta3: float,\
                  theta4: float, theta5: float) -> np.ndarray:
        s1 = sin(theta2+theta3+theta4)
        s2 = cos(theta2+theta3+theta4)
        s3 = Youbot.a1 - Youbot.a2*sin(theta2) - Youbot.a3*sin(theta2+theta3) - Youbot.d5*s1
        s4 = - Youbot.a2*cos(theta2) - Youbot.a3*cos(theta2+theta3) - Youbot.d5*s2
        # s5 = cos(theta1)*cos(theta5) - sin(theta5)*sin(theta1)*s2
        # s6 = cos(theta1)*sin(theta5) + cos(theta5)*sin(theta1)*s2
        # s7 = -sin(theta1)*cos(theta5) - sin(theta5)*cos(theta1)*s2
        # s8 = sin(theta1)*sin(theta5) - cos(theta5)*cos(theta1)*s2
        s9 = sin(theta1+theta_c)
        s10 = cos(theta1+theta_c)
        J = [[1, 0, -Youbot.dx0*sin(theta_c)-s3*s9, -s3*s9, s10*s4, s10*(Youbot.a2*cos(theta2)+s4), -Youbot.d5*s10*s2, 0],\
             [0, 1, Youbot.dx0*cos(theta_c)+s3*s10, s3*s10, s9*s4, s9*(Youbot.a2*cos(theta2)+s4), -Youbot.d5*s9*s2, 0],\
             [0, 0, 0, 0, s3-Youbot.a1, Youbot.a2*sin(theta2)+s3-Youbot.a1, -Youbot.d5*s1, 0],\
             [0, 0, 0, 0, s9, s9, s9, -s10*s1],\
             [0, 0, 0, 0, -s10, -s10, -s10, -s9*s1],\
             [0, 0, 1, 1, 0, 0, 0, s2]]
        return np.array(J)
    
    @staticmethod
    def jacobnn(x_c: float, y_c: float, theta_c: float, \
              theta1: float, theta2: float, theta3: float,\
                  theta4: float, theta5: float) -> np.ndarray:
        s1 = sin(theta2+theta3+theta4)
        s2 = cos(theta2+theta3+theta4)
        s3 = Youbot.a1 - Youbot.a2*sin(theta2) - Youbot.a3*sin(theta2+theta3) - Youbot.d5*s1
        s4 = - Youbot.a2*cos(theta2) - Youbot.a3*cos(theta2+theta3) - Youbot.d5*s2
        s9 = sin(theta1+theta_c)
        s10 = cos(theta1+theta_c)
        J = [[1, 0, -Youbot.dx0*sin(theta_c)-s3*s9, -s3*s9, s10*s4, s10*(Youbot.a2*cos(theta2)+s4), -Youbot.d5*s10*s2, 0],\
             [0, 1, Youbot.dx0*cos(theta_c)+s3*s10, s3*s10, s9*s4, s9*(Youbot.a2*cos(theta2)+s4), -Youbot.d5*s9*s2, 0],\
             [0, 0, 0, 0, s3-Youbot.a1, Youbot.a2*sin(theta2)+s3-Youbot.a1, -Youbot.d5*s1, 0],\
             [0, 0, 0, 0, s9, s9, s9, -s10*s1],\
             [0, 0, 0, 0, -s10, -s10, -s10, -s9*s1],\
             [0, 0, 0, 1, 0, 0, 0, s2]]
        return np.array(J)

    @staticmethod
    def jacobW(x_c: float, y_c: float, theta_c: float, \
                theta1: float, theta2: float, theta3: float,\
                    theta4: float, theta5: float) -> np.ndarray:
        s1 = sin(theta2+theta3+theta4)
        s2 = cos(theta2+theta3+theta4)
        s9 = sin(theta1+theta_c)
        s10 = cos(theta1+theta_c)
        J = [[0, 0, 0, 0, s9, s9, s9, -s10*s1],\
            [0, 0, 0, 0, -s10, -s10, -s10, -s9*s1],\
            [0, 0, 1, 1, 0, 0, 0, s2]]
        return np.array(J)
    
    @staticmethod
    def jacobCarW() -> np.ndarray:
        J = np.zeros((3,8))
        J[2,2] = 1
        return J

def csch(x:float) -> float:
        if x == 0:
            return 999999999
        return 1 / sinh(x)
def arccsch(x):
    return np.log(1/x + np.sqrt(1/x**2 + 1))

def S_operator(x : np.ndarray) -> np.ndarray:
	v = np.squeeze(x)
	S = [[0, -v[2], v[1]],\
		[v[2], 0, -v[0]],\
		[-v[1], v[0], 0]]
	return np.array(S)

def Rotx(theta: float) -> np.ndarray:
    """Creates a rotation matrix for a rotation around the x-axis by angle theta (in radians)."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def Roty(theta: float) -> np.ndarray:
    """Creates a rotation matrix for a rotation around the y-axis by angle theta (in radians)."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def Rotz(theta: float) -> np.ndarray:
    """Creates a rotation matrix for a rotation around the z-axis by angle theta (in radians)."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters:
    R (numpy.ndarray): A 3x3 rotation matrix.
    
    Returns:
    numpy.ndarray: A quaternion (q_w, q_x, q_y, q_z) as a 1D array.
    """
    q_w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    q_x = (R[2, 1] - R[1, 2]) / (4.0 * q_w)
    q_y = (R[0, 2] - R[2, 0]) / (4.0 * q_w)
    q_z = (R[1, 0] - R[0, 1]) / (4.0 * q_w)
    
    return np.array([q_x, q_y, q_z, q_w])