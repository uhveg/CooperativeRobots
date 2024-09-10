import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle, sqlite3, sys
import os
from robotics import Youbot
from logger import DEFAULT_LOG_COLUMNS

plt.style.use('https://raw.githubusercontent.com/uhveg/matplotlibStyles/main/thesis.mplstyle')
prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def loadData(cursor: sqlite3.Cursor, tablename: str) -> list[np.ndarray]:
    cols = ', '.join(DEFAULT_LOG_COLUMNS)
    cursor.execute(f'''SELECT time, {cols} FROM {tablename}''')
    rows = cursor.fetchall()
    nrows, ncols = len(rows), len(rows[0])
    t = np.array([rows[i][0] for i in range(nrows)])
    arrays = [t] + [np.array([pickle.loads(rows[i][j]) for i in range(len(rows))]).squeeze() for j in range(1,ncols) ]
    return arrays

def plotCositeSimilarity(dirfolder: str,
                         t: np.ndarray, 
                         data: list[np.ndarray], 
                         labels: list[str], 
                         limits: tuple[float, float] = None) -> None:
    assert len(data) == len(labels)
    fig, ax = plt.subplots()
    ax: plt.Axes
    for i in range(len(data)):
        # if data[i][-1] < 0:
        #     ax.plot(t, -data[i], label=f"-{labels[i]}")
        # else:
        ax.plot(t, data[i], label=labels[i])
    ax.set_xlabel(r"time (s)")
    ax.set_xlim((t[0], t[-1]))
    bottom, top = ax.get_ylim()
    # ax.set_ylim((bottom, 1.05))
    # ax.set_title(r"Cosine Similarity", y=0.87)
    if limits is not None:
        tf1, tf2 = limits
        ax.set_xlim((t[0], tf1))
        inset_ax:plt.Axes = ax.inset_axes([0.65,0.4,0.3,0.3])
        for i in range(len(data)):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], abs(data[i][(t >= tf2) & (t <= t[-1])]))
        inset_ax.set_xlim((tf2, t[-1]))
        inset_ax.margins(y=0.1)
        inset_ax.set_yticks(np.linspace(0.9999925,1.0000025,5))
    ax.legend(loc='lower right', ncols=2, fontsize=9)
    # ax.set_aspect('equal')
    plt.savefig(f'{dirfolder}/cosinesimee.pdf')
    plt.show()

def plotLoadPosition(dirfolder: str,
                    t:np.ndarray,
                    pos: np.ndarray,
                    desired: np.ndarray,
                    xlim: tuple[float, float]= None) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(t, pos[:,0], color=prop_cycle[4], label=r"$x_B(t)$")
    ax.plot(t, pos[:,1], color=prop_cycle[1], label=r"$y_B(t)$")
    # ax.plot(t, pos[:,2], color=prop_cycle[5], label=r"$z_B(t)$")
    ax.plot(t, desired[:,0], color=prop_cycle[2], linestyle='--', linewidth=1, label=r"$x_B^*(t)$")
    ax.plot(t, desired[:,1], color=prop_cycle[0], linestyle='--', linewidth=1, label=r"$y_B^*(t)$")
    # ax.plot(t, desired[:,2], color=prop_cycle[3], linestyle='--', linewidth=1, label=r"$z_B^*(t)$")

    ax.set_xlabel(r"time(s)")
    if xlim is None:
        ax.set_xlim((t[0], t[-1]))
    else:
        ax.set_xlim(xlim)
    ax.legend(loc='lower left', ncols=2)
    plt.savefig(f'{dirfolder}/loadpos.pdf')
    plt.show()

def plotLoadPositionTOP(dirfolder: str,
                    pos: np.ndarray,
                    EE: list[np.ndarray],
                    desired: np.ndarray,
                    limits: tuple[float, float] = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()
    alpha = 0.5
    for i in range(len(EE)):
        ax.plot(EE[i][:,0], EE[i][:,1], color=prop_cycle[i], label=rf"$\mathbf{{p}}_{i}(t)$", alpha=alpha)
        ax.scatter(EE[i][0,0], EE[i][0,1], facecolor="white", edgecolor=prop_cycle[i], marker="o", zorder=3, s=30, alpha=alpha)
        ax.scatter(EE[i][-1,0], EE[i][-1,1], color=prop_cycle[i], marker="o", zorder=3, s=30, alpha=alpha)
    
    ax.plot(pos[:,0], pos[:,1], color=prop_cycle[7], label=r"$\mathbf{p}_B(t)$")
    ax.scatter(pos[0,0], pos[0,1], facecolor="white", edgecolor=prop_cycle[7], marker="o", zorder=3, s=30)
    ax.scatter(pos[-1,0], pos[-1,1], color=prop_cycle[7], marker="o", zorder=3, s=30)

    ax.plot(desired[:,0], desired[:,1], color='#000000', linestyle='--', linewidth=1, label=r"$\mathbf{p}_B^*(t)$")
    for k in [0,-1]:
        ax.plot([EE[0][k,0], EE[1][k,0]], [EE[0][k,1], EE[1][k,1]], color='#000000', linestyle='-.')
        ax.plot([EE[0][k,0], EE[3][k,0]], [EE[0][k,1], EE[3][k,1]], color='#000000', linestyle='-.')
        ax.plot([EE[1][k,0], EE[2][k,0]], [EE[1][k,1], EE[2][k,1]], color='#000000', linestyle='-.')
        ax.plot([EE[2][k,0], EE[3][k,0]], [EE[2][k,1], EE[3][k,1]], color='#000000', linestyle='-.')
    ax.grid(False)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal')
    # ax.set_xticks(np.linspace(0, 10, 5))  # 5 ticks on x-axis
    # ax.set_yticks(np.linspace(-1, 1, 5))  # 5 ticks on y-axis
    # ax.set_zticks(np.linspace(-1, 1, 5))  # 5 ticks on z-axis
    ax.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5,1))
    plt.savefig(f'{dirfolder}/loadposTOP.pdf')
    plt.show()

def plotLoadPosition3d(dirfolder: str,
                    pos: np.ndarray,
                    EE: list[np.ndarray],
                    desired: np.ndarray,
                    limits: tuple[float, float] = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax: Axes3D

    alpha = 0.5
    for i in range(len(EE)):
        ax.plot(EE[i][:,0], EE[i][:,1], EE[i][:,2], color=prop_cycle[i], label=rf"$\mathbf{{p}}_{i}(t)$", alpha=alpha)
        ax.scatter(EE[i][0,0], EE[i][0,1], EE[i][0,2], facecolor="white", edgecolor=prop_cycle[i], marker="o", zorder=3, s=30, alpha=alpha)
        ax.scatter(EE[i][-1,0], EE[i][-1,1], EE[i][-1,2], color=prop_cycle[i], marker="o", zorder=3, s=30, alpha=alpha)

    ax.plot(pos[:,0], pos[:,1], pos[:,2], color=prop_cycle[7], label=r"$\mathbf{p}_B(t)$")
    ax.scatter(pos[0,0], pos[0,1], pos[0,2], facecolor="white", edgecolor=prop_cycle[7], marker="o", zorder=3, s=30)
    ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], color=prop_cycle[7], marker="o", zorder=3, s=30)

    # ax.plot(desired[:,0], desired[:,1], desired[:,2], color='#000000', linestyle='--', label=r"$\mathbf{p}_B^*(t)$")
    for al, k in zip([0.3,1], [0, -1]):
        ax.plot([EE[0][k,0], EE[1][k,0]], [EE[0][k,1], EE[1][k,1]], [EE[0][k,2], EE[1][k,2]], color='#000000', linestyle='-.', alpha=al)
        ax.plot([EE[0][k,0], EE[3][k,0]], [EE[0][k,1], EE[3][k,1]], [EE[0][k,2], EE[3][k,2]], color='#000000', linestyle='-.', alpha=al)
        ax.plot([EE[1][k,0], EE[2][k,0]], [EE[1][k,1], EE[2][k,1]], [EE[1][k,2], EE[2][k,2]], color='#000000', linestyle='-.', alpha=al)
        ax.plot([EE[2][k,0], EE[3][k,0]], [EE[2][k,1], EE[3][k,1]], [EE[2][k,2], EE[3][k,2]], color='#000000', linestyle='-.', alpha=al)
    ax.grid(False)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    # ax.set_aspect('equal')
    ax.view_init(elev=30, azim=-55)
    # ax.set_xticks(np.linspace(0, 10, 5))  # 5 ticks on x-axis
    ax.set_yticks(np.linspace(-1, 1, 5))  # 5 ticks on y-axis
    # ax.set_zticks(np.linspace(-1, 1, 5))  # 5 ticks on z-axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,0.9), ncols=3)
    plt.savefig(f'{dirfolder}/loadpos3d.pdf')
    plt.show()

def plotErrorsLoad(dirfolder: str,
                    t: np.ndarray, 
                    pos: np.ndarray,
                    dpos: np.ndarray,
                    limits: tuple[float, float] = None) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    errpos = dpos - pos
    ax.plot(t, errpos[:,0], label=r"$x_B^* - x_B$")
    ax.plot(t, errpos[:,1], label=r"$y_B^* - y_B$")
    ax.plot(t, errpos[:,2], label=r"$z_B^* - z_B$")
    ax.set_xlabel(r"time (s)")
    ax.set_xlim((t[0], t[-1]))

    if limits is not None:
        tf1, tf2 = limits
        ax.set_xlim((t[0], tf1))
        inset_ax:plt.Axes = ax.inset_axes([0.65,0.075,0.3,0.3])
        for i in range(3):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], errpos[:,i][(t >= tf2) & (t <= t[-1])])
        inset_ax.set_xlim((tf2, t[-1]))
        inset_ax.margins(y=0.1)

    ax.legend()
    plt.savefig(f'{dirfolder}/loaderrors.pdf')
    plt.show()

def plotErrorsLoadCosine(dirfolder: str,
                    t: np.ndarray, 
                    pos: np.ndarray,
                    dpos: np.ndarray,
                    cosim: list[np.ndarray],
                    limits: tuple[float, float] = None) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    errpos = dpos - pos
    ax.plot(t, errpos[:,0], label=r"$x_B^* - x_B$")
    ax.plot(t, errpos[:,1], label=r"$y_B^* - y_B$")
    ax.plot(t, errpos[:,2], label=r"$z_B^* - z_B$")
    ax.plot(t, cosim[0]-1, label=r'$S_C\left(\mathbf{r_{B_x}^*}, \mathbf{r_{B_x}}\right)-1$')
    ax.plot(t, cosim[1]-1, label=r'$S_C\left(\mathbf{r_{B_y}^*}, \mathbf{r_{B_y}}\right)-1$')
    ax.set_xlabel(r"time (s)")
    ax.set_xlim((t[0], t[-1]))

    if limits is not None:
        tf1, tf2 = limits
        ax.set_xlim((t[0], tf1))
        inset_ax:plt.Axes = ax.inset_axes([0.5,0.3,0.45,0.3])
        for i in range(3):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], errpos[:,i][(t >= tf2) & (t <= t[-1])])
        for i in range(2):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], (cosim[i]-1)[(t >= tf2) & (t <= t[-1])])
        inset_ax.set_xlim((tf2, t[-1]))
        inset_ax.margins(y=0.1)

    ax.legend(ncols=2)
    plt.savefig(f'{dirfolder}/loaderrorscosim.pdf')
    plt.show()

def plotQ(dir:str,
          t:np.ndarray, 
          jointdata: list[np.ndarray]
          ) -> None:
    namesj = [r'$x_B$', r'$y_B$', r'$\theta_c$'] + [rf"$\theta_{i+1}$" for i in range(5)]
    fig, ax = plt.subplots(8,1,sharex=True,figsize=(8,8))
    ax = ax.flatten()
    ax: list[plt.Axes]
    for k in range(len(jointdata)):
        for i in range(8):
            ax[i].plot(t, jointdata[k][:,i], color=prop_cycle[k])
            ax[i].set_xlim((t[0], t[-1]))
            ax[i].set_title(namesj[i], x=1.03, y=0, fontsize=10)
            ax[i].margins(y=0.3)
    for i in range(8):
        if i > 2:
            bottom, top = ax[i].get_ylim()
            # if abs(min(jointdata[:,i+3]) - Youbot.th_lower[i]) < 0.1:
            # ax[i].plot([t[0], t[-1]], [Youbot.th_lower[i-3], Youbot.th_lower[i-3]], color='#ff0000')
            ax[i].fill_between([t[0], t[-1]], [Youbot.th_lower[i-3], Youbot.th_lower[i-3]], -20, color='#ff000022', alpha=0.1)
            # if abs(max(jointdata[:,i+3]) - Youbot.th_upper[i]) < 0.1:
            # ax[i].plot([t[0], t[-1]], [Youbot.th_upper[i-3], Youbot.th_upper[i-3]], color='#ff0000')
            ax[i].fill_between([t[0], t[-1]], [Youbot.th_upper[i-3], Youbot.th_upper[i-3]], 20, color='#ff000022', alpha=0.1)
            ax[i].set_ylim((bottom, top))
    ax[-1].set_xlabel(r"time (s)")
    ax[0].legend([f"$\mathbf{{q}}_{i+1}$"for i in range(4)], loc='lower center', ncols=4, bbox_to_anchor=(0.5, 0.9))
    plt.savefig(f'{dir}/q.pdf')
    plt.show()

def plotDQ(dir:str,
          t:np.ndarray, 
          jointdata: list[np.ndarray]
          ) -> None:
    namesj = [r'$\dot x_B$', r'$\dot y_B$', r'$\dot \theta_c$'] + [rf"$\dot \theta_{i+1}$" for i in range(5)]
    fig, ax = plt.subplots(8,1,sharex=True,figsize=(8,8))
    ax = ax.flatten()
    ax: list[plt.Axes]
    for k in range(len(jointdata)):
        for i in range(8):
            ax[i].plot(t, jointdata[k][:,i], color=prop_cycle[k])
            ax[i].set_xlim((t[0], t[-1]))
            ax[i].set_title(namesj[i], x=1.03, y=0, fontsize=10)
            ax[i].margins(y=0.3)
            # if i > 2:
            #     bottom, top = ax[i].get_ylim()
            #     ax[i].fill_between([t[0], t[-1]], [Youbot.dth_low[i-3], Youbot.dth_low[i-3]], -20, color='#ff000022', alpha=0.1)
            #     ax[i].fill_between([t[0], t[-1]], [Youbot.dth_upp[i-3], Youbot.dth_upp[i-3]], 20, color='#ff000022', alpha=0.1)
            #     ax[i].set_ylim((bottom, top))
    ax[-1].set_xlabel(r"time (s)")
    ax[0].legend([f"$\mathbf{{\dot q}}_{i+1}$"for i in range(4)], loc='lower center', ncols=4, bbox_to_anchor=(0.5, 0.9))
    plt.savefig(f'{dir}/dq.pdf')
    plt.show()

def plotJoints(dir:str, name:str,
               t:np.ndarray, 
               jointdata: np.ndarray) -> None:
    fig, ax = plt.subplots(5,1,sharex=True)
    ax = ax.flatten()
    ax: list[plt.Axes]
    for i in range(5):
        ax[i].plot(t, jointdata[:,i+3], color=prop_cycle[7])
        ax[i].set_xlim((t[0], t[-1]))
        ax[i].set_title(rf"$\theta_{i+1}$", x=1.03, y=0.25, fontsize=10)
        ax[i].margins(y=0.3)
        bottom, top = ax[i].get_ylim()
        # if abs(min(jointdata[:,i+3]) - Youbot.th_lower[i]) < 0.1:
        ax[i].plot([t[0], t[-1]], [Youbot.th_lower[i], Youbot.th_lower[i]], color='#ff0000')
        ax[i].fill_between([t[0], t[-1]], [Youbot.th_lower[i], Youbot.th_lower[i]], -20, color='#ff000022', alpha=0.1)
        # if abs(max(jointdata[:,i+3]) - Youbot.th_upper[i]) < 0.1:
        ax[i].plot([t[0], t[-1]], [Youbot.th_upper[i], Youbot.th_upper[i]], color='#ff0000')
        ax[i].fill_between([t[0], t[-1]], [Youbot.th_upper[i], Youbot.th_upper[i]], 20, color='#ff000022', alpha=0.1)
        ax[i].set_ylim((bottom, top))
    ax[-1].set_xlabel(r"time (s)")
    plt.savefig(f'{dir}/{name}.pdf')
    plt.show()

def plotVelocities(dir:str, name:str,
                   t:np.ndarray,
                   dq:np.ndarray,
                   limits: tuple[float, float] = None) -> None:
    vels = [r'$\dot x_c$', r'$\dot y_c$', r'$\dot \theta_c$'] + [rf'$\dot \theta_{i}$' for i in range(1,6)]
    fig, ax = plt.subplots()
    ax: plt.Axes
    for i in range(dq.shape[1]):
        ax.plot(t, dq[:,i], label=vels[i])
    ax.set_xlim((t[0], t[-1]))
    ax.set_xlabel(r"time (s)")

    if limits is not None:
        tf1, tf2 = limits
        ax.set_xlim((t[0], tf1))
        inset_ax:plt.Axes = ax.inset_axes([0.65,0.075,0.3,0.3])
        for i in range(dq.shape[1]):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], dq[:,i][(t >= tf2) & (t <= t[-1])])
        inset_ax.set_xlim((tf2, t[-1]))
        inset_ax.margins(y=0.1)
    ax.legend(ncols=2, loc='lower center')
    plt.savefig(f'{dir}/{name}.pdf')
    plt.show()

def plotConstraints(dir:str,
                     name:str,
                     t:np.ndarray,
                     Eq:list[np.ndarray]) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    for i, eq in enumerate(Eq):
        ax.plot(t, eq[:,-1], color=prop_cycle[i])
    ax.set_xlim((t[0], t[-1]))
    ax.set_xlabel(r"time (s)")
    plt.savefig(f'{dir}/{name}.pdf')
    plt.show()

def plotNeurons(dir:str,
                name:str,
                t:np.ndarray,
                N:np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(t, N)
    ax.set_xlim((t[0], t[-1]))
    ax.set_xlabel(r"time (s)")
    plt.savefig(f'{dir}/{name}.pdf')
    plt.show()

def plotEdges(dir:str,
              t:np.ndarray,
              NL:list[np.ndarray]) -> None:
    edges = [[0,1],[1,2],[2,3],[0,3]]
    L = [0.95,1.6,0.95,1.6]
    fig, ax = plt.subplots()
    ax: plt.Axes
    for i in range(len(edges)):
        ax.plot(t, NL[i]-L[i], label=rf"$\lVert l_{{ {edges[i][0]}{edges[i][1]} }} \rVert - L^*$")
    ax.set_xlim((t[0], t[-1]))
    ax.set_ylim((-0.005,0.005))
    ax.set_xlabel(r"time (s)")
    ax.legend(loc='lower right')
    plt.savefig(f'{dir}/edges.pdf')
    plt.show()

def plotLoadZ(dirfolder: str,
              t:np.ndarray,
              pos: np.ndarray,
              desired: np.ndarray,
              xlim: tuple[float, float]= None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.plot(t, pos[:,2], color=prop_cycle[3], label=r"$z_B(t)$")
    ax.plot(t, desired[:,2], color=prop_cycle[0], linestyle='--', linewidth=1, label=r"$z_B^*(t)$")
    # ax.plot(t, desired[:,2], color=prop_cycle[0], linestyle='--', linewidth=1, label=r"$z_B^*(t)$")

    ax.set_xlabel(r"time(s)")
    if xlim is None:
        ax.set_xlim((t[0], t[-1]))
    else:
        ax.set_xlim(xlim)
    ax.legend(loc='lower right')
    plt.savefig(f'{dirfolder}/loadposZv.pdf')
    plt.show()

def plotLoadElevation(dirfolder: str,
                      t:np.ndarray,
                      loadR: np.ndarray,
                      desLoadR: np.ndarray,
                      xlim: tuple[float, float]= None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ex = np.array([np.arctan2(loadR[i,2,0], np.sqrt(loadR[i,0,0]**2 + loadR[i,1,0]**2)) for i in range(loadR.shape[0])])
    ey = np.array([np.arctan2(loadR[i,2,1], np.sqrt(loadR[i,0,1]**2 + loadR[i,1,1]**2)) for i in range(loadR.shape[0])])
    dex = np.array([np.arctan2(desLoadR[i,2,0], np.sqrt(desLoadR[i,0,0]**2 + desLoadR[i,1,0]**2)) for i in range(desLoadR.shape[0])])
    dey = np.array([np.arctan2(desLoadR[i,2,1], np.sqrt(desLoadR[i,0,1]**2 + desLoadR[i,1,1]**2)) for i in range(desLoadR.shape[0])])
    ax.plot(t, ex, color=prop_cycle[1], label=r"$\phi_x(t)$")
    ax.plot(t, ey, color=prop_cycle[4], label=r"$\phi_y(t)$")
    ax.plot(t, dex, color=prop_cycle[2], label=r"$\phi_x^*(t)$", linestyle='--')
    ax.plot(t, dey, color=prop_cycle[0], label=r"$\phi_y^*(t)$", linestyle='--')
    # ax.plot(t, ey, color='#000000', linestyle='--', linewidth=1, label=r"$z_B^*(t)$")


    ax.set_xlabel(r"time(s)")
    if xlim is None:
        ax.set_xlim((t[0], t[-1]))
    else:
        ax.set_xlim(xlim)
    ax.legend(loc='lower left', ncols=2)
    plt.savefig(f'{dirfolder}/elevation.pdf')
    plt.show()

def plotLoadElevationError(dirfolder: str,
                      t:np.ndarray,
                      loadR: np.ndarray,
                      desLoadR: np.ndarray,
                      xlim: tuple[float, float]= None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ex = np.array([np.arctan2(loadR[i,2,0], np.sqrt(loadR[i,0,0]**2 + loadR[i,1,0]**2)) for i in range(loadR.shape[0])])
    ey = np.array([np.arctan2(loadR[i,2,1], np.sqrt(loadR[i,0,1]**2 + loadR[i,1,1]**2)) for i in range(loadR.shape[0])])
    dex = np.array([np.arctan2(desLoadR[i,2,0], np.sqrt(desLoadR[i,0,0]**2 + desLoadR[i,1,0]**2)) for i in range(desLoadR.shape[0])])
    dey = np.array([np.arctan2(desLoadR[i,2,1], np.sqrt(desLoadR[i,0,1]**2 + desLoadR[i,1,1]**2)) for i in range(desLoadR.shape[0])])
    ax.plot(t, ex - dex, color=prop_cycle[1], label=r"$\phi_x^*(t) - \phi_x(t)$")
    ax.plot(t, ey - dey, color=prop_cycle[4], label=r"$\phi_y^*(t) - \phi_y(t)$")
    # ax.plot(t, dex, color=prop_cycle[2], label=r"$\phi_x^*(t)$", linestyle='--')
    # ax.plot(t, dey, color=prop_cycle[0], label=r"$\phi_y^*(t)$", linestyle='--')
    # ax.plot(t, ey, color='#000000', linestyle='--', linewidth=1, label=r"$z_B^*(t)$")


    ax.set_xlabel(r"time(s)")
    if xlim is None:
        ax.set_xlim((t[0], t[-1]))
    else:
        ax.set_xlim(xlim)
    ax.legend(loc='lower left', ncols=2)
    plt.savefig(f'{dirfolder}/elevationError.pdf')
    plt.show()

def plotLoadOrientationError(dir:str,
                             t:np.ndarray,
                             loadR:np.ndarray,
                             desR:np.ndarray):
    similarityX = np.array([np.dot(loadR[i,:,0], desR[i,:,0]) for i in range(loadR.shape[0])])
    similarityY = np.array([np.dot(loadR[i,:,1], desR[i,:,1]) for i in range(loadR.shape[0])])
    similarityZ = np.array([np.dot(loadR[i,:,2], desR[i,:,2]) for i in range(loadR.shape[0])])

    mse = np.array([np.mean((loadR[i,:,:].squeeze() - desR[i,:,:].squeeze())**2) for i in range(loadR.shape[0])])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t, similarityX-1, label=r"$S_C\left(\mathbf{r_{B_x}^*}(t), \mathbf{r_{B_x}}(t)\right)-1$")
    ax.plot(t, similarityY-1, label=r"$S_C\left(\mathbf{r_{B_y}^*}(t), \mathbf{r_{B_y}}(t)\right)-1$")
    ax.plot(t, similarityZ-1, label=r"$S_C\left(\mathbf{r_{B_z}^*}(t), \mathbf{r_{B_z}}(t)\right)-1$")
    # ax.plot(t, mse, label=r"$MSE$")
    ax.set_xlabel(r"time(s)")
    ax.set_xlim((t[0], t[-1]))
    ax.legend(ncols=1, fontsize=9)
    plt.savefig(f'{dir}/loadOrientationError.pdf')
    plt.show()

def plotEE( dir:str,
            t:np.ndarray,
            ee:list[np.ndarray]):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i, e in enumerate(ee):
        ax.plot(t, e[:,2], label=rf"$\mathbf{{p}}_{i}(t)|_3$")
    ax.set_xlabel(r"time(s)")
    ax.set_xlim((t[0], t[-1]))
    ax.legend(ncols=1, fontsize=9)
    plt.savefig(f'{dir}/EEzpos.pdf')
    plt.show()

class Data:
    def __init__(self, data:list) -> None:
        self.time = data[0]
        self.Q  = [data[1], data[10], data[19], data[28]]
        self.dQ = [data[2], data[11], data[20], data[29]]
        self.EE = [data[3], data[12], data[21], data[30]]
        self.RHO = [data[4], data[13], data[2], data[31]]
        self.ZETA = [data[5], data[14], data[23], data[32]]
        self.EQ = [data[6], data[15], data[24], data[33]]
        self.IQ = [data[7], data[16], data[25], data[34]]
        self.SCEEx = [data[8], data[17], data[26], data[35]]
        self.SCEEz = [data[9], data[18], data[27], data[36]]
        self.loadP = data[37]
        self.loadR = data[38]
        self.SCLx  = data[39]
        self.SCLy  = data[40]
        self.desP  = data[41]
        self.desR  = data[42]
        self.NL    = [data[43], data[44], data[45], data[46]]
        self.KL    = [data[47], data[48], data[49], data[50]]

if __name__ == "__main__":
    assert len(sys.argv) == 4
    args = sys.argv[1:]
    dir = f"images/{args[1]}/{args[2]}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    conn = sqlite3.connect(f'logging/{args[0]}')
    cursor = conn.cursor()
    
    dat = Data(loadData(cursor, args[2]))

    labels_SCload = [r'$S_C\left(\mathbf{r_{B_x}^*}, \mathbf{r_{B_x}}\right)$', r'$S_C\left(\mathbf{r_{B_y}^*}, \mathbf{r_{B_y}}\right)$']
    labels_SCee = [rf'$S_C\left(\mathbf{{d_x^{{(E_{i})}}, r_x^{{(E_{i})}}}}\right)$' for i in range(4)] + \
                 [rf'$S_C\left(\mathbf{{d_z^{{(E_{i})}}, r_z^{{(E_{i})}}}}\right)$' for i in range(4)]
    
    # plotLoadPosition(dir, dat.time, dat.loadP, dat.desP)
    # plotErrorsLoad(dir, log[0], pb, log[14], log[23], log[24])
    # plotLoadPosition3d(dir, dat.loadP, dat.EE, dat.desP)
    # plotLoadPositionTOP(dir, dat.loadP, dat.EE, dat.desP)
    # plotJoints(dir, "joints1", dat.time, dat.Q[0])
    # plotJoints(dir, "joints2", log[0], log[2])
    # plotVelocities(dir, "dq1", log[0], log[3])
    # plotVelocities(dir, "dq2", log[0], log[4])
    # plotConstraints(dir, "EQ1", log[0], log[15]) #15,18,19,22
    # plotNeurons(dir, "Rho1", log[0], log[16])
    # plotNeurons(dir, "Zeta1", log[0], log[17])
    # plotConstraints(dir, "Iq1", log[0], log[18])
    # plotConstraints(dir, "EQ2", log[0], log[19])
    # plotNeurons(dir, "Rho2", log[0], log[20])
    # plotNeurons(dir, "Zeta2", log[0], log[21])
    # plotConstraints(dir, "Iq2", log[0], log[22])

    # plotErrorsLoad(dir, dat.time, dat.loadP, dat.desP)
    # plotLoadZ(dir, dat.time, dat.loadP, dat.desP)
    # plotEdges(dir, dat.time, dat.NL)
    # plotCositeSimilarity(dir, dat.time, dat.SCEEx + dat.SCEEz, labels_SCee)
    # plotCositeSimilarity(dir, dat.time, [dat.SCLx, dat.SCLy], labels_SCload)
    # plotErrorsLoadCosine(dir, dat.time, dat.loadP, dat.desP, [dat.SCLx, dat.SCLy], limits=(110,55))
    # plotLoadPosition3d(dir, dat.loadP, dat.EE, dat.desP)
    # plotLoadPositionTOP(dir, dat.loadP, dat.EE, dat.desP)
    # plotQ(dir, dat.time, dat.Q)
    # plotDQ(dir, dat.time, dat.dQ)
    plotLoadElevation(dir, dat.time, dat.loadR, dat.desR)
    # plotLoadElevationError(dir, dat.time, dat.loadR, dat.desR)
    # plotLoadOrientationError(dir, dat.time, dat.loadR, dat.desR)
    plotEE(dir, dat.time, dat.EE)
    conn.close()
