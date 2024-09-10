from typing import Callable
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import numpy as np
import pickle, sqlite3, sys
import os
from robotics import Youbot

plt.style.use('https://raw.githubusercontent.com/uhveg/matplotlibStyles/main/thesis.mplstyle')
prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def loadData(cursor: sqlite3.Cursor, tablename: str) -> list[np.ndarray]:
    cursor.execute(f'''SELECT time, Youbot1, Youbot2, Youbot1_d, Youbot2_d, ee1, ee2, nl12,
                   cosim_ddede1, cosim_dcrxc1, cosim_derxe1, cosim_ddede2, cosim_dcrxc2, cosim_derxe2, de1,
                   EqConstraints1, rho1, zeta1, Ineq1, EqConstraints2, rho2, zeta2, Ineq2 FROM {tablename}''')
    rows = cursor.fetchall()
    nrows, ncols = len(rows), len(rows[0])
    t = np.array([rows[i][0] for i in range(nrows)])
    arrays = [t] + [np.array([pickle.loads(rows[i][j]) for i in range(len(rows))]).squeeze() for j in range(1,ncols) ]
    try:
        cursor.execute(f''' SELECT dpb, dde FROM {tablename}''')
        rows = cursor.fetchall()
        arrays += [np.array([pickle.loads(rows[i][j]) for i in range(len(rows))]).squeeze() for j in range(2) ]
    except:
        pass
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
        if data[i][-1] < 0:
            ax.plot(t, -data[i], label=f"-{labels[i]}")
        else:
            ax.plot(t, data[i], label=labels[i])
    ax.set_xlabel(r"time (s)")
    ax.set_xlim((t[0], t[-1]))
    bottom, top = ax.get_ylim()
    ax.set_ylim((bottom, 1.05))
    ax.set_title(r"Cosine Similarity", y=0.87)
    if limits is not None:
        tf1, tf2 = limits
        ax.set_xlim((t[0], tf1))
        inset_ax:plt.Axes = ax.inset_axes([0.65,0.4,0.3,0.3])
        for i in range(len(data)):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], abs(data[i][(t >= tf2) & (t <= t[-1])]))
        inset_ax.set_xlim((tf2, t[-1]))
        inset_ax.margins(y=0.1)
        inset_ax.set_yticks(np.linspace(0.9999925,1.0000025,5))
    ax.legend(loc='lower right', ncols=2, bbox_to_anchor=(1, 0.25), fontsize=5)
    plt.savefig(f'{dirfolder}/cosinesim.pdf')
    plt.show()

def plotLoadPosition(dirfolder: str,
                    t: np.ndarray, 
                    pos: np.ndarray,
                    limits: tuple[float, float] = None) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(t, pos[:,0], label=r"$x_B$")
    ax.plot(t, pos[:,1], label=r"$y_B$")
    ax.plot(t, pos[:,2], label=r"$z_B$")
    ax.set_xlabel(r"time (s)")
    ax.set_xlim((t[0], t[-1]))
    ax.legend()
    plt.savefig(f'{dirfolder}/loadpos.pdf')
    plt.show()

def plotLoadPosition3d(dirfolder: str,
                    pos: np.ndarray,
                    ee1: np.ndarray,
                    ee2: np.ndarray,
                    desired: np.ndarray,
                    limits: tuple[float, float] = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax: Axes3D
    # ax.plot(ee1[:,0], ee1[:,1], ee1[:,2], color=prop_cycle[0])
    # ax.scatter(ee1[0,0], ee1[0,1], ee1[0,2], facecolor="white", edgecolor=prop_cycle[0], marker="o", zorder=3, s=30)
    # ax.scatter(ee1[-1,0], ee1[-1,1], ee1[-1,2], color=prop_cycle[0], marker="o", label=r"$\mathbf{p}_i$", zorder=3, s=30)

    # ax.plot(ee2[:,0], ee2[:,1], ee2[:,2], color=prop_cycle[8])
    # ax.scatter(ee2[0,0], ee2[0,1], ee2[0,2], facecolor="white", edgecolor=prop_cycle[8], marker="o", zorder=3, s=30)
    # ax.scatter(ee2[-1,0], ee2[-1,1], ee2[-1,2], color=prop_cycle[8], marker="o", label=r"$\mathbf{p}_j$", zorder=3, s=30)

    ax.plot(desired[:,0], desired[:,1], desired[:,2], color='#000000', linestyle='--', label=r"$\mathbf{p}_B^*(t)$")
    
    ax.plot(pos[:,0], pos[:,1], pos[:,2], color=prop_cycle[7])
    ax.scatter(pos[0,0], pos[0,1], pos[0,2], facecolor="white", edgecolor=prop_cycle[7], marker="o", zorder=3, s=30)
    ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], color=prop_cycle[7], marker="o", label=r"$\mathbf{p}_B(t)$", zorder=3, s=30)

    # ax.plot([ee1[-1,0], ee2[-1,0]], [ee1[-1,1], ee2[-1,1]], [ee1[-1,2], ee2[-1,2]], color='#00000022')
    ax.grid(False)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.view_init(elev=15, azim=-30)
    # ax.set_xticks(np.linspace(0, 10, 5))  # 5 ticks on x-axis
    ax.set_yticks(np.linspace(-1, 1, 5))  # 5 ticks on y-axis
    # ax.set_zticks(np.linspace(-1, 1, 5))  # 5 ticks on z-axis
    ax.legend(loc='center left')
    plt.savefig(f'{dirfolder}/loadpos3d.pdf')
    plt.show()

def plotErrorsLoad(dirfolder: str,
                    t: np.ndarray, 
                    pos: np.ndarray,
                    dir: np.ndarray,
                    dpos: np.ndarray,
                    ddir: np.ndarray,
                    limits: tuple[float, float] = None) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    errpos = dpos - pos
    errDir = ddir - dir
    ax.plot(t, errpos[:,0], label=r"$x_B^* - x_B$")
    ax.plot(t, errpos[:,1], label=r"$y_B^* - y_B$")
    ax.plot(t, errpos[:,2], label=r"$z_B^* - z_B$")
    ax.plot(t, errDir[:,0], label=r"$r_{x_1}^* - r_{x_1}$")
    ax.plot(t, errDir[:,1], label=r"$r_{x_2}^* - r_{x_2}$")
    ax.plot(t, errDir[:,2], label=r"$r_{x_3}^* - r_{x_3}$")
    ax.set_xlabel(r"time (s)")
    ax.set_xlim((t[0], t[-1]))

    if limits is not None:
        tf1, tf2 = limits
        ax.set_xlim((t[0], tf1))
        inset_ax:plt.Axes = ax.inset_axes([0.65,0.075,0.3,0.3])
        for i in range(3):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], errpos[:,i][(t >= tf2) & (t <= t[-1])])
        for i in range(3):
            inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], errDir[:,i][(t >= tf2) & (t <= t[-1])])
        inset_ax.set_xlim((tf2, t[-1]))
        inset_ax.margins(y=0.1)

    ax.legend(ncols=2)
    plt.savefig(f'{dirfolder}/loaderrors.pdf')
    plt.show()

def plotErrorsLoadCosine(dirfolder: str,
                    t: np.ndarray, 
                    pos: np.ndarray,
                    dpos: np.ndarray,
                    cosim: np.ndarray,
                    limits: tuple[float, float] = None) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    errpos = dpos - pos
    ax.plot(t, errpos[:,0], label=r"$x_B^* - x_B$")
    ax.plot(t, errpos[:,1], label=r"$y_B^* - y_B$")
    ax.plot(t, errpos[:,2], label=r"$z_B^* - z_B$")
    ax.plot(t, cosim-1, label=r'$S_C\left(\mathbf{d_{E_1}}^*, \mathbf{d_{E_1}}\right)$')
    ax.set_xlabel(r"time (s)")
    ax.set_xlim((t[0], t[-1]))

    # if limits is not None:
    #     tf1, tf2 = limits
    #     ax.set_xlim((t[0], tf1))
    #     inset_ax:plt.Axes = ax.inset_axes([0.65,0.075,0.3,0.3])
    #     for i in range(3):
    #         inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], errpos[:,i][(t >= tf2) & (t <= t[-1])])
    #     for i in range(3):
    #         inset_ax.plot(t[(t >= tf2) & (t <= t[-1])], errDir[:,i][(t >= tf2) & (t <= t[-1])])
    #     inset_ax.set_xlim((tf2, t[-1]))
    #     inset_ax.margins(y=0.1)

    ax.legend(ncols=2)
    plt.savefig(f'{dirfolder}/loaderrorscosim.pdf')
    plt.show()

def plotQ(dir:str,
          t:np.ndarray, 
          jointdata: np.ndarray,
          jointdata2: np.ndarray
          ) -> None:
    namesj = [r'$x_B$', r'$y_B$', r'$\theta_c$'] + [rf"$\theta_{i+1}$" for i in range(5)]
    fig, ax = plt.subplots(8,1,sharex=True,figsize=(4.5,5))
    ax = ax.flatten()
    ax: list[plt.Axes]
    for i in range(jointdata.shape[1]):
        ax[i].plot(t, jointdata[:,i], color=prop_cycle[7])
        ax[i].plot(t, jointdata2[:,i], color=prop_cycle[5])
        ax[i].set_xlim((t[0], t[-1]))
        ax[i].set_title(namesj[i], x=1.03, y=0, fontsize=10)
        ax[i].margins(y=0.3)
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
    ax[0].legend(["$\mathbf{q}_i$", "$\mathbf{q}_j$"], loc='lower center', ncols=2, bbox_to_anchor=(0.5, 0.8))
    plt.savefig(f'{dir}/q.pdf')
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
                     Eq:np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(t, Eq)
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

if __name__ == "__main__":
    assert len(sys.argv) == 4
    args = sys.argv[1:]
    dir = f"images/{args[1]}/{args[2]}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    conn = sqlite3.connect(f'logging/{args[0]}')
    cursor = conn.cursor()
    
    log = loadData(cursor, args[2])
    pb = 0.5*(log[5] + log[6])
    # plotCositeSimilarity(dir, log[0], log[8:14], [r'$S_C\left(\mathbf{d_{E_1}}^*, \mathbf{d_{E_1}}\right)$',
    #                                         r'$S_C\left(\mathbf{d_{C_1}}, \mathbf{r_x^{(C_1)}}\right)$',
    #                                         r'$S_C\left(\mathbf{d_{E_1}}, \mathbf{r_x^{(E_1)}}\right)$',
    #                                         r'$S_C\left(\mathbf{d_{E_2}}^*, \mathbf{d_{E_2}}\right)$',
    #                                         r'$S_C\left(\mathbf{d_{C_2}}, \mathbf{r_x^{(C_2)}}\right)$',
    #                                         r'$S_C\left(\mathbf{d_{E_2}}, \mathbf{r_x^{(E_2)}}\right)$'])
    # plotLoadPosition(dir, log[0], pb)
    # plotErrorsLoad(dir, log[0], pb, log[14], log[23], log[24])
    plotLoadPosition3d(dir, pb, log[5], log[6], log[23])
    # plotJoints(dir, "joints1", log[0], log[1])
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

    # plotErrorsLoadCosine(dir, log[0], pb, log[23], log[8])

    # plotQ(dir, log[0], log[1], log[2])
    conn.close()
