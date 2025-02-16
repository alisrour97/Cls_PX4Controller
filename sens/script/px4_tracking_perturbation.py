import time
from typing import Sequence
import os
import datetime
from gen.models import jetson_pd as jetson_pd
from utils.trajectory import PiecewiseSplineTrajectory
import pickle
from base64 import b64decode
import numpy as np
import matplotlib.pyplot as plt
from cnst.constant import constants
import math
from mpl_toolkits.mplot3d import Axes3D
import utils.Functions as Fct
import seaborn as sns
import pandas as pd

c = constants()
# latex for figures
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \usepackage{lmodern}'
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern'
          }
plt.rcParams.update(params)



plt.rcParams.update({'figure.max_open_warning': 0})

now = datetime.datetime.now()
save_path = '../Paper_Journal_ICRA/save/controller_perturbation/controllers_Check_tracking' + now.strftime('%Y-%m-%d')
if not os.path.isdir(save_path):
    os.mkdir(save_path)


# Model initialization

t0 = time.time()
model = jetson_pd.Jetson(c.N_coeffs)
ODE = model.generate(3, verbose=True, mode=jetson_pd.Mode.NOGRAD, overwrite=True) # true if you changed the model and need to compile, otherwise false
ODE.set_integrator("dopri5")
model.set_default_state(c.init_waypoint)
t1 = time.time()
print(f"generating of PD model took {t1-t0} s")



filename = "../Paper_Journal_ICRA/save/trajectories/px4_opt___1_Tf_5_2023-08-17.straj"

INIT: Sequence[PiecewiseSplineTrajectory] = []
PI: Sequence[PiecewiseSplineTrajectory] = []
PI_k: Sequence[PiecewiseSplineTrajectory] = []
PI_ak: Sequence[PiecewiseSplineTrajectory] = []
TARGETS: Sequence[np.ndarray] = []

cases = [INIT, PI, TARGETS]
# cases = [INIT, TARGETS]

with open(filename, "rb") as dump:
    for i, line in enumerate(dump):
        cases[i % len(cases)].append(pickle.loads(b64decode(line)))


index = -3


G = ODE["Array_gains"]
for k in range(1):
    figure_1 = plt.figure(figsize=(16, 9))
    figure_0 = plt.figure(figsize=(16, 9))
    ax = figure_0.add_subplot(1, 1, 1)
    plt.axhline(y=1, linestyle='--', color='r', linewidth=4)
    ax.set_ylim(0, 1.35)



    # Choose a trajectory
    traj = INIT[k]
    ODE["Array_gains"] = G
    ODE.apply_parameters()
    print(traj.waypoints)

    model.integrate_along_trajectory(traj, c.N)  # inorder to get the last results
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
    u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
    times_vec = ODE.time_points
    u_nominal = (np.diff(u_intn, axis=0).T / np.diff(times_vec, axis=0).T).T
    # trajectory construction of the reference with all pos, vel and acceleration
    traj.Construct_trajectory()

    # times_vec = ODE.time_points
    # u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
    # u_nominal = (np.diff(u_intn, axis=0).T / np.diff(times_vec, axis=0).T).T
    #
    # for j in range(4):
    #     plt.plot(times_vec[1:], u_nominal[:, j]/c.umax)


    axis_1 = figure_1.add_subplot(1, 2, 1, projection="3d")
    axis_1.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k-', linewidth=3)
    axis_1.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2.5)
    for i in range(len(traj.waypoints)):
        if i != 2:
            axis_1.plot(*traj.waypoints[i][0, :3], "o", color="b", linewidth=3.5, markersize=8)
    axis_1.plot(*traj.waypoints[index][0, :3], "*", color="b", linewidth=3.5, markersize=14)

    axis_1.set_title(r'INIT', fontsize=c.fontsizetitle)
    axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.view_init(elev=30, azim=120)
    axis_1.set_xlim(0, 6)
    axis_1.set_ylim(0, 6)
    axis_1.set_zlim(0, 2.2)


    # Choose a trajectory
    traj = PI[k]

    # ODE["Array_gains"] = traj.get_controller_gains()
    ODE.apply_parameters()
    print(traj.get_controller_gains())
    print(traj.waypoints)

    model.integrate_along_trajectory(traj, c.N)  # inorder to get the last results
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]

    # times_vec = ODE.time_points
    # u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
    # u_nominal = (np.diff(u_intn, axis=0).T / np.diff(times_vec, axis=0).T).T
    #
    # for j in range(4):
    #     plt.plot(times_vec[1:], u_nominal[:, j]/c.umax)



    # trajectory construction of the reference with all pos, vel and acceleration
    traj.Construct_trajectory()
    axis_2 = figure_1.add_subplot(1, 2, 2, projection="3d")
    axis_2.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k-', linewidth=3)
    axis_2.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2.5)
    for i in range(len(traj.waypoints)):
        if i != c.N_waypoints + index:
            axis_2.plot(*traj.waypoints[i][0, :3], "o", color="b", linewidth=3.5, markersize=8)

    axis_2.plot(*traj.waypoints[index][0, :3], "*", color="b", linewidth=3.5, markersize=14)

    axis_2.set_title(r'PI', fontsize=c.fontsizetitle)
    axis_2.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_2.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_2.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_2.set_xlim(0, 5.5)
    axis_2.set_ylim(0, 5.5)
    axis_2.set_zlim(0, 2.2)
    axis_2.view_init(elev=30, azim=120)

    targets_INIT = []
    targets_PI = []
    N_sim = 50

    param_list = []
    targets_list =[]



    for j in range(N_sim):

        traj = INIT[k]
        ODE["Array_gains"] = G


        delta_p = np.array([np.random.uniform(-c.dev, c.dev, *np.shape(c.dev))*c.true_p[0],
                            np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                            np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                            np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                            np.random.uniform(0, c.dev2, *np.shape(c.dev2))*c.true_p[4]])

        # delta_p = delta_p_n[j]
        r = delta_p.T @ np.linalg.inv(c.W_range) @ delta_p

        if r > 1:
            delta_p = delta_p/math.sqrt(r)

        ODE["kf"] = c.true_p[0] + delta_p[0]
        ODE["gx"] = c.true_p[1] + delta_p[1]
        ODE["gy"] = c.true_p[2] + delta_p[2]
        ODE["gz"] = c.true_p[3] + delta_p[3]
        ODE["m"] = c.true_p[4] + delta_p[4]




        # gathering the states when perturbating the system for plotting later on
        ODE.apply_parameters()
        model.integrate_along_trajectory(traj, c.N)
        states = model.ODE.last_result[:, model.ODE.states_indices["q"]]


        # times_vec = ODE.time_points
        # u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
        # u_nominal = (np.diff(u_intn, axis=0).T / np.diff(times_vec, axis=0).T).T
        #
        # for j in range(4):
        #     plt.plot(times_vec[1:], u_nominal[:, j]/c.umax)



        indice = np.where(np.isin(ODE.time_points, traj.waypoints_t[index]))[0]
        targets_INIT.append(np.linalg.norm(states[indice, :3] - TARGETS[k][0, :3]))
        axis_1.plot3D(states[:, 0], states[:, 1], states[:, 2], 'g--', linewidth=1, alpha=0.5)
        Fct.draw_ellipse(TARGETS[k][0, :3], states[indice, :3], axis_1, 0.03)

        # collecting data
        param_list.append(delta_p)
        targets_list.append(np.linalg.norm(states[indice, :3] - TARGETS[k][0, :3]))


        traj = PI[k]
        # ODE["Array_gains"] = traj.get_controller_gains()


        # gathering the states when perturbating the system for plotting later on
        ODE.apply_parameters()
        model.integrate_along_trajectory(traj, c.N)
        states = model.ODE.last_result[:, model.ODE.states_indices["q"]]



        times_vec = ODE.time_points
        u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
        u_nominal = (np.diff(u_intn, axis=0).T / np.diff(times_vec, axis=0).T).T

        for j in range(4):
            plt.plot(times_vec[1:], u_nominal[:, j]/c.umax)

        ############################################################################
        # times_vec = ODE.time_points
        # u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
        # u_nominal = (np.diff(u_intn, axis=0).T / np.diff(times_vec, axis=0).T).T
        #
        # for j in range(4):
        #     plt.plot(times_vec[1:], u_nominal[:, j]/c.umax)


        ##########################################################

        targets_PI.append(np.linalg.norm(states[indice, :3] - TARGETS[k][0, :3]))
        axis_2.plot3D(states[:, 0], states[:, 1], states[:, 2], 'g--', linewidth=1, alpha=0.5, label=r'$P \neq P_{c}$ Controller tracking with pertubation')
        Fct.draw_ellipse(TARGETS[k][0, :3], states[indice, :3], axis_2, 0.02)

        # if j == 0:
        #     axis_2.legend(loc='upper center', fancybox=True, ncol=2, bbox_to_anchor=(0.5, 0), fontsize=30)


    param_list = np.array(param_list)
    targets_list = np.array(targets_list)
    indexing = np.argmax(targets_list)
    print(param_list[indexing] + c.true_p)
    plt.show()
    plt.savefig(save_path + '/tracking_'+ "Ntraj_"+ str(k) + '_Nsim_' + str(N_sim) + '.pdf')



plt.show()

targets_INIT = np.array(targets_INIT)
targets_PI = np.array(targets_PI)


fig, ax = plt.subplots(1, 1, figsize=(9, 5), sharey=True)

# data = [targets_INIT, targets_PI]

data = np.vstack((targets_INIT, targets_PI))
data = data.T

df_targets = pd.DataFrame(data)
df_targets.columns = ["INIT", "PI"]

sns.violinplot(data=df_targets, linewidth=4, ax=ax)
ax.set_ylabel(r'Window reach in [\text{m}]', fontsize=c.fontsizetitle)
ax.set_title(r'PX4 Controller', fontsize=c.fontsizetitle)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.yaxis.grid()

ax.set_yticks(np.arange(0, 0.5, 0.1))



plt.savefig(save_path + '/violin_all__'+ '_Nsim_' + str(N_sim) + '.pdf')

print("Finished")