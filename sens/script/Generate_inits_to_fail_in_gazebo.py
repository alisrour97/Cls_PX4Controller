import time
import datetime
from gen.models import jetson_pd as jetson_pd
from utils.trajectory import PiecewiseSplineTrajectory
from scipy.linalg import norm
from opt.optimize import sensitivity_gain_optimisation, sensitivity_optimisation, sensitivity_joint_optimisation
import matplotlib.pyplot as plt
import pickle
from base64 import b64encode
import numpy as np
import random
import math
from cnst.constant import constants
import utils.Functions as Fct
from opt.optimize import cost


c = constants() # class where we define all the constants to be used in all the framework
# latex for figures
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \usepackage{lmodern}'
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern'
          }
plt.rcParams.update(params)
plt.rcParams.update({'figure.max_open_warning': 0})

now = datetime.datetime.now()



#######################################################################################3
###saving trajectories
traj_filename = "../Paper_Journal_ICRA/save/trajectories/px4_inits_2_" + str(c.N_traj) +"_" + "Tf_"+ str(c.Tf)+ "_" + now.strftime('%Y-%m-%d') +'.straj'
t0 = time.time()
model = jetson_pd.Jetson(c.N_coeffs)
ODE = model.generate(3, verbose=True, mode=jetson_pd.Mode.NOGRAD, overwrite=True)
model.set_default_state(c.init_waypoint)
model.ODE.set_integrator("dopri5")
t1 = time.time()
print(f" Loading the model took {t1 - t0} s")
########################################
PI_mask = np.full((jetson_pd.N_states, jetson_pd.N_par), False)
PI_mask[:3, :] = True  # only improve the xyz position for all parameters

nlc = lambda grad, x, time_vec, states: model.nonlcon_init(grad, states, time_vec, c.umin, c.umax, 0.01)

trajectories = np.empty(c.N_traj, np.object_)
INIT_JustWaypoints = np.empty(c.N_traj, np.object_)
INIT_InputSat_out = np.empty(c.N_traj, np.object_)
INIT_Target_out = np.empty(c.N_traj, np.object_)

INIT = np.empty(c.N_traj, np.object_)
PI = np.empty(c.N_traj, np.object_)
PI_g = np.empty(c.N_traj, np.object_)
PI_c_g = np.empty(c.N_traj, np.object_)


with open(traj_filename, "wb") as dump:
    pass

###########################################

# Initialize waypoints
point1 = c.init_waypoint






#opt step for PI and time
##########################################################################################
t_start = time.time()
i = 0

while i < c.N_traj: # loop for the trajectories

    # # Preconditioning for opt
    t0 = time.time()
    # (np.random.rand() - 0.5)
    # The window

    index = -3

    point3 = np.vstack([
        [2.7, 1.7, 1.8, 0],
        [2, 2, -0.5, 0],
        [-0.5, -0.5, 0, 0],
    ])

    point5 = np.vstack([
        [4, 4, 1, 0],
        [0, 0, 0, 0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    t1 = time.time()
    wp = [point1, point3, point5]
    wp_t = np.array([0, 2 * c.Tf / c.N_pieces, c.Tf])


    traj = PiecewiseSplineTrajectory(wp_t, wp)
    traj.interpolate_waypoint_at(0.8 * c.Tf / c.N_pieces, free=True)
    traj.interpolate_waypoint_at(2.8 * c.Tf / c.N_pieces, free=True)
    for m in range(1, c.N_waypoints):
        traj.free_waypoint(m)  # free all points which are not the origin

    traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    traj._waypoints_mask[index, :1, :] = False  # Fix window
    traj._waypoints_mask[index, 1, 2:] = False  # Fix window
    traj._waypoints_mask[:, :, 3] = False
    traj._waypoints_mask[-1, :, :] = False  # Fix final point


    print(traj.waypoints_mask)



    # index = -3
    #
    # point3 = np.vstack([
    #     [1.2, 1.4, 1.5, 0],
    #     [1, 1, 0, 0],
    #     [0, 0, 0, 0],
    # ])
    #
    # point5 = np.vstack([
    #     [4.2, 4.2, 1.5, 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    #
    # t1 = time.time()
    # wp = [point1, point5]
    # wp_t = np.array([0, c.Tf])
    #
    #
    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    # traj.interpolate_waypoint_at(1 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(2 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(3 * c.Tf / c.N_pieces, free=True)
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # traj._waypoints_mask[index, :1, :] = False  # Fix window
    # traj._waypoints_mask[:, :, 3] = False
    # traj._waypoints_mask[-1, :, :] = False  # Fix final point


    # print(traj.waypoints)
    #
    # print(traj.waypoints_mask)


    cost_init, t_cost = traj.optimize(model, nlc, traj.waypoints[index], index=index, use_target=True, lower_bounds=c.lb,
                                         upper_bounds=c.ub, optim_time=c.init_opt_time, N=c.N, dx_init=c.dx_init)


    t = np.linspace(0, t_cost, len(cost_init))

    plt.plot(t, cost_init)


    # cost1 = cost(traj, ODE, num=500, index=index)
    # print(cost1)

    INIT[i] = traj.deepcopy()  # save

    figure_1 = plt.figure(figsize=(16, 9))

    # # # Choose a trajectory
    # traj = INIT[i]


    model.integrate_along_trajectory(traj, c.N)  # inorder to get the last results
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
    times_vec = ODE.time_points

    u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
    u_nominal = (np.diff(u_intn, axis=0).T / np.diff(times_vec, axis=0).T).T

    indice = np.where(np.isin(times_vec, traj.waypoints_t[index]))[0]

    d = states[indice, :3] - traj.waypoints[index, 0, :3]
    print(f'distance to the window is the following: {d}')

    figure_0 = plt.figure(figsize=(16, 9))
    ax = figure_0.add_subplot(1, 1, 1)
    plt.axhline(y=1, linestyle='--', color='r', linewidth=4)
    for j in range(4):
        plt.plot(times_vec[1:], u_nominal[:, j]/c.umax)

    plt.axhline(y=0.8, linestyle='--', color='m', linewidth=2)

    ax.set_xlabel(r'$time (sec)\ [\text{m}]$', fontsize=c.fontsizelabel)
    ax.set_ylabel("The input " , fontsize=c.fontsizetitle)
    ax.set_ylim(0, 1.2)


    # traj.Construct_trajectory()
    traj.plotting()
    axis_1 = figure_1.add_subplot(1, 1, 1, projection="3d")
    axis_1.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k-', linewidth=3)
    axis_1.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2.5)
    for l in range(len(traj.waypoints)):
        if l != c.N_waypoints + index:
            axis_1.plot(*traj.waypoints[l][0, :3], "o", color="b", linewidth=3.5, markersize=8)
    axis_1.plot(*traj.waypoints[index][0, :3], "*", color="b", linewidth=3.5, markersize=12)
    axis_1.set_title(r'INIT', fontsize=c.fontsizetitle)
    axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_xlim(-1, 5)
    axis_1.set_ylim(-1, 5)
    axis_1.set_zlim(0, 3)

    plt.show()


    with open(traj_filename, "ab") as dump:
        dump.write(b"\n".join(map(lambda x: b64encode(pickle.dumps(x)), [INIT[i]])))
        dump.write(b"\n" + b64encode(pickle.dumps(wp[1])))
        dump.write(b"\n")
    plt.pause(1e-6)
    i = i + 1

t_end = time.time()

print(f'########################################')
print(f"GENERATED {c.N_traj} TRAJECTORIES IN {t_end - t_start} s")
print(f'########################################')
