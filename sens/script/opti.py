import time
import datetime

from gen.models import jetson_pd as jetson_pd
from utils.trajectory import PiecewiseSplineTrajectory
from scipy.linalg import norm
from opt.optimize import sensitivity_gain_optimisation, sensitivity_optimisation, sensitivity_joint_optimisation
# from opt.optimize import sensitivity_gain_optimisation, sensitivity_optimisation, sensitivity_joint_optimisation
import matplotlib.pyplot as plt
import pickle
from base64 import b64encode
import numpy as np
import random
import math
from cnst.constant import constants
import utils.Functions as Fct
from typing import Sequence
import pickle
from base64 import b64decode
from utils.Functions import *

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
traj_filename = "../Paper_Journal_ICRA/save/trajectories/px4_opt__with_gains_" + str(c.N_traj) +"_" + "Tf_"+ str(c.Tf)+ "_" + now.strftime('%Y-%m-%d') +'.straj'
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

nlc = lambda grad, x, time_vec, states: model.nonlcon(grad, states, time_vec, c.umin, c.umax, 0.01)
nlc_init = lambda grad, x, time_vec, states: model.nonlcon_init(grad, states, time_vec, c.umin, c.umax, 0.01)

# trajectories = np.empty(c.N_traj, np.object_)
# INIT_JustWaypoints = np.empty(c.N_traj, np.object_)
# INIT_InputSat_out = np.empty(c.N_traj, np.object_)
# INIT_Target_out = np.empty(c.N_traj, np.object_)

INIT: Sequence[PiecewiseSplineTrajectory] = [] # initial
PI = np.empty(c.N_traj, np.object_)
# PI_g = np.empty(c.N_traj, np.object_)
PI_c_g = np.empty(c.N_traj, np.object_)
TARGETS: Sequence[np.ndarray] = []# Targets
# For init
filename = "../Paper_Journal_ICRA/save/trajectories/px4_inits_2_1_Tf_5_2023-08-17.straj"
# Read all of this from the filename
cases = [INIT, TARGETS] # all cases in a list
with open(filename, "rb") as dump:
    for i, line in enumerate(dump):
        cases[i % len(cases)].append(pickle.loads(b64decode(line)))


with open(traj_filename, "wb") as dump:
    pass

###########################################




#opt step for PI and time
##########################################################################################
t_start = time.time()
i = 0


# Initialize waypoints
point1 = c.init_waypoint

            # G = np.zeros(15)
            # for i in range(4):
            #     G[3*i] = x_t[len(x_ini):len(x_ini)+i+1]
            #     G[3*i+1] = x_t[len(x_ini):len(x_ini)+i+1]
            #     G[3*i+2] = x_t[len(x_ini):len(x_ini)+i+2]
            # G[12:] = x_t[len(x_ini)+12:]
            # print(G)


G = ODE["Array_gains"]


while i < c.N_traj: # loop for the trajectories




    # point3 = np.vstack([
    #     [2, 2, 2.2, 0],
    #     [0, 2, 0, 0],
    #     [0, 1, 0, 0],
    # ])
    #
    # point5 = np.vstack([
    #     [3.5, 3.5, 1, 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    #
    # t1 = time.time()
    # wp = [point1, point3, point5]
    # wp_t = np.array([0, c.Tf / 2, c.Tf])
    #
    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    #
    # traj.interpolate_waypoint_at(1 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(2 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(3 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(5 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(6 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(7 * c.Tf / c.N_pieces, free=True)
    #
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # # traj._waypoints_mask[1, 0, :] = False # 1st Intermediate Fix xyz
    # traj._waypoints_mask[4, :2, :] = False  # window Fix xyz
    # traj._waypoints_mask[4, 1, 1] = True  # window variable vy
    # traj._waypoints_mask[8, :, :] = False  # Fix final point


    # point3 = np.vstack([
    #     [4, 2, 2.1, 0],
    #     [0, 3, 0, 0],
    #     [0, 3, 0, 0],
    # ])
    #
    # point5 = np.vstack([
    #     [0, 4, 1, 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])

    # index = -4
    # point3 = np.vstack([
    #     [2, 2, 2.1, 0],
    #     [0, 2.5, -0.7, 0],
    #     [0, 2.5, -0.7, 0],
    # ])
    #
    # point5 = np.vstack([
    #     [3.5, 3.5, 1, 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    #
    # t1 = time.time()
    # wp = [point1, point3, point5]
    # wp_t = np.array([0, c.Tf / 2, c.Tf])
    #
    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    #
    # traj.interpolate_waypoint_at(1 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(2 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(4 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(5 * c.Tf / c.N_pieces, free=True)
    #
    #
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # traj._waypoints_mask[3, :1, :] = False
    # traj._waypoints_mask[6, :, :] = False  # Fix final point
    #
    #
    #
    # cost_init = traj.optimize(model, nlc_init, wp[1], index=index, use_target=True, lower_bounds=c.lb,
    #                                      upper_bounds=c.ub, optim_time=c.init_opt_time, N=c.N, dx_init=c.dx_init)



    ###########################################################################
    # index = -1
    #
    # point3 = np.vstack([
    #     [2.5, 2.5, 2.2, 0],
    #     [2, 2, 0, 0],
    #     [2, 2, 0, 0],
    # ])
    #
    # point5 = np.vstack([
    #     [4, 4, 1.8, 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    #
    # t1 = time.time()
    # wp = [point1, point3, point5]
    # wp_t = np.array([0, 2 * c.Tf / c.N_pieces, c.Tf])
    #
    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    # traj.interpolate_waypoint_at(1 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(3 * c.Tf / c.N_pieces, free=True)
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # traj._waypoints_mask[-1, :, :] = False  # Fix final point
    #
    # cost_init = traj.optimize(model, nlc, wp[-1], index=index, use_target=True, lower_bounds=c.lb,
    #                           upper_bounds=c.ub, optim_time=c.init_opt_time, N=c.N, dx_init=c.dx_init)

    # index = -1
    # point5 = np.vstack([
    #     [3.5 + (np.random.rand() - 0.5), 3.5 + (np.random.rand() - 0.5), 1.5 + (np.random.rand() - 0.5), 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    #
    # t1 = time.time()
    # wp = [point1, point5]
    # wp_t = np.array([0, c.Tf])
    #
    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    # traj.interpolate_waypoint_at(1 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(2 * c.Tf / c.N_pieces, free=True)
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # traj._waypoints_mask[-1, :, :] = False  # Fix final point
    #
    # print(traj.waypoints_mask)

    # index = -2
    #
    # point3 = np.vstack([
    #     [2, 2, 2, 0],
    #     [1, 1, 0, 0],
    #     [0, 0, 0, 0],
    # ])
    # point5 = np.vstack([
    #     [3.5 + (np.random.rand() - 0.5), 3.5 + (np.random.rand() - 0.5), 1.5 + (np.random.rand() - 0.5), 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    #
    # t1 = time.time()
    # wp = [point1, point3, point5]
    # wp_t = np.array([0, 1.2 * c.Tf / c.N_pieces, c.Tf])
    #
    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    # # traj.interpolate_waypoint_at(1 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(2 * c.Tf / c.N_pieces, free=True)
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # traj._waypoints_mask[index, :1, :] = False  # Fix final point

    # index = -2
    #
    # point3 = np.vstack([
    #     [2, 2, 2, 0],
    #     [1, 1, 0, 0],
    #     [0, 0, 0, 0],
    # ])
    # point5 = np.vstack([
    #     [3.5 + (np.random.rand() - 0.5), 3.5 + (np.random.rand() - 0.5), 1.5 + (np.random.rand() - 0.5), 0],
    #     [0, 0, 0, 0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    #
    # t1 = time.time()
    # wp = [point1, point3, point5]
    # wp_t = np.array([0, 1.2 * c.Tf / c.N_pieces, c.Tf])
    #
    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    # # traj.interpolate_waypoint_at(1 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(2 * c.Tf / c.N_pieces, free=True)
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # traj._waypoints_mask[index, :1, :] = False  # Fix final point

    index = -3
    traj = INIT[i]

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

    # traj = PiecewiseSplineTrajectory(wp_t, wp)
    # traj.interpolate_waypoint_at(1.05 * c.Tf / c.N_pieces, free=True)
    # traj.interpolate_waypoint_at(2.5 * c.Tf / c.N_pieces, free=True)
    # for m in range(1, c.N_waypoints):
    #     traj.free_waypoint(m)  # free all points which are not the origin
    #
    # traj._waypoints_t_mask[:] = False  # For time masking : Fix all the times
    # traj._waypoints_mask[index, :1, :] = False  # Fix window
    # traj._waypoints_mask[index, 2:, :] = False
    # traj._waypoints_mask[index, 1, 1:] = False
    # traj._waypoints_mask[:, :, 3] = False
    # traj._waypoints_mask[-1, :, :] = False  # Fix final point


    print(traj.waypoints)

    print(traj.waypoints_mask)


    cost_init, t_cost = traj.optimize(model, nlc, traj.waypoints[index], index=index, use_target=True, lower_bounds=c.lb,
                                         upper_bounds=c.ub, optim_time=c.init_opt_time, N=c.N, dx_init=c.dx_init)



    # cost_init = traj.optimize(model, nlc, wp[-1], index=index, use_target=True, lower_bounds=c.lb,
    #                           upper_bounds=c.ub, optim_time=c.init_opt_time, N=c.N, dx_init=c.dx_init)


    # cost_init = traj.optimize(model, nlc, traj.waypoints[index], index=index, use_target=True, lower_bounds=c.lb,
    #                           upper_bounds=c.ub, optim_time=c.init_opt_time, N=c.N, dx_init=c.dx_init)

    #############################################################################



    traj._waypoints_mask[index, 2:, :] = True
    traj._waypoints_mask[index, 1, 1:] = True
    traj._waypoints_mask[1:-1, :, 3] = True
    traj._waypoints_mask[-1, 0, :3] = True

    print(traj.waypoints_mask)

    model.integrate_along_trajectory(traj, c.N)  # inorder to get the last results
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
    times_vec = ODE.time_points
    indice = np.where(np.isin(times_vec, wp_t[1]))[0]
    d = states[indice, :3] - point3[0, :3]
    print(f'distance to the window is the following: {d}')

    # figure_1 = plt.figure(figsize=(16, 9))


    # ##trajectory construction of the reference with all pos, vel and acceleration
    # figure_1 = plt.figure(figsize=(16, 9))
    # traj.Construct_trajectory()
    # axis_1 = figure_1.add_subplot(1, 1, 1, projection="3d")
    # axis_1.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k-', linewidth=3)
    # axis_1.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2.5)
    # for l in range(len(traj.waypoints) - 1):
    #     axis_1.plot(*traj.waypoints[l][0, :3], "o", color="b", linewidth=3.5, markersize=8)
    # axis_1.plot(*traj.waypoints[-1][0, :3], "*", color="b", linewidth=3.5, markersize=12)
    # axis_1.set_title(r'INIT', fontsize=c.fontsizetitle)
    # axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
    # axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
    # axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
    # axis_1.set_xlim(0, 6)
    # axis_1.set_ylim(0, 6)
    # axis_1.set_zlim(0.5, 2)
    #
    # plt.show()




    # PI[i], PI_opt_cost, cost_time, T_PI = sensitivity_optimisation(model, traj, nonlcon=nlc,
    #                                                   target_point=traj.waypoints[index], index=index, lower_bounds=c.lb, upper_bounds=c.ub,
    #                                                   PI_mask=PI_mask, optimization_time=c.optimization_time_PI, delta= c.dx_PI)





    # PI_g[i], PI_opt_cost_g, cost_time_g= sensitivity_gain_optimisation(model, INIT[i], nonlcon=nlc,
    #                                                   target_point=target_points[i], PI_mask=PI_mask, gains=gains,
    #                                                   optimization_time=c.optimization_time_PI, delta =c.dx_PI)



    PI_c_g[i], PI_opt_cost_c_g, cost_time_c_g, T_PI = sensitivity_joint_optimisation(model, INIT[i], nonlcon=nlc,
                                                      target_point=traj.waypoints[index], index=index, lower_bounds=c.lb, upper_bounds=c.ub,
                                                      PI_mask=PI_mask, gains=G, optimization_time=c.optimization_time_PI, delta =c.dx_PI)



    model.integrate_along_trajectory(PI_c_g[i], c.N)  # inorder to get the last results
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
    times_vec = ODE.time_points
    indice = np.where(np.isin(times_vec, wp_t[1]))[0]
    d1 = states[indice, :3] - point3[0, :3]
    print(f'distance to the window is the following: {d1}')


    t = np.linspace(0, T_PI, len(cost_time_c_g))
    plt.plot(t, cost_time_c_g)
    plt.show()

    with open(traj_filename, "ab") as dump:
        # dump.write(b"\n".join(map(lambda x: b64encode(pickle.dumps(x)), [INIT[i], PI[i], PI_c_g[i], PI_g[i]])))
        dump.write(b"\n".join(map(lambda x: b64encode(pickle.dumps(x)), [INIT[i], PI_c_g[i]])))
        dump.write(b"\n" + b64encode(pickle.dumps(wp[1])))
        dump.write(b"\n")
    plt.pause(1e-6)
    i = i + 1


t_end = time.time()

print(f'########################################')
print(f"GENERATED {c.N_traj} TRAJECTORIES IN {t_end - t_start} s")
print(f'########################################')


