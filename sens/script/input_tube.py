import math
import time
from typing import Sequence
import datetime
import os
from numpy.linalg import norm
from gen.models import jetson_pd as jetson_pd
from utils.trajectory import PiecewiseSplineTrajectory
import pickle
from base64 import b64decode
import numpy as np
import matplotlib.pyplot as plt
from cnst.constant import constants
import utils.Functions as Fct

# latex for figures
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \usepackage{lmodern}'
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern'
          }
plt.rcParams.update(params)

"""class of constants"""
c = constants()

""" Load The Model """

t0 = time.time()
model = jetson_pd.Jetson(c.N_coeffs)
ODE = model.generate(3, verbose=True, mode=jetson_pd.Mode.NOGRAD, overwrite=True)
ODE.set_integrator("dopri5", nsteps=10**6)
model.set_default_state(c.init_waypoint)
t1 = time.time()
print(f"generating took {t1-t0} s")


"""Path to save figures"""

now = datetime.datetime.now()
save_path ='../Paper_Journal_ICRA/save/tube_inputs/tube_check_'+ now.strftime(
    '%Y-%m-%d')
if not os.path.isdir(save_path):
    os.mkdir(save_path)


"""Data Reading trajectory"""
filename = "../Paper_Journal_ICRA/save/trajectories/px4_opt___1_Tf_5_2023-08-17.straj"


INIT: Sequence[PiecewiseSplineTrajectory] = []
PI: Sequence[PiecewiseSplineTrajectory] = []
PI_k: Sequence[PiecewiseSplineTrajectory] = []
PI_ak: Sequence[PiecewiseSplineTrajectory] = []
TARGETS: Sequence[np.ndarray] = []
# cases = [INIT, PI_a, PI_k, PI_ak, TARGETS]
cases = [INIT, PI, TARGETS]
# cases = [INIT, TARGETS]

with open(filename, "rb") as dump:
    for i, line in enumerate(dump):
        cases[i % len(cases)].append(pickle.loads(b64decode(line)))


import csv
"""Choose a trajectory"""

traj = INIT[0]

"""Integrate along the Trajectory"""
model.integrate_along_trajectory(traj, c.N)
states = ODE.last_result
u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
t = ODE.time_points
states_nominal_init = (np.diff(u_intn, axis=0).T / np.diff(t, axis=0).T).T

"""Tube Calculations"""
inputs_max, inputs_min , radius_init = Fct.tube_inputs(states, model)


traj = PI[0]

"""Integrate along the Trajectory"""
model.integrate_along_trajectory(traj, c.N)
states = ODE.last_result
u_intn = model.ODE.last_result[:, model.ODE.states_indices["u_int"]]
t = ODE.time_points
states_nominal = (np.diff(u_intn, axis=0).T / np.diff(t, axis=0).T).T

"""Tube Calculations"""
inputs_max, inputs_min , radius_pi = Fct.tube_inputs(states, model)

# Open a new file for writing
radius_init /= 1100**2
nom_init = states_nominal_init[:-1, 2]/1100**2
radius_pi /= 1100**2
nom_pi = states_nominal[:-1, 2]/1100**2
data = np.column_stack((radius_init, nom_init, radius_pi, nom_pi))
with open(save_path + '/radius.csv', 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    writer.writerow(['init', 'nom_init', 'pi', 'nom_pi'])
    writer.writerows(data)

# delta_p_n = np.array([ [0, 0, 0, 0, 0],
#                        [-3.77719027e-07, -9.42252097e-03, 2.88607907e-02, -6.76093085e-03, 9.94074072e-02],
#                        [-1.92706788e-07, -9.09148502e-03, 3.65899317e-02, -6.48676648e-03, 3.01777379e-02],
#                        [-1.47738861e-07, 2.76141168e-02, -2.60982115e-02, -3.25582855e-03, 6.63055268e-02],
#                        [-1.07137979e-07, 2.05253707e-02, -3.11552278e-02, 1.15097132e-02, 4.74312474e-02],
#                        [-3.25536806e-07, -2.11741035e-02, 2.27194958e-02, -3.56022386e-03, 1.27866053e-01],
#                        [-2.30015793e-07, 1.33930267e-02, -3.42056443e-02, 4.75267044e-03, 2.83083655e-02],
#                        [-7.58090699e-08, -1e-02, 3.10910410e-02, 4.44176317e-03, 1.35e-01],
#                        [-1.96629469e-07, 1.25183775e-02, 3.18787195e-02, -1.16125450e-02, 1.80496306e-02],
#                        [-2.90798214e-07, -1.56401765e-02, 2.90132485e-02, 1.38814982e-03, 1.17953876e-02],
#                        [1.48680533e-07, -1.88639438e-02, -3.33421769e-02, 7.68409140e-03, 9.04144500e-03]
#                     ])

# for i in range(len(delta_p_n)):
#     delta_p_n[i, :] = delta_p_n[i, :] + c.true_p

"""Perturbing the System"""
Nsim = 60 #In pertubation
list_inputs = []
umax = 0
max_delta_p = np.zeros(5)
for i in range (Nsim):
    delta_p = np.array([np.random.uniform(-c.dev, c.dev, *np.shape(c.dev))*c.true_p[0],
                        np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                        np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                        np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                        np.random.uniform(0, 0.9*c.dev2, *np.shape(c.dev2))*c.true_p[4]])

    r = delta_p.T @ np.linalg.inv(c.W_range) @ delta_p
    if r > 1:
        delta_p = delta_p / math.sqrt(r)

    # delta_p = delta_p_n[i]


    ODE["kf"]   = c.true_p[0] + delta_p[0]
    ODE["gx"]   = c.true_p[1] + delta_p[1]
    ODE["gy"]   = c.true_p[2] + delta_p[2]
    ODE["gz"] = c.true_p[3] + delta_p[3]
    ODE["m"]    = c.true_p[4] + delta_p[4]




    # gathering the states when perturbating the system for plotting later on
    ODE.apply_parameters()

    model.integrate_along_trajectory(traj, c.N)
    u_int = model.ODE.last_result[:, ODE.states_indices["u_int"]]
    u = (np.diff(u_int, axis=0).T / np.diff(t, axis=0).T).T
    tmp = np.max(u, axis=0)
    if tmp[1] > umax:
        umax = tmp[1]
        max_delta_p = np.array([ODE["kf"], ODE["gx"], ODE["gy"], ODE["gz"], ODE["m"]])
    list_inputs.append(u)



print(max_delta_p)
print(umax)
"""Plotting the tubes"""
u = ["w1", "w2", "w3", "w4"]

for j, param in enumerate(u):

    figure_0 = plt.figure(figsize=(16, 9))
    ax = figure_0.add_subplot(1, 1, 1)
    plt.axhline(y=c.umax, linestyle='--', color='r', linewidth=4)
    plt.axhline(y=c.umin, linestyle='--', color='r', linewidth=4)
    plt.plot(t[1:], states_nominal[:, j])
    plt.plot(t[1:], inputs_max[:, j], 'r-')
    plt.plot(t[1:], inputs_min[:, j], 'r-')
    for i in range(len(list_inputs)):
        plt.plot(t[1:], list_inputs[i][:, j], 'k--')

    plt.axhline(y=0.8*c.umax, linestyle='--', color='m', linewidth=2)
    ax.set_xlabel(r'$time (sec)\ [\text{m}]$', fontsize=c.fontsizelabel)
    ax.set_ylabel("The input " + param, fontsize=c.fontsizetitle)

    ax.set_title("Tube of the input " + param + " with perturbating the parameters kf, gx, gy and m ")

    plt.savefig(save_path + '/input_tube_check' + "-" + " All Params " + " _ " + " of the input " + " _ " + param + '.pdf')


print("Finished")

