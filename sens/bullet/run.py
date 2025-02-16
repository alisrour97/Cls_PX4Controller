import os
import time
from typing import Sequence
import numpy as np
import pybullet as p
import pybullet_data
from cnst.constant import constants
from utils.trajectory import PiecewiseSplineTrajectory
import pickle
from base64 import b64decode
from gen.models import jetson_pd as jetson_pd
from pybullet_utils import bullet_client as bc
import math
import cv2
c = constants()


p.connect(p.GUI)  # Connect to PyBullet
p.setAdditionalSearchPath(pybullet_data.getDataPath())


planeId = p.loadURDF("plane.urdf")

# Setting all components of the simulation
p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 3, 1.5])



window_orientation = p.getQuaternionFromEuler([-80, 0, 90])  # No rotation
# Create a box
window_position = [0 + 0.3, 0 - 0.4, 1]  # X, Y, Z coordinates of the box
window_id = p.loadURDF("Mine/window.urdf", basePosition=window_position, baseOrientation=window_orientation)

# Adjust the collision properties to enable repulsion
p.changeDynamics(window_id, -1, restitution=1.0)  # Set the restitution coefficient to 1.0 for maximum rebound
# Disable gravity for the window object by setting its mass to 0
p.changeDynamics(window_id, -1, mass=0.0)






"""Model initialization"""

t0 = time.time()
model = jetson_pd.Jetson(c.N_coeffs)
ODE = model.generate(3, verbose=True, mode=jetson_pd.Mode.NOGRAD, overwrite=True) # true if you changed the model and need to compile, otherwise false
ODE.set_integrator("dopri5")
model.set_default_state(c.init_waypoint)
t1 = time.time()
print(f"generating of PD model took {t1-t0} s")

"""Import the trajectories"""
filename = "../Paper_Journal_ICRA/save/trajectories/px4_opt___1_Tf_5_2023-08-16.straj"
# filename = "../Paper_Journal_ICRA/save/trajectories/px4_opt_1_Tf_5.5_2023-06-18.straj"
INIT: Sequence[PiecewiseSplineTrajectory] = [] # initial
PI: Sequence[PiecewiseSplineTrajectory] = [] # optimized for a
TARGETS: Sequence[np.ndarray] = [] # Targets
cases = [INIT, PI, TARGETS] # all cases in a list

# Read all of this from the filename
with open(filename, "rb") as dump:
    for i, line in enumerate(dump):
        cases[i % len(cases)].append(pickle.loads(b64decode(line)))




traj = INIT[0]

wpts = traj.waypoints
pos_iris = wpts[0][0, :3]
iris_id = p.loadURDF("iris/iris.urdf", basePosition=pos_iris, baseOrientation=[0, 0, 0, 1])

for w in range(len(traj.waypoints)):

    # Set the color of the sphere to transparent red
    color = [0, 0, 1, 0.5]  # RGBA color values, where 0.5 represents a transparency of 50%
    # Create a small sphere
    radius = 0.05
    sphere_collision_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
    sphere_visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius)
    # Create the sphere body
    sphere_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_collision_id,
                                  baseVisualShapeIndex=sphere_visual_id)
    # Set the initial position and orientation of the sphere
    initial_position = wpts[w][0, :3]  # X, Y, Z coordinates of the desired position
    initial_orientation = [0, 0, 0, 1] # Quaternion orientation (no rotation)
    p.resetBasePositionAndOrientation(sphere_id, initial_position, initial_orientation)
    p.changeVisualShape(sphere_id, -1, rgbaColor=color)

time.sleep(5)

Nsim = 20
counter_collision = 0
n = 0
text_id2 = p.addUserDebugText(f"Collision Counter: {counter_collision}     Nsim: 1", window_position + np.array([0, 0, 1]),
                              textColorRGB=[0, 1, 0], textSize=3)




for n in range(Nsim):


    if n > 0:

        delta_p = np.array([np.random.uniform(-c.dev, c.dev, *np.shape(c.dev)) * c.true_p[0],
                            np.random.uniform(-c.dev, c.dev, *np.shape(c.dev)) * c.true_p[1],
                            np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                            np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                            np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                            np.random.uniform(0, c.dev2, *np.shape(c.dev2)) * c.true_p[5]])

        # r = delta_p.T @ np.linalg.inv(c.W_range) @ delta_p
        # delta_p = delta_p / math.sqrt(r)

        ODE["kf"] = c.true_p[0] + delta_p[0]
        ODE["ktau"] = c.true_p[1] + delta_p[1]
        ODE["gx"] = c.true_p[2] + delta_p[2]
        ODE["gy"] = c.true_p[3] + delta_p[3]
        ODE["gz"] = c.true_p[4] + delta_p[4]
        ODE["m"] = c.true_p[5] + delta_p[5]

        # gathering the states when perturbating the system for plotting later on
        ODE.apply_parameters()


    model.integrate_along_trajectory(traj, c.N)  # inorder to get the last results
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
    #####################################


    ##################################

    # Enable the real-time simulation
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81)
    control_dt = c.Ts
    p.setTimestep = control_dt
    state_t = 0.
    i = 0

    # Visualization parameters
    if n == 0:
        lineColor = [0, 0, 0]  # red color for the line
        lineWidth = 4  # width of the line
    else:
        lineColor = [1, 0, 0]  # red color for the line
        lineWidth = 2  # width of the line

    tmp = 0
    text_id = p.addUserDebugText("Window", window_position + np.array([0, 0, 0.5]), textColorRGB=[1, 0, 0], textSize=0.5)

    while i < len(states):
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        for joint in range(1, 5):
            p.setJointMotorControl2(iris_id, joint, p.POSITION_CONTROL, 100)

        pos_iris = states[i, :3]
        ori_iris = np.concatenate((states[i, 7:10], np.atleast_1d(states[i, 6])))

        p.resetBasePositionAndOrientation(iris_id, pos_iris, ori_iris)

        if i > 0:
            p.addUserDebugLine(states[i-1, :3], states[i, :3], lineColor, lineWidth)

        contact = p.getContactPoints(iris_id, window_id)

        if len(contact) > 0:
            tmp = 1
            p.addUserDebugText(f" Warning Collision Detected!!", window_position + np.array([0, 0, 0.5]),
                               textColorRGB=[1, 0, 0], textSize=2, replaceItemUniqueId=text_id)


        time.sleep(control_dt)
        i += 1
        state_t += control_dt
        p.stepSimulation()


    if tmp == 1:
        p.removeUserDebugItem(text_id)
        counter_collision = counter_collision + 1
        # Remove previous text
        if text_id2 is not None:
            p.removeUserDebugItem(text_id2)
        text_id2 = p.addUserDebugText(f"Collision Counter: {counter_collision}     Nsim: {n+1}", window_position + np.array([0, 0, 1]),
                           textColorRGB=[0, 1, 0], textSize=3)

    if tmp == 0:
        if text_id2 is not None:
            p.removeUserDebugItem(text_id2)
        text_id2 = p.addUserDebugText(f"Collision Counter: {counter_collision}     Nsim: {n+1}", window_position + np.array([0, 0, 1]),
                           textColorRGB=[0, 1, 0], textSize=3)

# pos_iris = states[-1, :3]
# ori_iris = np.concatenate((states[-1, 7:10], np.atleast_1d(states[-1, 6])))

time.sleep(10)
p.disconnect()  # Disconnect from the simulation




