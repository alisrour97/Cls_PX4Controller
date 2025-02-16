import os
import time
import inspect
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
import csv
import pyulog.ulog2csv as ulog2csv
from utils.Functions import extract_number
c = constants()


p.connect(p.GUI)  # Connect to PyBullet
p.setAdditionalSearchPath(pybullet_data.getDataPath())


planeId = p.loadURDF("plane.urdf")



Fixed_height = 1
index_w = 0.5   #  should be between [0, 1]
pos_iris = np.array([-2, -2, 1])


#####Variable according to the trajectory

# point_w = np.array([0.7, -0.3, 1.8])
# Fixed_time = 4.99 # In seconds

point_w = np.array([2, 0, 2.1])
Fixed_time = 6.99 #In seconds


###### cariable according to the target location
# window_orientation = p.getQuaternionFromEuler([0, 0, 10])  # No rotation
window_orientation = p.getQuaternionFromEuler([0, 0, 20])  # No rotation
# Create a box
# window_position = point_w + np.array([-0.38, 0.35, 0])  # X, Y, Z coordinates of the box
window_position = point_w  # X, Y, Z coordinates of the box


# Setting all components of the simulation
# p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=-195.2, cameraPitch=-17.4, cameraTargetPosition=point_w)
# p.resetDebugVisualizerCamera(cameraDistance=2.1, cameraYaw=-97.2, cameraPitch=-20.6, cameraTargetPosition=point_w)
p.resetDebugVisualizerCamera(cameraDistance=2.1, cameraYaw=656.4, cameraPitch=-29.4, cameraTargetPosition=point_w)

######################################################################################
# window_position = np.array([-0.4, 0, 1])  # X, Y, Z coordinates of the box
# window_id = p.loadURDF("Mine/window_2.urdf", basePosition=window_position, baseOrientation=window_orientation)
window_id = p.loadURDF("Mine/ring.urdf",  basePosition=window_position, baseOrientation=window_orientation)
# Adjust the collision properties to enable repulsion
p.changeDynamics(window_id, -1, restitution=1.0)  # Set the restitution coefficient to 1.0 for maximum rebound
# Disable gravity for the window object by setting its mass to 0
p.changeDynamics(window_id, -1, mass=0.0)

#################################################################################################################################

output = "/home/tesla/Desktop/Sensitivity_Exp/sensitivity_cls/sens/bullet/out"
TEST_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


messages = ["vehicle_local_position", "vehicle_local_position_setpoint", "vehicle_attitude"]
messages_str = ','.join(messages)
delimiter = ','
time_s = 0
time_e = 0

pos_keys = ['timestamp', 'x', 'y', 'z']
att_keys = ['timestamp', 'q[0]', 'q[1]', 'q[2]', 'q[3]']


# dir_name = "In_init/"
dir_name = "In_pi/"


############################################# Point of interest ###########3


# Set the color of the sphere to transparent red
color = [0, 0, 1, 0.7]  # RGBA color values, where 0.5 represents a transparency of 50%
# Create a small sphere
radius = 0.05
sphere_collision_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
sphere_visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius)
# Create the sphere body
sphere_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_collision_id,
                              baseVisualShapeIndex=sphere_visual_id)
# Set the initial position and orientation of the sphere
initial_position = point_w  # X, Y, Z coordinates of the desired position
initial_orientation = [0, 0, 0, 1]  # Quaternion orientation (no rotation)
p.resetBasePositionAndOrientation(sphere_id, initial_position, initial_orientation)
p.changeVisualShape(sphere_id, -1, rgbaColor=color)

################################################################################################################

directory_path = os.path.join(TEST_PATH, dir_name)
# Iterate over all files in the directory
iris_id = p.loadURDF("iris/iris.urdf", basePosition=pos_iris, baseOrientation=[0, 0, 0, 1])




# Get a list of all .ulg files in the directory and sort them by the extracted number
ulg_files = [filename for filename in os.listdir(directory_path) if filename.endswith('.ulg')]
sorted_ulg_files = sorted(ulg_files, key=extract_number)

counter = 0
n = 0
counter_collision = 0
text_id2 = p.addUserDebugText(f"Collision Counter: {counter_collision}     Nsim: 1", window_position + np.array([-0.5, 0, 1]),
                              textColorRGB=[0, 0, 0], textSize=2)


for filename in sorted_ulg_files:

    if filename.endswith('.ulg'):  # Check if the file has a .ulg extension
        # Construct the full path of the log file
        log_file_path = os.path.join(directory_path, filename)

        # Perform operations on the log file
        print("Processing log file:", log_file_path)
        # Add your code here to perform specific operations on the log file

        # Convert log files to CSV
        ulog2csv.convert_ulog2csv(log_file_path, messages=messages_str, output=output, delimiter=delimiter, time_s=time_s, time_e=time_e)

    # Open the CSV file
    with open('out/'+ os.path.splitext(filename)[0] +'_vehicle_local_position_setpoint_0.csv', 'r') as file:
        reader = csv.DictReader(file)
        # Get the fieldnames from the CSV file
        fieldnames = reader.fieldnames
        # Find the indices of the keys in the fieldnames
        key_indices = [fieldnames.index(key) for key in pos_keys]
            # Initialize lists to store the time and values
        pos_ref = [[] for _ in key_indices]

        # Iterate over each row in the CSV file
        for row in reader:
            # Get the values for the keys
            for i, index in enumerate(key_indices):
                key_value = float(row[fieldnames[index]])
                pos_ref[i].append(key_value)

    pos_ref = np.array(pos_ref)
    pos_ref = pos_ref.T
    # Identify the indices where x, y, z are not NaN
    valid_indices = ~np.isnan(pos_ref[:, 1])  # Assuming x values are in the first column
    # Find the first and last valid indices
    first_valid_index = np.argmax(valid_indices)
    last_valid_index = len(pos_ref) - 1 - np.argmax(valid_indices[::-1])
    # Delete NaN values
    pos_ref = pos_ref[first_valid_index:last_valid_index + 1, :]
    pos_ref[:, -1] *= -1
    pos_ref[:, 0] /= 1e6

    # Find the indices where 'z' changes from 1
    change_indices = np.where(np.diff(pos_ref[:, 3]) != 0)[0]
    # Find the index where 'z' first changes from 1 and consider indices after that
    start_index = change_indices[np.argmax(pos_ref[change_indices, -1] != Fixed_height)] - 1

    pos_ref = pos_ref[start_index:-1, :]
    # pos_ref[:, 0] -= pos_ref[0, 0]

    time_start = pos_ref[0, 0]
    tmp = np.where(pos_ref[:, 0] >= time_start + Fixed_time)
    index_f = np.min(tmp)
    time_finish = pos_ref[index_f, 0]

    pos_ref[:, 1], pos_ref[:, 2] = pos_ref[:, 2].copy(), pos_ref[:, 1].copy()

    # Open the CSV file
    with open('out/' + os.path.splitext(filename)[0] + '_vehicle_local_position_0.csv', 'r') as file:
        reader = csv.DictReader(file)
        # Get the fieldnames from the CSV file
        fieldnames = reader.fieldnames
        # Find the indices of the keys in the fieldnames
        key_indices = [fieldnames.index(key) for key in pos_keys]

        # Initialize lists to store the time and values
        pos = [[] for _ in key_indices]
        # Iterate over each row in the CSV file
        for row in reader:
            # Get the values for the keys
            for i, index in enumerate(key_indices):
                key_value = float(row[fieldnames[index]])
                pos[i].append(key_value)

    pos = np.array(pos)
    pos = pos.T

    pos[:, 0] /= 1e6
    # pos[:, 2:4] *= -1
    pos[:, -1] *= -1
    # pos[:, -2] *= -1
    index_low = np.argmax(pos[:, 0] >= time_start)
    index_high = np.argmax(pos[:, 0] >= time_finish)

    pos = pos[index_low:index_high, :]

    pos[:, 1], pos[:, 2] = pos[:, 2].copy(), pos[:, 1].copy()

    # Open the CSV file
    with open('out/' + os.path.splitext(filename)[0] + '_vehicle_attitude_0.csv', 'r') as file:
        reader = csv.DictReader(file)
        # Get the fieldnames from the CSV file
        fieldnames = reader.fieldnames
        # Find the indices of the keys in the fieldnames
        key_indices = [fieldnames.index(key) for key in att_keys]

        # Initialize lists to store the time and values
        q_states = [[] for _ in key_indices]
        # Iterate over each row in the CSV file
        for row in reader:
            # Get the values for the keys
            for i, index in enumerate(key_indices):
                key_value = float(row[fieldnames[index]])
                q_states[i].append(key_value)


    q_states = np.array(q_states)
    q_states = q_states.T
    q_states[:, 0] /= 1e6
    q_states = q_states[index_low:index_high, :]

    q_states[:, 1], q_states[:, 2], q_states[:, 3], q_states[:, 4] = q_states[:, 2].copy(), q_states[:, 3].copy(), q_states[:, 4].copy(), q_states[:, 1].copy()
    ##########################################################
    ##########################################################

    min_distance = float('inf')
    min_distance_index = None
    for l in range(len(pos[:index_f])):
        dist = np.linalg.norm(point_w - pos[l, 1:])

        if dist < min_distance:
            min_distance = dist
            min_distance_index = l

    # Set the color of the sphere to transparent red
    color2 = [1, 0, 0, 1]  # RGBA color values, where 0.5 represents a transparency of 50%
    # Create a small sphere
    radius2 = 0.015
    sphere_collision_id2 = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius2)
    sphere_visual_id2 = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius2)
    # Create the sphere body
    sphere_id2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_collision_id2,
                                  baseVisualShapeIndex=sphere_visual_id2)
    # Set the initial position and orientation of the sphere
    initial_position2 = pos[min_distance_index, 1:]  # X, Y, Z coordinates of the desired position
    initial_orientation2 = [0, 0, 0, 1]  # Quaternion orientation (no rotation)
    p.resetBasePositionAndOrientation(sphere_id2, initial_position2, initial_orientation2)
    p.changeVisualShape(sphere_id2, -1, rgbaColor=color2)




    ##################################

    # Enable the real-time simulation
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81)
    control_dt = Fixed_time/len(pos)
    control_dt = 0.02
    if filename == '12.ulg':
        control_dt = Fixed_time/len(pos)
    p.setTimestep = control_dt
    state_t = 0.
    i = 0


    if counter == 0:
        time.sleep(5)
        lineColor = [1, 0, 0]  # red color for the line
        lineWidth = 6  # width of the line
        counter = 1

    else:
        lineColor = [0, 1, 0]  # red color for the line
        lineWidth = 3.5  # width of the line



    tmp = 0
    text_id = p.addUserDebugText("Window", window_position + np.array([0, 0, 2]), textColorRGB=[0.55, 0.55, 0.55], textSize=0.5)

    while i < len(pos):
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        for joint in range(1, 5):
            p.setJointMotorControl2(iris_id, joint, p.POSITION_CONTROL, 100)

        pos_iris = pos[i, 1:]
        ori_iris = q_states[i, 1:]

        p.resetBasePositionAndOrientation(iris_id, pos_iris, ori_iris)

        if i > 0:
            p.addUserDebugLine(pos[i-1, 1:], pos[i, 1:], lineColor, lineWidth)

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
                           textColorRGB=[0, 0, 0], textSize=2)

    if tmp == 0:
        if text_id2 is not None:
            p.removeUserDebugItem(text_id2)
        text_id2 = p.addUserDebugText(f"Collision Counter: {counter_collision}     Nsim: {n+1}", window_position + np.array([0, 0, 1]),
                           textColorRGB=[0, 0, 0], textSize=2)


    n = n + 1
time.sleep(10)
p.disconnect()  # Disconnect from the simulation




