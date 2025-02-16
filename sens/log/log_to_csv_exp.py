import inspect
import csv
import pyulog.ulog2csv as ulog2csv
import tempfile
import os
import numpy as np
from cnst.constant import constants
import matplotlib.pyplot as plt

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



figure_1 = plt.figure(figsize=(16, 9))
axis_1 = figure_1.add_subplot(1, 1, 1, projection="3d")

Fixed_height = 1
index_w = 0.5   #  should be between [0, 1]

#
# point_w = np.array([-0.3, 0.7, 1.8])
#inversing x and y ; adding +2 to x and y ; inverse y by negative sign
point_w = np.array([2.7, -1.7, 1.8])
Fixed_time = 4.99 # In seconds

# point_w = np.array([0, 2, 2.1])
# point_w = np.array([4, -2, 2.1])
# Fixed_time = 6.99 #In seconds

name = "PI"


output = "/home/tesla/Desktop/Sensitivity_Exp/sensitivity_cls/sens/log/out"

TEST_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
directory_path = os.path.join(TEST_PATH, 'In/')


messages = ["vehicle_local_position", "vehicle_local_position_setpoint", "actuator_motors"]
messages_str = ','.join(messages)
delimiter = ','
time_s = 0
time_e = 0

pos_keys = ['timestamp', 'x', 'y', 'z']
act_keys = ['timestamp', 'control[0]', 'control[1]', 'control[2]', 'control[3]']
list_d = []
counter = 0
# Iterate over all files in the directory
for filename in os.listdir(directory_path):
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
    pos_ref = pos_ref[first_valid_index:last_valid_index+1, :]
    # pos_ref[:, 2:4] *= -1
    pos_ref[:, -1] *= -1
    # pos_ref[:, -2] *= -1
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

    # Open the CSV file
    with open('out/'+ os.path.splitext(filename)[0] +'_vehicle_local_position_0.csv', 'r') as file:
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

    pos[:, 0] /=1e6
    # pos[:, 2:4] *= -1
    pos[:, -1] *= -1
    # pos[:, -2] *= -1
    index_low = np.argmax(pos[:, 0] >= time_start)
    index_high = np.argmax(pos[:, 0] >= time_finish)

    pos = pos[index_low:index_high, :]

    # index_window = int(index_f/2)
    index_window = int(index_w * index_f)

    # Open the CSV file
    with open('out/' + os.path.splitext(filename)[0] + '_actuator_motors_0.csv', 'r') as file:
        reader = csv.DictReader(file)
        # Get the fieldnames from the CSV file
        fieldnames = reader.fieldnames
        # Find the indices of the keys in the fieldnames
        key_indices = [fieldnames.index(key) for key in act_keys]

        # Initialize lists to store the time and values
        u_motor = [[] for _ in key_indices]
        # Iterate over each row in the CSV file
        for row in reader:
            # Get the values for the keys
            for i, index in enumerate(key_indices):
                key_value = float(row[fieldnames[index]])
                u_motor[i].append(key_value)

    u_motor = np.array(u_motor)
    u_motor = u_motor.T
    u_motor[:, 0] /= 1e6

    index_low_u = np.argmax(u_motor[:, 0] >= time_start)
    index_high_u = np.argmax(u_motor[:, 0] >= time_finish)

    u_motor = u_motor[index_low_u:index_high_u, :]
    u_motor[:, 0] -= u_motor[0, 0]


    #########################Function Spatial Distance ###################


    min_distance = float('inf')
    min_distance_index = None

    for l in range(len(pos)):
        dist = np.linalg.norm(point_w - pos[l, 1:])

        if dist <min_distance:
            min_distance = dist
            min_distance_index = l

    list_d.append(min_distance)
    print("Minimum distance:", min_distance)
    print("Time of minimum distance:", min_distance_index)
    #####################################################################
    # list_d.append(np.linalg.norm(pos[index_window, 1:] - point_w))

    axis_1.plot3D(pos_ref[:index_f, 2], pos_ref[:index_f, 1], pos_ref[:index_f, 3], 'k-', linewidth=3)
    axis_1.plot3D(pos[:, 2], pos[:, 1], pos[:, 3], 'r-', linewidth=1)
    axis_1.plot(pos[min_distance_index, 2], pos[min_distance_index, 1], pos[min_distance_index, 3], "x", color="b", linewidth=3.5, markersize=5)
    axis_1.plot(*pos_ref[0, 1:], "o", color="b", linewidth=3.5, markersize=8)
    if counter == 0:
        axis_1.plot(pos_ref[index_window, 2], pos_ref[index_window, 1], pos_ref[index_window, 3], "*", color="b", linewidth=4.5, markersize=15)
        counter = 1
    axis_1.set_title(r'Real Experiments' + name, fontsize=c.fontsizetitle)

    axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
    # axis_1.view_init(elev=30, azim=120)
    # axis_1.set_xlim(-2.5, 2.5)
    # axis_1.set_ylim(-2.5, 2.5)
    # axis_1.set_zlim(0, 2.5)

    axis_1.set_xlim(-4, 0)
    axis_1.set_ylim(0, 4)
    axis_1.set_zlim(0, 2.5)

list_d = np.array(list_d)
print(list_d)
fig, ax = plt.subplots(1, 1, figsize=(9, 5), sharey=True)


df_targets = pd.DataFrame(list_d)
df_targets.columns = [name]
sns.violinplot(data=df_targets, linewidth=4, ax=ax)
ax.set_ylabel(r'Window reach in [\text{m}]', fontsize=c.fontsizetitle)
ax.set_title(r'PX4 Controller', fontsize=c.fontsizetitle)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.yaxis.grid()

ax.set_yticks(np.arange(0, 0.5, 0.1))

plt.show()
print("Finished")