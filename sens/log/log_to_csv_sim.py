import inspect
import csv
import pyulog.ulog2csv as ulog2csv
import tempfile
import os
import numpy as np
from cnst.constant import constants
import matplotlib.pyplot as plt

#TODO: choose the duration and the hieght you start the trajectory
Fixed_height = 1
Fixed_time = 6.99 # In seconds

window_time = Fixed_time/2

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



output = "/home/tesla/Desktop/Sensitivity_Exp/sensitivity_cls/sens/log/out"

TEST_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
directory_path = os.path.join(TEST_PATH, 'In/')


messages = ["vehicle_local_position", "vehicle_local_position_setpoint"]
messages_str = ','.join(messages)
delimiter = ','
time_s = 0
time_e = 0

pos_keys = ['timestamp', 'x', 'y', 'z']

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
    pos_ref[:, 2:4] *= -1
    # pos_ref[:, -1] *= -1
    pos_ref[:, 0] /= 1e6

    # Find the indices where 'z' changes from 1
    change_indices = np.where(np.diff(pos_ref[:, 3]) != 0)[0]
    # Find the index where 'z' first changes from 1 and consider indices after that
    start_index = change_indices[np.argmax(pos_ref[change_indices, -1] != Fixed_height)] - 1


    pos_ref = pos_ref[start_index:-1, :]

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
    pos[:, 2:4] *= -1
    # pos[:, -1] *= -1
    index_low = np.argmax(pos[:, 0] >= time_start)
    index_high = np.argmax(pos[:, 0] >= time_finish)

    pos = pos[index_low:index_high, :]



    axis_1.plot3D(pos_ref[:index_f, 1], pos_ref[:index_f, 2], pos_ref[:index_f, 3], 'k-', linewidth=3)
    axis_1.plot3D(pos[:, 1], pos[:, 2], pos[:, 3], 'r-', linewidth=0.8)
    axis_1.plot(*pos_ref[0, 1:], "o", color="b", linewidth=3.5, markersize=8)
    if counter ==0:
        counter +=1
        axis_1.plot(*pos_ref[int(index_f/2), 1:], "*", color="b", linewidth=3.5, markersize=24)
    axis_1.set_title(r'Gazebo', fontsize=c.fontsizetitle)
    axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
    axis_1.view_init(elev=30, azim=120)
    axis_1.set_xlim(0, 5)
    axis_1.set_ylim(0, 5)
    axis_1.set_zlim(0, 2.2)


plt.show()
print("Finished")