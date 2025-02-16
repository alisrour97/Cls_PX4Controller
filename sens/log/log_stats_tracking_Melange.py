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
import datetime
from utils.Functions import extract_number
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
save_path = '/home/tesla/Desktop/Sensitivity_Exp/sensitivity_cls/sens/log/All/save/' + now.strftime(
    '%Y-%m-%d')
if not os.path.isdir(save_path):
    os.mkdir(save_path)

Fixed_height = 1
index_w = 0.5   #  should be between [0, 1]





output = "/home/tesla/Desktop/Sensitivity_Exp/sensitivity_cls/sens/log/All/out"
TEST_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/All'

messages = ["vehicle_local_position", "vehicle_local_position_setpoint", "actuator_motors"]
messages_str = ','.join(messages)
delimiter = ','
time_s = 0
time_e = 0

pos_keys = ['timestamp', 'x', 'y', 'z']
dir_name = ["In_init_t1_s/", "In_pi_t1_s", "In_init_t1_e/", "In_pi_t1_e", "In_init_t2_s/", "In_pi_t2_s", "In_init_t2_e/", "In_pi_t2_e"]
name = ["$INIT$", "$OPT_a$", "$INIT$", "$OPT_a$", "$INIT$", "$OPT_a$", "$INIT$", "$OPT_a$"]
figure_1 = plt.figure(figsize=(16, 8))
figure_1.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.9, wspace=0.01, hspace=0.1)

typo = 'name' # should be sim or exp

for k in range(len(dir_name)):

    if k < 4:
        point_w = np.array([2, 0, 2.1])
        Fixed_time = 6.99  # In seconds
    else:
        point_w = np.array([0.7, -0.3, 1.8])
        Fixed_time = 4.99 # In seconds

    if k < 2:
        typo ='sim'
    elif k > 3 and k < 6:
        typo = 'sim'
    else:
        typo = 'exp'


    directory_path = os.path.join(TEST_PATH, dir_name[k])

    ulg_files = [filename for filename in os.listdir(directory_path) if filename.endswith('.ulg')]
    sorted_ulg_files = sorted(ulg_files, key=extract_number)

    axis_1 = figure_1.add_subplot(2, 4, k + 1, projection="3d")

    counter = 0
    # Iterate over all files in the directory
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
        with open('All/out/'+ os.path.splitext(filename)[0] +'_vehicle_local_position_setpoint_0.csv', 'r') as file:
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
        if typo == 'sim':
            print(1)
            pos_ref[:, 1] *= -1
            pos_ref[:, 1:3] += -2
            pos_ref[:, 1], pos_ref[:, 2] = pos_ref[:, 2].copy(), pos_ref[:, 1].copy()


        # Open the CSV file
        with open('All/out/'+ os.path.splitext(filename)[0] +'_vehicle_local_position_0.csv', 'r') as file:
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
        pos[:, -1] *= -1

        index_low = np.argmax(pos[:, 0] >= time_start)
        index_high = np.argmax(pos[:, 0] >= time_finish)

        pos = pos[index_low:index_high, :]
        pos[:, 1], pos[:, 2] = pos[:, 2].copy(), pos[:, 1].copy()

        if typo == 'sim':
            pos[:, 1] *= -1
            pos[:, 1:3] -= 2
            pos[:, 1], pos[:, 2] = pos[:, 2].copy(), pos[:, 1].copy()

        index_window = int(index_w * index_f)

        #########################Function Spatial Distance ###################

        min_distance = float('inf')
        min_distance_index = None
        for l in range(len(pos)):
            dist = np.linalg.norm(point_w - pos[l, 1:])

            if dist <min_distance:
                min_distance = dist
                min_distance_index = l

        print("Minimum distance:", min_distance)
        print("Time of minimum distance:", min_distance_index)
        #####################################################################


        if filename == "1.ulg": # for nominal case plotting
            axis_1.plot(pos_ref[index_window, 1], pos_ref[index_window, 2], pos_ref[index_window, 3], "o", color="b", linewidth=5, markersize=15, alpha=0.4, label="Desired Target: $r_d(t_w)$")
            axis_1.plot3D(pos_ref[:index_f, 1], pos_ref[:index_f, 2], pos_ref[:index_f, 3], 'k--', linewidth=2.2, label='Reference Trajectory')
            axis_1.plot3D(pos[:, 1], pos[:, 2], pos[:, 3], 'r', linewidth=2, label="Nominal Controller tracking")
            if k < 4:
                axis_1.set_title(name[k], fontsize=c.fontsizetitle/2 + 4)

        else:
            axis_1.plot3D(pos[:, 1], pos[:, 2], pos[:, 3], color='green', linestyle='--', linewidth=2, alpha=0.45, label=r'$p \neq p_{c}$ Controller tracking with pertubation')
            if k == 5 and counter == 0:
                axis_1.legend(loc='center', fancybox=True, ncol=3, bbox_to_anchor=(0.95, -0.25), fontsize=16)
                counter = 1

        # plot Closest point and plot as well the starting point
        axis_1.plot(pos[min_distance_index, 1], pos[min_distance_index, 2], pos[min_distance_index, 3], "o", color="r", linewidth=3.5, markersize=3, alpha=1, label='Closest positions to target')
        axis_1.plot(*pos_ref[0, 1:], "o", color="k", linewidth=3.5, markersize=8)


        axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel/2 +2)
        axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel/2 +2)
        axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel/2 +2)
        axis_1.tick_params(axis='x', labelsize=c.fontsizelabel/2 + 2)
        axis_1.tick_params(axis='y', labelsize=c.fontsizelabel / 2 + 2)
        axis_1.tick_params(axis='z', labelsize=c.fontsizelabel / 2 + 2)


        if k == 0:
            axis_1.annotate(r'Traj1', xy=(0.2, 0.2), xycoords="axes fraction", xytext=(-80, 80),
                        textcoords="offset points",
                        ha="center", va="center", fontsize=c.fontsizetitle/2 + 4)

            axis_1.annotate(r'Gazebo', xy=(1.55, 0.75), xycoords="axes fraction", xytext=(-80, 80),
                        textcoords="offset points",
                        ha="center", va="center", fontsize=c.fontsizetitle/2 + 4)

            axis_1.annotate(r'Traj2', xy=(0.2, -1), xycoords="axes fraction", xytext=(-80, 80),
                        textcoords="offset points",
                        ha="center", va="center", fontsize=c.fontsizetitle/2 + 4)


            axis_1.annotate(r'Experiment', xy=(4.1, 0.75), xycoords="axes fraction", xytext=(-80, 80),
                        textcoords="offset points",
                        ha="center", va="center", fontsize=c.fontsizetitle/2 + 4)




        axis_1.grid(False)

        if k < 4:
            axis_1.view_init(elev=50, azim=270)
            axis_1.set_xlim(-2.1, 2.4)
            axis_1.set_ylim(-2.1, 2.1)
            axis_1.set_zlim(0, 2.25)
        else:
            axis_1.view_init(elev=30, azim=180)
            axis_1.set_xlim(-2.5, 2.5)
            axis_1.set_ylim(-2.5, 2.5)
            axis_1.set_zlim(0, 2)


        ########Figure of the inputs


figure_1.savefig(save_path + '/tracking_all__'+ '.pdf')
plt.show()
print("Finished")