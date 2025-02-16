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
save_path = '/home/tesla/Desktop/Sensitivity_Exp/sensitivity_cls/sens/log/save/' + now.strftime(
    '%Y-%m-%d')
if not os.path.isdir(save_path):
    os.mkdir(save_path)

Fixed_height = 1
index_w = 0.5   #  should be between [0, 1]

# choose the type here only
typo = 'sim'
# point_w = np.array([0.7, -0.3, 1.8])
# Fixed_time = 4.99 # In seconds

# add +2 and inverse x and y and then put negative sign on x

point_w = np.array([2, 0, 2.1])
Fixed_time = 6.99 #In seconds



output = "/home/tesla/Desktop/Sensitivity_Exp/sensitivity_cls/sens/log/out"
TEST_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

messages = ["vehicle_local_position", "vehicle_local_position_setpoint", "actuator_motors"]
messages_str = ','.join(messages)
delimiter = ','
time_s = 0
time_e = 0

pos_keys = ['timestamp', 'x', 'y', 'z']
act_keys = ['timestamp', 'control[0]', 'control[1]', 'control[2]', 'control[3]']


dir_name = ["In_init/", "In_pi/"]
name = ["INIT", "PI"]
figure_1 = plt.figure(figsize=(16, 12))
figure_1.subplots_adjust(left=0.13, bottom=0.12, right=0.915, top=0.93, wspace=0.02, hspace=0.12)

figure_2 = plt.figure(figsize=(16, 12))
figure_2.subplots_adjust(left=0.13, bottom=0.12, right=0.915, top=0.93, wspace=0.02, hspace=0.2)


for k in range(len(dir_name)):

    directory_path = os.path.join(TEST_PATH, dir_name[k])

    ulg_files = [filename for filename in os.listdir(directory_path) if filename.endswith('.ulg')]
    sorted_ulg_files = sorted(ulg_files, key=extract_number)

    axis_1 = figure_1.add_subplot(1, 2, k + 1, projection="3d")
    axis_2 = figure_2.add_subplot(2, 1, k + 1)

    list_d = []
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

        pos_ref[:, 1], pos_ref[:, 2] = pos_ref[:, 2].copy() , pos_ref[:, 1].copy()
        if typo == 'sim':
            pos_ref[:, 1] *= -1
            pos_ref[:, 1:3] += -2
            pos_ref[:, 1], pos_ref[:, 2] = pos_ref[:, 2].copy(), pos_ref[:, 1].copy()


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
        pos[:, 1], pos[:, 2] = pos[:, 2].copy(), pos[:, 1].copy()

        if typo == 'sim':
            pos[:, 1] *= -1
            pos[:, 1:3] -= 2
            pos[:, 1], pos[:, 2] = pos[:, 2].copy(), pos[:, 1].copy()

        # index_window = int(index_f/2)
        index_window = int(index_w * index_f)



        # Open the CSV file
        with open('out/'+ os.path.splitext(filename)[0] +'_actuator_motors_0.csv', 'r') as file:
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
        u_motor[:, 0] /=1e6

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



        # Add a common ylabel for the column

        axis_2.axhline(y=1, linestyle='--', color='r', linewidth=4)
        axis_2.axvline(x=Fixed_time / 2, linestyle='--', color='k', linewidth=2)
        axis_2.set_ylim(0.4, 1.1)
        axis_2.yaxis.grid()
        axis_2.xaxis.grid()

        if filename == "1.ulg": # for nominal case plotting
            axis_1.plot(pos_ref[index_window, 1], pos_ref[index_window, 2], pos_ref[index_window, 3], "o", color="b", linewidth=5, markersize=15, alpha=0.6, label="Desired Target: $r_d(t_w)$")
            axis_1.set_title(name[k], fontsize=c.fontsizetitle+4)
            axis_2.set_title(name[k], fontsize=c.fontsizetitle+4)
            axis_1.plot3D(pos_ref[:index_f, 1], pos_ref[:index_f, 2], pos_ref[:index_f, 3], 'k--', linewidth=2, label='Reference Trajectory')
            axis_1.plot3D(pos[:, 1], pos[:, 2], pos[:, 3], 'r', linewidth=1.5, label="Nominal Controller tracking")
            axis_2.plot(u_motor[:-1, 0], u_motor[:-1, 2], 'r', linewidth=2.5,  label=r'$p = p_c$ Nominal actuator')
        else:
            axis_1.plot3D(pos[:, 1], pos[:, 2], pos[:, 3], color='gray', linewidth=1, label=r'$p \neq p_{c}$ Controller tracking with pertubation')
            axis_2.plot(u_motor[:-1, 0], u_motor[:-1, 2], color='gray', linewidth=1.5, label=r'$p \neq p_{c}$ Perturbed actuator ')
            if k == 1 and counter == 0:
                axis_1.legend(loc='upper center', fancybox=True, ncol=2, bbox_to_anchor=(0, 0.1), fontsize=30)
                axis_2.set_xlabel(r'$t\ [\text{s}]$', fontsize=c.fontsizelabel)
                axis_2.axhline(y=1, linestyle='--', color='r', linewidth=4, label='bound maximum speed')
                axis_2.legend(loc='upper center', fancybox=True, ncol=3, bbox_to_anchor=(0.5, 0.2), fontsize=24)
                counter = 1

        # plot Closest point and plot as well the starting point
        axis_1.plot(pos[min_distance_index, 1], pos[min_distance_index, 2], pos[min_distance_index, 3], "o", color="r", linewidth=3.5, markersize=3, alpha=0.9, label='closest positions to target')
        axis_1.plot(*pos_ref[0, 1:], "o", color="r", linewidth=3.5, markersize=8)


        axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
        axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
        axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)


        #
        # axis_1.view_init(elev=30, azim=180)
        # axis_1.set_xlim(-4, 4)
        # axis_1.set_ylim(-4, 4)
        # axis_1.set_zlim(0, 2)

        # axis_1.view_init(elev=50, azim=200)
        # axis_1.set_xlim(-4, 4)
        # axis_1.set_ylim(0, 4)
        # axis_1.set_zlim(0, 2.5)

        # axis_1.view_init(elev=50, azim=200)
        # axis_1.set_xlim(-2, 2)
        # axis_1.set_ylim(-2, 2)
        # axis_1.set_zlim(0, 2.5)

        ########Figure of the inputs



    if k == 0:
        d1 = np.array(list_d)
    else:
        d2 = np.array(list_d)


figure_2.text(0.06, 0.5, r'$u_3(t)$  Normalized', fontsize=c.fontsizelabel + 20, va='center', rotation='vertical')
figure_1.savefig(save_path + '/tracking_all__' + '.pdf')
figure_2.savefig(save_path + '/inputs_all__' + '.pdf')


fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharey=True)

d = np.vstack((d1, d2))

df_targets = pd.DataFrame(d.T)
df_targets.columns = name
sns.violinplot(data=df_targets, order=["INIT", "PI"], linewidth=4, ax=ax)
ax.set_ylabel(r'Target reach in [\text{m}]', fontsize=c.fontsizetitle)
ax.set_title(r'PX4 Controller', fontsize=c.fontsizetitle)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.yaxis.grid()

ax.set_yticks(np.arange(0, 0.4, 0.1))

plt.savefig(save_path + '/violin_plot_'+ '.pdf')

plt.show()
print("Finished")