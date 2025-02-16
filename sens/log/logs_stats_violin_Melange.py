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


# figure_1 = plt.figure(figsize=(16, 10))
# figure_1.subplots_adjust(left=0.1, bottom=0.12, right=0.93, top=0.85, wspace=0.015, hspace=0.1)

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


    list_d = []
    counter = 0
    # Iterate over all files in the directory
    for filename in sorted_ulg_files:
        if filename.endswith('.ulg'):  # Check if the file has a .ulg extension
            # Construct the full path of the log file
            log_file_path = os.path.join(directory_path, filename)

            # Perform operations on the log file
            # print("Processing log file:", log_file_path)
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

        list_d.append(min_distance)
        # print("Minimum distance:", min_distance)
        # print("Time of minimum distance:", min_distance_index)
        #####################################################################


    if k == 0:
        d1 = np.array(list_d)
    elif k == 1:
        d2 = np.array(list_d)
    elif k == 2:
        d3 = np.array(list_d)
    elif k == 3:
        d4 = np.array(list_d)
    elif k == 4:
        d5 = np.array(list_d)
    elif k == 5:
        d6= np.array(list_d)
    elif k == 6:
        d7 = np.array(list_d)
    elif k == 7:
        d8 = np.array(list_d)


# name = ["$INIT$", "$OPT_a$"]
#
# fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey="row")
# fig.subplots_adjust(left=0.15, bottom=0.12, right=0.98, top=0.85, wspace=0.015, hspace=0.15)
#
# d_p_1 = np.vstack((d1, d2))
# d_p_2 = np.vstack((d3, d4))
# d_p_3 = np.vstack((d5, d6))
# d_p_4 = np.vstack((d7, d8))
#
# df_targets_1 = pd.DataFrame(d_p_1.T)
# df_targets_1.columns = name
#
# df_targets_2= pd.DataFrame(d_p_2.T)
# df_targets_2.columns = name
#
# df_targets_3 = pd.DataFrame(d_p_3.T)
# df_targets_3.columns = name
#
# df_targets_4 = pd.DataFrame(d_p_4.T)
# df_targets_4.columns = name
#
#
# sns.boxenplot(data=df_targets_1, order=["$INIT$", "$OPT_a$"],  linewidth=4, ax=axes[0, 0])
# axes[0, 0].tick_params(axis='both', which='major', labelsize=30)
# axes[0, 0].yaxis.grid()
# axes[0, 0].set_yticks(np.arange(0, 0.3, 0.1))
#
# sns.boxenplot(data=df_targets_2, order=["$INIT$", "$OPT_a$"],  linewidth=4, ax=axes[0, 1])
# axes[0, 1].tick_params(axis='both', which='major', labelsize=30)
# axes[0, 1].yaxis.grid()
# axes[0, 1].set_yticks(np.arange(0, 0.3, 0.1))
#
# sns.boxenplot(data=df_targets_3, order=["$INIT$", "$OPT_a$"],  linewidth=4, ax=axes[1, 0])
# axes[1, 0].tick_params(axis='both', which='major', labelsize=30)
# axes[1, 0].yaxis.grid()
# axes[1, 0].set_yticks(np.arange(0, 0.5, 0.2))
#
# sns.boxenplot(data=df_targets_4, order=["$INIT$", "$OPT_a$"],  linewidth=4, ax=axes[1, 1])
# axes[1, 1].tick_params(axis='both', which='major', labelsize=30)
# axes[1, 1].yaxis.grid()
# axes[1, 1].set_yticks(np.arange(0, 0.5, 0.2))
#
# axes[0, 0].annotate(r'Traj 1', xy=(-0.05, 0.2), xycoords="axes fraction", xytext=(-80, 80),
#                 textcoords="offset points",
#                 ha="center", va="center", fontsize=c.fontsizetitle)
#
# axes[0, 0].annotate(r'Traj 2', xy=(-0.05, -0.9), xycoords="axes fraction", xytext=(-80, 80),
#                 textcoords="offset points",
#                 ha="center", va="center", fontsize=c.fontsizetitle)
#
#
# axes[0, 0].annotate(r'Gazebo', xy=(0.7, 0.9), xycoords="axes fraction", xytext=(-80, 80),
#                 textcoords="offset points",
#                 ha="center", va="center", fontsize=c.fontsizetitle)
#
# axes[0, 0].annotate(r'Real Experiments', xy=(1.7, 0.9), xycoords="axes fraction", xytext=(-80, 80),
#                 textcoords="offset points",
#                 ha="center", va="center", fontsize=c.fontsizetitle)
#
#
#
#
#
# # Save the figure and show it
# fig.savefig(save_path + '/histogram_all__'+ '.pdf')
# plt.show()
# print("Finished")
#
# name = ["$INIT$", "$OPT_a$"]
# d_p_1 = np.vstack((d1, d2))
#
# df_targets_1 = pd.DataFrame(d_p_1.T)
# df_targets_1.columns = name


# name = ["$INIT$", "$OPT_a$"]
#
#
# num_bins = np.arange(0, 0.75, 0.01)
# # Create DataFrames for the two datasets
# df_d1 = pd.DataFrame({name[0]: d1})
# df_d2 = pd.DataFrame({name[1]: d2})
# # Create DataFrames for the two datasets
# df_d5 = pd.DataFrame({name[0]: d5})
# df_d6 = pd.DataFrame({name[1]: d6})
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# fig.subplots_adjust(left=0.1, bottom=0.19, right=0.98, top=0.87, wspace=0.15, hspace=0.15)
#
# # Plot d1 and d2 superimposed
# ax = axes[0]
# sns.histplot(data=df_d1, x=name[0], bins=num_bins, label=name[0], alpha=0.5, ax=ax)
# sns.histplot(data=df_d2, x=name[1], bins=num_bins, label=name[1], alpha=0.5, ax=ax)
# ax.set_title('Traj1', fontsize=30)
# ax.set_xlabel('')
# ax.set_ylabel('')
# ax.legend()
#
# # Plot d5 and d6 superimposed
# ax = axes[1]
# sns.histplot(data=df_d5, x=name[0], bins=num_bins, label=name[0], alpha=0.5, ax=ax)
# sns.histplot(data=df_d6, x=name[1], bins=num_bins, label=name[1], alpha=0.5, ax=ax)
# ax.set_title('Traj2', fontsize=30)
# ax.set_xlabel('')
# ax.set_ylabel('')
# ax.legend()
#
# # Add common title, x-label, and y-label
# fig.suptitle(r'Gazebo', fontsize=30)
# fig.text(0.5, 0.04, r'Distance to the desired target $r_d(t_w)$', ha='center', fontsize=30)
# fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical', fontsize=30)
#
# # # Save the figure and show it
# # plt.tight_layout()
# # Replace 'save_path' with the path where you want to save the figure
# fig.savefig(save_path + '/histogram_sim'+ '.pdf')
# plt.show()



name = ["$INIT$", "$OPT_a$"]

num_bins = np.arange(0, 0.75, 0.01)
# Create DataFrames for the two datasets
df_d1 = pd.DataFrame({name[0]: d1})
df_d2 = pd.DataFrame({name[1]: d2})
# Create DataFrames for the two datasets
df_d5 = pd.DataFrame({name[0]: d5})
df_d6 = pd.DataFrame({name[1]: d6})

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.subplots_adjust(left=0.1, bottom=0.19, right=0.98, top=0.87, wspace=0.15, hspace=0.15)

# Define colors for the samples
sample_colors = ['blue', 'red']

# Plot d1 and d2 superimposed
ax = axes[0]
for i, df in enumerate([df_d1, df_d2]):
    sns.histplot(data=df, x=name[i], bins=num_bins, label=name[i], alpha=0.5, ax=ax, color=sample_colors[i])
    mean_val = df[name[i]].mean()  # Calculate the mean
    ax.axvline(mean_val, color=sample_colors[i], linewidth=2, linestyle='dashed', label=f'Average: {mean_val:.2f}')
ax.set_title('Traj1', fontsize=30)
ax.set_xlabel('')
ax.set_ylabel('')
ax.legend()

# Plot d5 and d6 superimposed
ax = axes[1]
for i, df in enumerate([df_d5, df_d6]):
    sns.histplot(data=df, x=name[i], bins=num_bins, label=name[i], alpha=0.5, ax=ax, color=sample_colors[i])
    mean_val = df[name[i]].mean()  # Calculate the mean
    ax.axvline(mean_val, color=sample_colors[i], linewidth=2, linestyle='dashed', label=f'Average: {mean_val:.2f}')
ax.set_title('Traj2', fontsize=30)
ax.set_xlabel('')
ax.set_ylabel('')
ax.legend()

# Add common title, x-label, and y-label
fig.suptitle(r'Gazebo', fontsize=30)
fig.text(0.5, 0.04, r'Distance to the desired target', ha='center', fontsize=30)
fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical', fontsize=30)


# Replace 'save_path' with the path where you want to save the figure
fig.savefig(save_path + '/histogram_sim' + '.pdf')
plt.show()










# name = ["$INIT$", "$OPT_a$"]
#
# num_bins = np.arange(0, 0.35, 0.01)
# # Create DataFrames for the two datasets
# df_d1 = pd.DataFrame({name[0]: d3})
# df_d2 = pd.DataFrame({name[1]: d4})
# # Create DataFrames for the two datasets
# df_d5 = pd.DataFrame({name[0]: d7})
# df_d6 = pd.DataFrame({name[1]: d8})
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# fig.subplots_adjust(left=0.1, bottom=0.19, right=0.98, top=0.87, wspace=0.15, hspace=0.15)
#
# # Plot d1 and d2 superimposed
# ax = axes[0]
# sns.histplot(data=df_d1, x=name[0], bins=num_bins, label=name[0], alpha=0.5, ax=ax)
# sns.histplot(data=df_d2, x=name[1], bins=num_bins, label=name[1], alpha=0.5, ax=ax)
# ax.set_title('Traj1', fontsize=30)
# ax.set_xlabel('')
# ax.set_ylabel('')
# ax.legend()
#
# # Plot d5 and d6 superimposed
# ax = axes[1]
# sns.histplot(data=df_d5, x=name[0], bins=num_bins, label=name[0], alpha=0.5, ax=ax)
# sns.histplot(data=df_d6, x=name[1], bins=num_bins, label=name[1], alpha=0.5, ax=ax)
# ax.set_title('Traj2', fontsize=30)
# ax.set_xlabel('')
# ax.set_ylabel('')
# ax.legend()
#
# # Add common title, x-label, and y-label
# fig.suptitle(r'Experiments', fontsize=30)
# fig.text(0.5, 0.04, r'Distance to the desired target $r_d(t_w)$', ha='center', fontsize=30)
# fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical', fontsize=30)
#
# # # Save the figure and show it
# # plt.tight_layout()
# # Replace 'save_path' with the path where you want to save the figure
# fig.savefig(save_path + '/histogram_exp'+ '.pdf')
# plt.show()




name = ["$INIT$", "$OPT_a$"]

num_bins = np.arange(0, 0.35, 0.01)
# Create DataFrames for the two datasets
df_d1 = pd.DataFrame({name[0]: d3})
df_d2 = pd.DataFrame({name[1]: d4})
# Create DataFrames for the two datasets
df_d5 = pd.DataFrame({name[0]: d7})
df_d6 = pd.DataFrame({name[1]: d8})

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.subplots_adjust(left=0.1, bottom=0.19, right=0.98, top=0.87, wspace=0.15, hspace=0.15)

# Define colors for the samples
sample_colors = ['blue', 'red']

# Plot d1 and d2 superimposed
ax = axes[0]
for i, df in enumerate([df_d1, df_d2]):
    sns.histplot(data=df, x=name[i], bins=num_bins, label=name[i], alpha=0.5, ax=ax, color=sample_colors[i])
    mean_val = df[name[i]].mean()  # Calculate the mean
    ax.axvline(mean_val, color=sample_colors[i], linewidth=2, linestyle='dashed', label=f'Average: {mean_val:.2f}')
ax.set_title('Traj1', fontsize=30)
ax.set_xlabel('')
ax.set_ylabel('')
ax.legend()

# Plot d5 and d6 superimposed
ax = axes[1]
for i, df in enumerate([df_d5, df_d6]):
    sns.histplot(data=df, x=name[i], bins=num_bins, label=name[i], alpha=0.5, ax=ax, color=sample_colors[i])
    mean_val = df[name[i]].mean()  # Calculate the mean
    ax.axvline(mean_val, color=sample_colors[i], linewidth=2, linestyle='dashed', label=f'Average: {mean_val:.2f}')
ax.set_title('Traj2', fontsize=30)
ax.set_xlabel('')
ax.set_ylabel('')
ax.legend()

# Add common title, x-label, and y-label
fig.suptitle(r'Experiments', fontsize=30)
fig.text(0.5, 0.04, r'Distance to the desired target', ha='center', fontsize=30)
fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical', fontsize=30)
# Replace 'save_path' with the path where you want to save the figure
fig.savefig(save_path + '/histogram_exp'+ '.pdf')
plt.show()