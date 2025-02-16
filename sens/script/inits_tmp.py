import time
from utils.trajectory import PiecewiseSplineTrajectory
import matplotlib.pyplot as plt
import pickle
from base64 import b64encode
import numpy as np
from cnst.constant import constants

c =constants()

#######################################################################################3
###saving trajectories
traj_filename = "../Paper_Journal_ICRA/save/trajectories/px4_init_tmp_1m_" + str(c.N_traj) +"_" + "Tf_"+ str(c.Tf) +'.straj'

with open(traj_filename, "wb") as dump:
    pass

###########################################

# Initialize waypoints
point1 = c.init_waypoint

# The window
point2 = np.vstack([
    [1, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0.0],
])



t1 = time.time()
wp = [point1, point2]
wp_t = np.array([0, c.Tf])

traj = PiecewiseSplineTrajectory(wp_t, wp)



with open(traj_filename, "ab") as dump:
    dump.write(b"\n".join(map(lambda x: b64encode(pickle.dumps(x)), [traj])))
    dump.write(b"\n" + b64encode(pickle.dumps(wp[1])))
    dump.write(b"\n")
plt.pause(1e-6)

t_end = time.time()

print(f'########################################')
