Project dedicated for sensiitivity experiments.
Robust trajectory generation based on the PX4 controller for Acanthis drone below.

![Acanthis](Acanthis_description.pdf)

## Getting Started

- **Installation** <br>

Use the following sudo commands

```
sudo apt install clang
sudo apt-get install libomp-dev

```

- **Make Directory and clone Repo** <br>

```
mkdir sens_exp && cd sens_exp
git clone https://gitlab.inria.fr/asrour/sensitivity_exp.git

```

- **Configure a virtual enviroment OR skip this if you prefer to install global on machine** <br>

In the same directory that you cloned the project, you can create and source the virtual environment by:

```
cd sensitivity_exp
python3 -m venv env
source env/bin/activate


```
- **Install dependencies in virtual environment** <br>

```
pip install -r requiremnts.txt

```
- **Build the framework as a package** <br>

The `-e` is essential as this would make it possible to modify the package without building it upon each modification-

```
pip install -e .

```
- **sens package** <br>
You can view the package by

```
pip list

```

- **Run the Following script that generates the ODE problem** <br>
This script should generate the model with all sensitivity related stuff

```
python3 -m sens.gen.models.jetson_pd

```



## Framework Overview

After installing the sensitivity framework as the `sens` package, you are ready now to start discovering the framework.
Mainly the sens package consists of 5 folders "directories" which are:

- **cnst** <br>
    - Contains a python file `constant.py` which includes a class with all constants to be used in the framework
- **gen** <br>
    - Contains Folder `lib` for symbolic generation (No need to edit scripts in this directory)
        - a script `base_model.py` defines an abstract class of the model and functions declarations
        - a script `sym_gen.py` which contains a class called Jitparam based on Jitcode for generating symbolic model `.so` file
    
    - Contains a Folder `models` where you have to define your ODE problem, model, controller and sensitivity
        - a script `jetson_pd.py` is the script containing all these definitions and responsible of generating the model depending on jtiparam class in `sym_gen.py`
- **opt** <br>
    - a script `optimize.py` where different functions implemented for different optimization problems depending on Nonlinear optimizar from `nlopt` package in python
- **script** <br>
    - This folder contains several scripts that you can run as examples and they are documented very well
- **utils** <br>
    - `Functions.py` defines some symbolic functions used in `jetson_pd.py` and also other functions like evaluation of the tubes on states.
    - `trajectory.py` contains a class named `PiecewiseSplineTrajectory` where we define the trajectory and do all the planning and pre optimization depending on this class


## Learn by examples
Inside `sens.script` you would find several examples that you need to run to explore the package
These examples/turorials are for the sake of using the framework:
- [How to optimize trajectories?](https://gitlab.inria.fr/asrour/sensitivity_exp/-/wikis/Optimization)

