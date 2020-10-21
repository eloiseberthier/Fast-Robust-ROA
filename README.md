# Fast-Robust-ROA
This repository contains the source code for the experiments of *Fast Robust Stability Region Estimation for Nonlinear Control Systems*, by Eloïse Berthier, Justin Carpentier and Francis Bach.

This code computes certified regions of attraction around an equilibrium, given:
+ a dynamics f
+ an equilibrium point (x0, u0) such that f(x0, u0)=0,
+ a feedback controller u computed with an LQR around (x0, u0),
+ a function to compute entrywise bounds on the Jacobian or Hessian of f.

## Requirements
The code is written in Python but the first-order certificate calls a Matlab backend engine.
Other requirements include the `pinocchio` ([here](https://github.com/stack-of-tasks/pinocchio)) and `example-robot-data` ([there] (https://github.com/Gepetto/example-robot-data)) libraries, only used for the last example on a robotic arm.
 
## Reproducing the results and plots of the paper
The main results (Table 1 and 2) can be reproduced by running `vanderpol_runner.py`, `satellite_runner.py`, `pendulum_runner.py` (x0 can be changed in `pendulum_config.py` to switch between the bottom of top positions), and `ur5_runner.py`. The results are stored in a csv file in a folder of the same name. Figure 1 is produced by running `Plot-results.ipynb`. Figures 2 and 3 are produced by the script `tv_vanderpol.py`.

## Adding your own dynamics
The main algorithm is defined in `robust_bounds.py`. You can add another dynamics by creating two files:
+ `mydynamics_config.py` defines f, an oracle for the bounds on the Jacobian/Hessian, the parameters of the LQR...
+ `mydynamics_runner.py` chooses which certificate to run and defines some parameters.
If the files follow the same structure as for the existing dynamics, you only need to run `mydynamics_runner.py`.
