# Full State Estimator

Full state estimator for robots.

# Known dependencies

Pinocchio [https://stack-of-tasks.github.io/pinocchio/download.html]

Python installs, should be available in pip3
- casadi
- casadi-kin-dyn

# Building Pinocchio from source
To get the casadi bindings natively (i.e. skipping the casadi-kin-dyn library), we need to compile Pinocchio from source, from the branch pinocchio3-preview

https://stack-of-tasks.github.io/pinocchio/download.html

You'll also need to grab the FindCASADI.cmake file from Casadi, put it in the cmake folder, include it in the CMakeLists.txt, and rename it to Findcasadi.cmake.  

# Starting
We first set up ROS and get the robot model loaded, 
`source urdf/devel/setup.bash`
`roscore &`
`roslaunch ur_description load_ur16e.launch`

For the optimization of params, an offline trajectory is needed, which is stored in a .pkl file of the same name as the bag it's generated from.
`python3 main.py --new_traj --bag [bagname]`

Then we can optimize the parameters.
`python3 main.py --opt_param --bag [bagname]`

The optimized parameters are set as rosparams, which will be loaded when an EKF is carried out.

You can then do an offline EKF pass by `python3 main.py --new_traj --bag [bagname]`, optionally giving `--est_geom` or `est_stiff` to estimate these parameters online. If you'd like ot run the filter online, leave out the `--new_traj`
