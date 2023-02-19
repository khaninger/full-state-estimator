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
source urdf/devel/setup.bash
roslaunch racer_description racer.launch (or) roslaunch ur_description load_ur16e.launch
