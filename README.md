# MOPLAonCPP
This is a C++ code to implement the Multi-Order Power Law Approach(MOPLA) for modelling multi-scale fabrics in Earth's lithosphere. The approach involves solving general Eshelby inclusion formalism for anisotropic viscous materials along with the homogenization(Jiang 2014). 

The code simulates the deformation of a number of ellipsoidal element in a medium.  

The final state of the system is calculated after a number of deformation steps.

# Experiment And Results

The original goal of this project was to test if a faster version than the available Matlab based package(Qu et al 2016) could be fully developed in C++. 

However our findings revealed that Matlab along with certain bottleneck operations written in C (implemented in Qu et al 2016) still provides a faster and easy to use version for the academic community. Please refer to [this] (https://github.com/ankibues/MOPLA_Application_Matlab) repository for further implementations. 
