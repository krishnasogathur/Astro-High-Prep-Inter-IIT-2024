Refer to presentation.pdf to get a general overview of work done. 

altx2bund includes demo version of our XRF calculations. Optimized weights are printed out in the terminal (set Xset.chatter and Xset.logChatter = False to turn it off in define_xrf_localmodel.py).

Our novelties include 

FASTER.py -> computes primary and secondary XRF contributions using tensorized integrals
get_constants_xrf_new_V2.py -> element-wise cross sections of photoelectric attenuation, total attenuation at characteristic line energies, and total attenuation at all energies.
rayleigh_compute.py -> Scattering integral with the assumption of coherent scattering off a matrix of elements on Lunar surface.

To run the code, change response_path to pwd/test in define_xrf_localmodel.py. 
