#################################################
#################################################

#### COMPUTING THE RAYLEIGH SCATTER INTEGRAL ####

#################################################
#################################################

import numpy as np
from scipy.interpolate import interp1d
from common_modules import *
from memo import memoize_last

'''

This program is our approach toward accounting for counts from scattered solar spectrum at the X-Ray SCDs. 
For this calculation, we invoke the scattering integral with the assumption of coherent scattering
because of which the integral becomes a element-wise product, thanks to the mighty Delta function. 

'''



@memoize_last
def get_scattering_cs(no_elements, n_lines, n_ebins, xrf_lines, energy_solar):
    scattering_cs = np.zeros([no_elements,n_lines, n_ebins])
    for k in range(0,no_elements):
        tmp3 = np.where(xrf_lines.energy_nist[k,:] != 0)
        x_interp = (xrf_lines.energy_nist[k,tmp3])[0,:]
        y_interp = (xrf_lines.scatteringcs_nist[k,tmp3])[0,:]
        func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate') 
        scattering_cs[k,:,:] =  func_interp(energy_solar).reshape(1, -1)
    # print(scattering_cs, scattering_cs.shape,"\n\n\n\n\n\n\n\n")
    # exit(0)
    return scattering_cs


def rayleigh_compute(energy_solar , counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf):
    no_elements = len(at_no)
    n_lines = xrf_lines.edgeenergy.shape[1]
    n_ebins = len(energy_solar)
    binsize = energy_solar[1] - energy_solar[0]
    weight = weight / np.sum(weight)
#   inteprolate sigma(E) 
    scattering_cs = get_scattering_cs(no_elements, n_lines, n_ebins, xrf_lines, energy_solar)

   # Convert angles from degrees to radians
    i_angle_rad = np.radians(i_angle)
    e_angle_rad = np.radians(e_angle)
    
    # Calculate the cosecant of the angles (csc(x) = 1/sin(x))
    csc_i = 1 / np.sin(i_angle_rad)
    csc_e = 1 / np.sin(e_angle_rad)

    angle_term = csc_i + csc_e

    total_attn_solar = (const_xrf.total_attn).squeeze(axis = (0,1))[:,np.newaxis,:] 
    mask = (const_xrf.total_attn_mask).squeeze(axis = 2)  # Shape for future reference: (no_elements, n_lines, n_ebins)
    total_attn_solar = total_attn_solar * mask
    total_attn_solar[total_attn_solar == 0] = np.inf
    evaluated_rayleigh = (binsize/angle_term) * ( np.sum(
        (scattering_cs * counts_solar[np.newaxis, np.newaxis, :]) / (total_attn_solar ), axis=1
    )* weight[:, np.newaxis]) .sum(axis = 0)

    return evaluated_rayleigh
