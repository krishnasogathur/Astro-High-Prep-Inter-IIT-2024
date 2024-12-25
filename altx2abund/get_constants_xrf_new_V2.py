'''
;====================================================================================================
;                              X2ABUNDANCE
;
; Package for determining elemental weight percentages from XRF line fluxes
;
; Algorithm developed by P. S. Athiray (Athiray et al. 2015)
; Codes in IDL written by Netra S Pillai
; Codes for XSPEC localmodel developed by Ashish Jacob Sam and Netra S Pillai
;
; Developed at Space Astronomy Group, U.R.Rao Satellite Centre, Indian Space Research Organisation
;
;====================================================================================================

This file contains the function get_constants_xrf that interpolates the cross-sections from the database to the input energy axis and also takes into account inter-element effects
'''

from common_modules import *
from scipy.interpolate import interp1d
import time
from memo import *

'''

This is part of the modified version of original x2abundance implementation. 
This program computes 3 necessary constants (all memoized to be one-time computations): 

element-wise photoelectric attenuation cross-sections, total attenuation cross-sections at characteristic line energies, and total attenuation cross-sections at all energies.

These are further to be used in computations of primary and secondary integrals.   


'''

@memoize_last
def get_interpolation_values(no_elements, n_lines, xrf_lines):
    interpolation_funcs = []
    for k in range(no_elements):
        nonzero_indices = xrf_lines.energy_nist[k, :] != 0
        x_interp = xrf_lines.energy_nist[k, nonzero_indices]
        y_interp = xrf_lines.totalcs_nist[k, nonzero_indices]
        interpolation_funcs.append(interp1d(x_interp, y_interp, fill_value="extrapolate"))

    interpolated_values = np.zeros((no_elements, n_lines, no_elements))
    for i in range(no_elements):
        for j in range(n_lines):
            line_energy = xrf_lines.lineenergy[i, j]
            rad_rate = xrf_lines.radrate[i,j]
            if line_energy > 0 and rad_rate > 0:
                # Vectorize calculation over all elements
                interpolated_values[i, j, :] = np.array([interpolation_funcs[k](line_energy) for k in range(no_elements)])
                # total_attn_at_line[i, j] = np.sum(weight * interpolated_values, axis=2)
    return interpolated_values

@memoize_last
def get_conds_musampleeincident(no_elements, n_lines, n_ebins, xrf_lines, energy):
    condn1 = (xrf_lines.lineenergy > 0) & (xrf_lines.radrate > 0) #creating a boolean mask essentially
    musample_eincident = np.zeros((no_elements, n_ebins))
    # musample_eincident_xspec = np.zeros((no_elements, n_ebins_xspec))
    for k in range(0,no_elements):
        tmp3 = np.where(xrf_lines.energy_nist[k,:] != 0)
        x_interp = (xrf_lines.energy_nist[k,tmp3])[0,:]
        y_interp = (xrf_lines.totalcs_nist[k,tmp3])[0,:]
        func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')
        musample_eincident[k, :] = func_interp(energy)
        # musample_eincident_xspec[k,:] = func_interp(energy_xspec)


    # Masking function : one time
    condn2 = np.ones([no_elements, n_lines, n_ebins])
    # condn2_xspec = np.ones([no_elements, n_lines, n_ebins_xspec])
    for i in range(no_elements):
        for j in range(n_lines):
            edge_energy = xrf_lines.edgeenergy[i,j]
            tmp4 = np.where(energy < edge_energy)
            if (np.size(tmp4) != 0):
                condn2[i,j,tmp4] = 0.0
            # tmp4_xspec = np.where(energy_xspec < edge_energy)
            # if (np.size(tmp4) != 0):
                # condn2_xspec[i,j,tmp4_xspec] = 0.0
    
    return condn1, condn2, musample_eincident

@memoize_last
def get_photoelec_attn(no_elements, n_lines, n_ebins, xrf_lines, energy):
    photoelec_attn = np.zeros([no_elements,n_lines,n_ebins])
    # Loop through elements, no need for the second loop over 'j' anymore
    for i in range(0, no_elements):
        # Extract the line energy, radiation rate, and edge energy for this element
        line_energy = xrf_lines.lineenergy[i, :]
        rad_rate = xrf_lines.radrate[i, :]
        edge_energy = xrf_lines.edgeenergy[i, :]
        
        # Mask condition for valid XRF lines (line_energy > 0 and rad_rate > 0)
        valid_lines = (line_energy > 0) & (rad_rate > 0)

        # Create the interpolation function based on non-zero energy values
        tmp3 = np.where(xrf_lines.energy_nist[i, :] != 0)
        x_interp = xrf_lines.energy_nist[i, tmp3][0, :]
        y_interp = xrf_lines.photoncs_nist[i, tmp3][0, :]
        func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')

        # Perform the interpolation for all energy bins
        muelement_eincident = func_interp(energy)
        
        # Apply the valid_lines mask correctly
        # valid_lines is 1D, so we need to apply it across all energy bins
        photoelec_attn[i, valid_lines, :] = muelement_eincident[np.newaxis, np.newaxis, :]

        # Create a mask for energy < edge_energy (for all j)
        energy_mask = energy < edge_energy[:, np.newaxis]

        # Apply the mask to set values below the edge energy to zero
        photoelec_attn[i, energy_mask] = 0.0
    return photoelec_attn
    
def get_constants_xrf(energy:list, at_no:list,weight:list,xrf_lines:Xrf_Lines) -> Const_Xrf:
    
    #; Function to compute the different cross sections necessary for computing XRF lines

	#Original: weight = weight/total(weight) ; To confirm that weight fractions are taken
    no_elements = np.size(at_no)
    n_ebins = np.size(energy)
    
    #; Identify the number of lines for which xrf computation is done - just by checking array sizes of xrf_lines
    tmp2 = xrf_lines.edgeenergy
    n_lines = np.shape(tmp2)[1]

    
#; Computing total attenuation of sample at characteristic line energies
    
    interpolated_values = get_interpolation_values(no_elements, n_lines, xrf_lines)
    
    total_attn_at_line = np.sum(weight * interpolated_values, axis=2)

    
    condn1, condn2, musample_eincident = get_conds_musampleeincident(no_elements, n_lines, n_ebins, xrf_lines, energy)
    
    # print(np.shape(weight) , np.shape(weight.T) , np.shape(musample_eincident) , np.shape(musample_eincident.T), "\n\n\n\n\n\n\n\n")
    # exit(0)   

    total_attn_scalar =  np.sum(musample_eincident * weight[:,np.newaxis], axis = 0)

    # total_attn_scalar_xspec = musample_eincident_xspec.T @ weight
    # total_attn_scalar_xspec = total_attn_scalar_xspec.flatten()

    # st = time.time()

    # Assumption 1: Rayleigh integral involves neither condn1 nor condn2 

    total_attn = total_attn_scalar[np.newaxis, np.newaxis, :]  # Shape: (1, 1, n_ebins)
    # total_attn = np.tile(total_attn, (no_elements, n_lines, 1))  # Shape: (no_elements, n_lines, n_ebins)

    # For Rayleigh scattering mu term we
    # total_attn_xspec = total_attn_scalar_xspec[np.newaxis,np.newaxis,:]  # Shape: (1, 1, n_ebins) 
    # total_attn_xspec = np.tile(total_attn_xspec, (no_elements,n_lines, 1))  # Shape: (no_elements, n_ebins_xspec)
 
    # Applying the mask
    total_attn = total_attn * condn1[:,:,np.newaxis] *condn2

    # total_attn_xspec = total_attn_xspec * condn1[:,:,np.newaxis] *condn2_xspec
     
     
    # IGNORE WHAT'S COMMENTED BELOW: I WROTE SOME BULLSHIT

    ''' Assumption 2: Rayleigh integral involves condn1 but not condn2:

    total_attn_rayleigh = total_attn_scalar[np.newaxis, np.newaxis, :]  # Shape: (1, 1, n_ebins)
    total_attn_rayleigh = np.tile(total_attn_rayleigh, (no_elements, n_lines, 1)) * condn1[:,:,np.newaxis]  # Shape: (no_elements, n_lines, n_ebins)


    total_attn_xspec_rayleigh = total_attn_scalar_xspec[np.newaxis, :]  # Shape: (1, 1, n_ebins)
    total_attn_xspec_rayleigh = np.tile(total_attn_xspec_rayleigh, (no_elements, n_lines, 1)) * condn1[:,:,np.newaxis]  # Shape: (no_elements, n_lines, n_ebins_xspec)
 
    # Applying the mask
    total_attn = total_attn_rayleigh *condn2
    total_attn_xspec = total_attn_xspec_rayleigh *condn2_xspec
     
    # I think Assumption 1 makes more sense

      '''

    # et = time.time()

    # print(et - st , '\n\n\n\n\n')
    # exit(0)
    photoelec_attn = get_photoelec_attn(no_elements, n_lines, n_ebins, xrf_lines, energy)
    
    return Const_Xrf(total_attn_at_line, total_attn , photoelec_attn, 0)


### COOLNEW: OUR FASTER TENSORIZED IMPLEMENTATION OF INTEGRAL CALCULATIONS

@memoize_last
def get_constants_xrfcoolnew(energy:list,  at_no:list, xrf_lines:Xrf_Lines) -> Const_Xrf:

    no_elements = np.size(at_no)
    n_ebins = np.size(energy)
    
    tmp2 = xrf_lines.edgeenergy
    n_lines = np.shape(tmp2)[1]

    interpolated_values = get_interpolation_values(no_elements, n_lines, xrf_lines)
    
    # total_attn_at_line = np.sum(weight * interpolated_values, axis=2)
    total_attn_at_line = interpolated_values[:,:,:,np.newaxis]

    condn1, condn2, musample_eincident = get_conds_musampleeincident(no_elements, n_lines, n_ebins, xrf_lines, energy)
    

    # total_attn_scalar =  np.sum(musample_eincident * weight[:,np.newaxis], axis = 0)
    total_attn_scalar =  musample_eincident

    total_attn = total_attn_scalar[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1,no_elements, n_ebins)

    # total_attn = total_attn * condn1[:,:,np.newaxis, np.newaxis] *condn2[:,:,np.newaxis,:]

    photoelec_attn = get_photoelec_attn(no_elements, n_lines, n_ebins, xrf_lines, energy)
    
    return Const_Xrf(total_attn_at_line, total_attn , photoelec_attn, condn1[:,:,np.newaxis, np.newaxis] *condn2[:,:,np.newaxis,:])