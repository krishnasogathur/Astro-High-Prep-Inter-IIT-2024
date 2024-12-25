import numpy as np
from common_modules import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from memo import memoize_last
from scipy.interpolate import interp1d
import time

'''

This program is the faster alternative to xrf_comp_new_v2.py. It computes primary and secondary XRF integrals in a fast-manner using 
memoized computation of jump ratios and combinations of tensorized computations between integral and element-wise sums.


'''


denom = []
num = []
tot = []
@memoize_last
def compute_ratio_jumpall_primary(no_elements, n_lines, n_ebins, energy, xrf_lines ):
    # secondary_xrf_linewise = np.zeros((no_elements, n_lines, no_elements, n_lines))
    ratio_jumpall = np.zeros((no_elements, n_lines, n_ebins))
 
    # start_time = time.time()
    for i in range(no_elements):
        # for j in range(1,n_lines):
        for j in range(n_lines):
            if (j <= 1):#; Jump ratio probability for K-transitions
                tmp2 = np.where(energy >= xrf_lines.edgeenergy[i, j])
                ratio_jumpall[i,j,tmp2] = 1.0 - 1.0/xrf_lines.jumpfactor[i, j]
            else:#; Jump ratio probability for L and higher transitions
                tmp3 = np.where(energy > xrf_lines.edgeenergy[i, 1])
                ratio_jumpall[i,j,tmp3] = 1.0/PRODUCT(xrf_lines.jumpfactor[i, 1:j])*(1.-1.0/xrf_lines.jumpfactor[i, j])
                for k in range(2, j+1):
                    tmp4 = np.where((energy < xrf_lines.edgeenergy[i, k-1]) & (energy > xrf_lines.edgeenergy[i, k]))
                    if (n_elements(tmp4)!=0):
                        if (k != j):
                            ratio_jumpall[i,j,tmp4] = 1.0/PRODUCT(xrf_lines.jumpfactor[i, k:j])*(1.-1.0/xrf_lines.jumpfactor[i, j])
                        else:
                            ratio_jumpall[i,j,tmp4] = (1.-1.0/xrf_lines.jumpfactor[i, j])


            # # Compute ratio_jump for primary XRF
            # if j <= 1:  # K-transitions
            #     ratio_jumpall[i,j,energy >= xrf_lines.edgeenergy[i,j]] = 1.0 - 1.0 / xrf_lines.jumpfactor[i,j]
            # else:  # L and higher transitions
            #     for k in range(2, j + 1):
            #         mask = (energy < xrf_lines.edgeenergy[i,k - 1]) & (energy > xrf_lines.edgeenergy[i,k])
            #         ratio_jumpall[i,j,mask] = 1.0 / np.prod(xrf_lines.jumpfactor[i, k:j]) * (1.0 - 1.0 / xrf_lines.jumpfactor[i,j])
            # Compute ratio_jump for primary XRF
            # 
            # TEST IT!!
            # ratio_jumpall[i,j,energy >= xrf_lines.edgeenergy[i,j]] = 1.0 - inv_jumpfactor[i,j]
            # for k in range(2, j + 1):
            #     mask = (energy < xrf_lines.edgeenergy[i,k - 1]) & (energy > xrf_lines.edgeenergy[i,k])
            #     ratio_jumpall[i ,j,mask]*= np.prod(inv_jumpfactor[i, k:j])

    return ratio_jumpall

@memoize_last
def compute_ratio_jumpall_secondary(no_elements, n_lines, xrf_lines ):
    ratio_jump_secondary_all = np.zeros((no_elements, n_lines, no_elements, n_lines))
 
    # start_time = time.time()
    for i in range(no_elements):
        # for j in range(1,n_lines):
        for j in range(n_lines):
            lineenergy = xrf_lines.lineenergy[i, j]

            secondaries_index_2D = np.where(xrf_lines.edgeenergy < lineenergy)
            secondaries_index_2D = np.array(secondaries_index_2D)
            n_secondaries = (np.shape(secondaries_index_2D))[1]
            for k in range(0, n_secondaries):
                i_secondary = secondaries_index_2D[0,k]
                j_secondary = secondaries_index_2D[1,k]
            
                #; Computing the probability coming from the jump ratios for secondaries
                element_jumpfactor_secondary = xrf_lines.jumpfactor[i_secondary, :]
                if (j_secondary <= 1):#; Jump ratio probability for K-transitions
                    ratio_jump_secondary_all[i,j, i_secondary, j_secondary] = 1.0 - 1.0/xrf_lines.jumpfactor[i_secondary, j_secondary]    #we replaced i with i_secondary
                else:#; Jump ratio probability for L and higher transitions
                    ratio_jump_secondary_all[i,j, i_secondary, j_secondary] = 1.0/np.prod(element_jumpfactor_secondary[1:j_secondary])*(1.-1.0/element_jumpfactor_secondary[j_secondary])
                # ratio_jump_secondary = ratio_jumpall[i_secondary, j_secondary]
                


    return ratio_jump_secondary_all


@memoize_last
def partial_den(musample_eincident, musample_echarline, i_angle, e_angle):
    return  (musample_echarline * (np.sin(np.radians(i_angle))))# np.sin(np.radians(e_angle))))

@memoize_last
def partial_num(xrf_lines, ratio_jumpall, muelement_eincident):
    return xrf_lines.fluoryield[:,:,np.newaxis] * xrf_lines.radrate[:,:,np.newaxis] * ratio_jumpall *muelement_eincident 



### COOLNEW: OUR FASTER TENSORIZED IMPLEMENTATION OF INTEGRAL CALCULATIONS

def xrf_comp_new_coolnew_backup(energy, counts, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf):

    weightexp = weight[np.newaxis,np.newaxis,:,np.newaxis]
    no_elements = len(at_no)
    n_lines = xrf_lines.edgeenergy.shape[1]
    n_ebins = len(energy)
    binsize = energy[1] - energy[0]

    
    ratio_jumpall = compute_ratio_jumpall_primary(no_elements, n_lines, n_ebins, energy, xrf_lines )
              
    musample_eincident = const_xrf.total_attn #0 each (8,1,3000,1)
    musample_eincident_mask = const_xrf.total_attn_mask
    musample_echarline = const_xrf.total_attn_at_line #2 (8,5,8)
    muelement_eincident = const_xrf.photoelec_attn
    pxrf_denom= partial_den(musample_eincident, musample_echarline, i_angle, e_angle)
    # print((weightexp*musample_eincident).shape, musample_eincident_mask.shape, ((weightexp*musample_eincident)*musample_eincident_mask).shape)
    # exit(0)
    pxrf_denom = np.sum((weightexp*musample_eincident), axis = 2)*musample_eincident_mask.squeeze(axis=2) +   np.sum(weightexp * pxrf_denom, axis = 2)
 
    ###########


    pxrf_Qall = weightexp.squeeze(axis=(0,1,3)) [:,np.newaxis,np.newaxis]* partial_num(xrf_lines, ratio_jumpall, muelement_eincident)

    primary_xrf = (binsize * np.sum((pxrf_Qall * counts) / pxrf_denom, axis = 2))
    primary_xrf = np.nan_to_num(primary_xrf, nan=0.0)


    musample_eincident_all = (np.sum(weightexp * musample_eincident, axis = 2) * musample_eincident_mask).squeeze(axis = 2)
 
    musample_echarline_all = np.sum(weightexp * musample_echarline, axis = 2).squeeze(axis = 2)[:,:,np.newaxis]



    secondary_xrf_linewise = np.zeros((no_elements, n_lines, no_elements, n_lines))
    secondary_xrf = np.zeros((no_elements, n_lines))


    ### BELOW CODE COMPUTES SECONDARY XRF LINES IN A CUMBERSOME MANNER; WE DEVELOPED A FASTER VERSION AKIN TO PRIMARY IN THE ACTUAL VERSION OF THE CODE. 

    # for i in range(0, no_elements):
    #     for j in range(0, n_lines):
    #         pxrf_Q = pxrf_Qall[i,j,:]
    #         musample_eincident  = musample_eincident_all[i,j]
    #         musample_echarline  = musample_echarline_all[i,j]

            
    #         if((xrf_lines.lineenergy[i, j] > 0) and (xrf_lines.radrate[i, j] > 0)):
                    
    #             #; Computing secondary xrf(i.e secondary enhancement in other lines due to this line)
    #             secondaries_index_2D = np.where(xrf_lines.edgeenergy < xrf_lines.lineenergy[i,j])
    #             secondaries_index_2D = np.array(secondaries_index_2D)
    #             n_secondaries = (np.shape(secondaries_index_2D))[1]
    #             for k in range(0, n_secondaries):
    #                 i_secondary = secondaries_index_2D[0,k]
    #                 j_secondary = secondaries_index_2D[1,k]
                    
    #                 fluoryield_secondary = xrf_lines.fluoryield[i_secondary, j_secondary]
    #                 radrate_secondary = xrf_lines.radrate[i_secondary, j_secondary]
    #                 lineenergy_secondary = xrf_lines.lineenergy[i_secondary, j_secondary]
                    
    #                 #; Computing the probability coming from the jump ratios for secondaries
    #                 element_jumpfactor_secondary = xrf_lines.jumpfactor[i_secondary, :]
    #                 if (j_secondary <= 1):#; Jump ratio probability for K-transitions
    #                     ratio_jump_secondary = 1.0 - 1.0/xrf_lines.jumpfactor[i, j_secondary]
    #                 else:#; Jump ratio probability for L and higher transitions
    #                     ratio_jump_secondary = 1.0/np.prod(element_jumpfactor_secondary[1:j_secondary])*(1.-1.0/element_jumpfactor_secondary[j_secondary])
                        
    #                 if((lineenergy_secondary > 0) and (radrate_secondary > 0)):

    #                     #pull out interpolator generator
    #                     musample_echarline_secondary = musample_echarline_all[i_secondary, j_secondary]
    #                     muelement_eincident_secondary = const_xrf.photoelec_attn[i_secondary, j_secondary, :]
                        
    #                     x_interp = energy
    #                     y_interp = muelement_eincident_secondary
    #                     func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')
    #                     muelement_pline_secondary = func_interp(xrf_lines.lineenergy[i, j])

                        
    #                     L = 0.5*((((np.sin(i_angle * np.pi/180))/(musample_eincident))*np.log(1+(musample_eincident)/(np.sin(i_angle * np.pi/180)*musample_echarline))) + (((np.sin(e_angle * np.pi/180))/(musample_echarline_secondary))*np.log(1+(musample_echarline_secondary)/(np.sin(e_angle * np.pi/180)*musample_echarline))))
    #                     zero_index = np.where(musample_eincident == 0)#; This is to avoid places of division with 0
    #                     # print(L.shape)
    #                     if (n_elements(zero_index) != 0):
    #                         L[zero_index] = 0
                            
    #                     sxrf_denom = musample_eincident*(1.0/np.sin(i_angle * np.pi/180)) + musample_echarline_secondary*(1.0/np.sin(e_angle * np.pi/180))
    #                     sxrf_Q = weight[i_secondary]*muelement_pline_secondary*fluoryield_secondary*radrate_secondary*ratio_jump_secondary
    #                     # print(sxrf_Q, "b")
    #                     secondary_xrf_linewise[i, j, i_secondary, j_secondary] = (1.0/np.sin(i_angle * np.pi/180))*total((counts*pxrf_Q*sxrf_Q*L*binsize)/(sxrf_denom))
    #                     if (secondary_xrf_linewise[i, j, i_secondary, j_secondary] > 0):
    #                         secondary_xrf[i_secondary, j_secondary] = secondary_xrf[i_secondary, j_secondary] + secondary_xrf_linewise[i, j, i_secondary, j_secondary]
                        

    return Xrf_Struc(primary_xrf, secondary_xrf, primary_xrf+secondary_xrf)


@memoize_last
def f(no_elements, n_lines, xrf_lines, const_xrf, energy , musample_echarline):
        #Keep in mind: each mu here is not added over weights yet
    secondaries_index_mask = np.zeros((no_elements, n_lines,no_elements, n_lines))
    musample_echarline_all_secondary = np.zeros((no_elements,n_lines, no_elements, no_elements,n_lines))   # 8x5x1x8x5
    muelement_pline_secondary_all = np.zeros((no_elements,n_lines, no_elements,n_lines))   


    for i in range(0, no_elements):
        for j in range(0, n_lines):
            secondaries_index_mask[i,j, :, :] = xrf_lines.edgeenergy < xrf_lines.lineenergy[i,j] #first ij still primary, second two are for secondary
            secondaries_index = np.where(secondaries_index_mask[i,j,:,:]>0)
            secondaries_index = np.array(secondaries_index)
            n_secondaries = (np.shape(secondaries_index))[1]
                
            for k in range(0, n_secondaries): #can remove this loop
                i_secondary = secondaries_index[0,k]
                j_secondary = secondaries_index[1,k]            
                # musample_echarline_secondary = musample_echarline_all[i_secondary, j_secondary]
                muelement_eincident_secondary = const_xrf.photoelec_attn[i_secondary, j_secondary,:] #weight, energy array
                musample_echarline_all_secondary[i,j,:,i_secondary,j_secondary] = musample_echarline[i_secondary, j_secondary].squeeze(axis=1)
                x_interp = energy
                y_interp = muelement_eincident_secondary
                func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')
                muelement_pline_secondary_all[i,j,i_secondary,j_secondary] = func_interp(xrf_lines.lineenergy[i,j]) # 8x5x8x5
    # musample_echarline_all_secondary = musample_echarline[:,:,:, np.newaxis] * secondaries_index_mask[:,:,np.newaxis,:,:]


    fluoryield_secondary = xrf_lines.fluoryield[:,:,np.newaxis, np.newaxis]
    radrate_secondary =    xrf_lines.radrate   [:,:,np.newaxis, np.newaxis] * secondaries_index_mask
    lineenergy_secondary = xrf_lines.lineenergy[:,:,np.newaxis, np.newaxis] * secondaries_index_mask

    positive_mask = ((xrf_lines.lineenergy > 0) & (xrf_lines.radrate > 0) )[:,:,np.newaxis, np.newaxis]
    positive_mask_secondary = (lineenergy_secondary > 0) & (radrate_secondary>0)
    
    fluoryield_radrate_secondary = (fluoryield_secondary*positive_mask)*radrate_secondary*positive_mask_secondary

    ratio_jump_secondary = compute_ratio_jumpall_secondary(no_elements, n_lines, xrf_lines)
    # return secondaries_index_mask, musample_echarline_all_secondary, muelement_pline_secondary_all
    partial_secondary_num = muelement_pline_secondary_all*fluoryield_radrate_secondary*ratio_jump_secondary

    return positive_mask_secondary, partial_secondary_num, musample_echarline_all_secondary, secondaries_index_mask

@memoize_last
def g(i_angle, e_angle, musample_eincident, musample_echarline_all_secondary):

    musample_eincident = musample_eincident / (np.sin(i_angle * np.pi/180)) 
    musample_echarline_all_secondary = musample_echarline_all_secondary / (np.sin(e_angle * np.pi/180)) 
    return musample_eincident, musample_echarline_all_secondary
    
