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

This file contains the function xrf_comp that computes the actual XRF line intensity from each element.
'''

from common_modules import *
import numpy as np
from scipy.interpolate import interp1d

def xrf_comp(energy: list, counts, i_angle, e_angle, at_no: list, weight: list, xrf_lines:Xrf_Lines, const_xrf:Const_Xrf) -> Xrf_Struc:
    no_elements = n_elements(at_no)
    totalweight = np.sum(weight)
    # ;Just to ensure that weight is converted to weight fractions
    weight = weight/totalweight
    tmp1 = xrf_lines.edgeenergy
    # ; To get the number of lines for each element which is to be computed
    n_lines = np.shape(tmp1)[1]
    n_ebins = n_elements(energy)
    binsize = energy[1] - energy[0]
    
    primary_xrf = dblarr(no_elements, n_lines)
    secondary_xrf = dblarr(no_elements, n_lines)
    secondary_xrf_linewise = dblarr(no_elements, n_lines, no_elements, n_lines)
    


    for i in range(0, no_elements):
        for j in range(0, n_lines):
            fluoryield = xrf_lines.fluoryield[i, j]
            radrate = xrf_lines.radrate[i, j]
            lineenergy = xrf_lines.lineenergy[i, j]
            
            #; Computing the probability coming from the jump ratios
            element_jumpfactor = xrf_lines.jumpfactor[i, :]
            element_edgeenergy = xrf_lines.edgeenergy[i, :]
            ratio_jump = dblarr(n_ebins)
            if (j <= 1):#; Jump ratio probability for K-transitions
                tmp2 = np.where(energy >= element_edgeenergy[j])
                ratio_jump[tmp2] = 1.0 - 1.0/element_jumpfactor[j]
            else:#; Jump ratio probability for L and higher transitions
                tmp3 = np.where(energy > element_edgeenergy[1])
                ratio_jump[tmp3] = 1.0/PRODUCT(element_jumpfactor[1:j])*(1.-1.0/element_jumpfactor[j])
                for k in range(2, j+1):
                    tmp4 = np.where((energy < element_edgeenergy[k-1]) & (energy > element_edgeenergy[k]))
                    if (n_elements(tmp4)!=0):
                        if (k != j):
                            ratio_jump[tmp4] = 1.0/PRODUCT(element_jumpfactor[k:j])*(1.-1.0/element_jumpfactor[j])
                        else:
                            ratio_jump[tmp4] = (1.-1.0/element_jumpfactor[j])
                            
            if((lineenergy > 0) and (radrate > 0)):
                
                #; Computing primary xrf
                musample_eincident = const_xrf.total_attn[i, j, :]
                musample_echarline = const_xrf.total_attn_at_line[i, j]
                muelement_eincident = const_xrf.photoelec_attn[i, j, :]

                pxrf_denom = musample_eincident*(1.0/np.sin(i_angle * np.pi/180)) + musample_echarline*(1.0/np.sin(e_angle * np.pi/180))
                pxrf_Q = weight[i]*muelement_eincident*fluoryield*radrate*ratio_jump
    
                primary_xrf[i, j] = (1.0/np.sin(i_angle * np.pi/180))*total((pxrf_Q*counts*binsize)/(pxrf_denom))
                
                #; Computing secondary xrf(i.e secondary enhancement in other lines due to this line)
                secondaries_index_2D = np.where(xrf_lines.edgeenergy < lineenergy)
                secondaries_index_2D = np.array(secondaries_index_2D)
                n_secondaries = (np.shape(secondaries_index_2D))[1]
                for k in range(0, n_secondaries):
                    i_secondary = secondaries_index_2D[0,k]
                    j_secondary = secondaries_index_2D[1,k]
                    
                    fluoryield_secondary = xrf_lines.fluoryield[i_secondary, j_secondary]
                    radrate_secondary = xrf_lines.radrate[i_secondary, j_secondary]
                    lineenergy_secondary = xrf_lines.lineenergy[i_secondary, j_secondary]
                    
                    #; Computing the probability coming from the jump ratios for secondaries
                    element_jumpfactor_secondary = xrf_lines.jumpfactor[i_secondary, :]
                    if (j_secondary <= 1):#; Jump ratio probability for K-transitions
                        ratio_jump_secondary = 1.0 - 1.0/element_jumpfactor[j_secondary]
                    else:#; Jump ratio probability for L and higher transitions
                        ratio_jump_secondary = 1.0/PRODUCT(element_jumpfactor_secondary[1:j_secondary])*(1.-1.0/element_jumpfactor_secondary[j_secondary])
                        
                    if((lineenergy_secondary > 0) and (radrate_secondary > 0)):
                        musample_echarline_secondary = const_xrf.total_attn_at_line[i_secondary, j_secondary]
                        # muelement_eincident_secondary = const_xrf.total_attn[i_secondary, j_secondary, :]
                        muelement_eincident_secondary = const_xrf.photoelec_attn[i_secondary, j_secondary, :]
                        x_interp = energy
                        y_interp = muelement_eincident_secondary
                        func_interp = interp1d(x_interp, y_interp, fill_value='extrapolate')
                        muelement_pline_secondary = func_interp(lineenergy)
                        
                        L = 0.5*((((np.sin(i_angle * np.pi/180))/(musample_eincident))*np.log(1+(musample_eincident)/(np.sin(i_angle * np.pi/180)*musample_echarline))) + (((np.sin(e_angle * np.pi/180))/(musample_echarline_secondary))*np.log(1+(musample_echarline_secondary)/(np.sin(e_angle * np.pi/180)*musample_echarline))))
                        zero_index = np.where(musample_eincident == 0)#; This is to avoid places of division with 0
                        if (n_elements(zero_index) != 0):
                            L[zero_index] = 0
                            
                        sxrf_denom = musample_eincident*(1.0/np.sin(i_angle * np.pi/180)) + musample_echarline_secondary*(1.0/np.sin(e_angle * np.pi/180))
                        sxrf_Q = weight[i_secondary]*muelement_pline_secondary*fluoryield_secondary*radrate_secondary*ratio_jump_secondary
                        # print(sxrf_Q, 'a')
                        
                        secondary_xrf_linewise[i, j, i_secondary, j_secondary] = (1.0/np.sin(i_angle * np.pi/180))*total((counts*pxrf_Q*sxrf_Q*L*binsize)/(sxrf_denom))
                        if (secondary_xrf_linewise[i, j, i_secondary, j_secondary] > 0):
                            secondary_xrf[i_secondary, j_secondary] = secondary_xrf[i_secondary, j_secondary] + secondary_xrf_linewise[i, j, i_secondary, j_secondary]
                            
    total_xrf = primary_xrf + secondary_xrf
    return Xrf_Struc(primary_xrf, secondary_xrf, total_xrf)

