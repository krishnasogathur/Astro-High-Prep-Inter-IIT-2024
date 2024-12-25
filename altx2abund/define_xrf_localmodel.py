
'''
@authors: Krishna Balaji, Aadyot Bhardwaj, Yeswanth Patnana, Aditya Mohapatra, Ganesh Balaji, Vatsal Ramanuj

Updated X2ABUNDACE.

This is the local model developed by IIT Madras, defined for fitting CLASS data using PyXspec. 

solarscatter_localmodel - models contribution from scattering of solar spectrum off of lunar surface
xrf_localmodel - models contribution from X-Ray fluorescence on the lunar surface.

'''
# Importing necessary modules
import numpy as np
from xspec import *
import xraylib
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
# from get_constants_xrf_new_V2 import get_constants_xrf

from get_constants_xrf_new_V2 import get_constants_xrfcoolnew#, get_constants_xrf as get_constants_xrf2
from FASTER import xrf_comp_new_coolnew_backup as xrf_comp_new_coolnew  
from xrf_comp_new_V2 import xrf_comp as xrf_comp_old
import os
import time

#####profiled


total_local_time = 0
start = []
end = []
reader = []
time_get_xrf_0 = []
time_get_xrf_lines_1 = []
time_get_xrf_constants_2 = []
time_get_xrf_compute_3 = []
time_get_xrf_fill_4 = []
st_total = time.time()


matmultime = 0

# Xset.chatter = Xset.logChatter = False
# Getting the static parameters for the local model
static_parameter_file = "static_par_localmodel.txt"
fid = open(static_parameter_file,"r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])

(energy_solar,tmp1_solar,counts_solar) = readcol(solar_file,format='F,F,F')

i_angle = 90.0 - solar_zenith_angle
e_angle = 90.0 - emiss_angle

# Defining some input parameters required for x2abund xrf computation modules
at_no = np.array([26,22,20,14,13,12,11,8])
# Computing the XRF line intensities
k_lines = np.array([xraylib.KL1_LINE, xraylib.KL2_LINE, xraylib.KL3_LINE, xraylib.KM1_LINE, xraylib.KM2_LINE, xraylib.KM3_LINE, xraylib.KM4_LINE, xraylib.KM5_LINE])
l1_lines = np.array([xraylib.L1L2_LINE, xraylib.L1L3_LINE, xraylib.L1M1_LINE, xraylib.L1M2_LINE, xraylib.L1M3_LINE, xraylib.L1M4_LINE, xraylib.L1M5_LINE, xraylib.L1N1_LINE, xraylib.L1N2_LINE, xraylib.L1N3_LINE, xraylib.L1N4_LINE, xraylib.L1N5_LINE, xraylib.L1N6_LINE, xraylib.L1N7_LINE])
l2_lines = np.array([xraylib.L2L3_LINE, xraylib.L2M1_LINE, xraylib.L2M2_LINE, xraylib.L2M3_LINE, xraylib.L2M4_LINE, xraylib.L2M5_LINE, xraylib.L2N1_LINE, xraylib.L2N2_LINE, xraylib.L2N3_LINE, xraylib.L2N4_LINE, xraylib.L2N5_LINE, xraylib.L2N6_LINE, xraylib.L2N7_LINE])
l3_lines = [xraylib.L3M1_LINE, xraylib.L3M2_LINE, xraylib.L3M3_LINE, xraylib.L3M4_LINE, xraylib.L3M5_LINE, xraylib.L3N1_LINE,xraylib.L3N2_LINE, xraylib.L3N3_LINE, xraylib.L3N4_LINE, xraylib.L3N5_LINE, xraylib.L3N6_LINE, xraylib.L3N7_LINE]

z1 = time.time()
xrf_lines = get_xrf_lines(at_no, xraylib.K_SHELL, k_lines, xraylib.L1_SHELL, l1_lines, xraylib.L2_SHELL, l2_lines, xraylib.L3_SHELL, l3_lines)
z2 = time.time()
time_get_xrf_lines_1.append(z2-z1)


const_xrf = get_constants_xrfcoolnew(energy_solar, at_no, xrf_lines)


# Defining the model function
def xrf_localmodel_old(energy, parameters, flux):
    global energy_solar, tmp1_solar, counts_solar, xrf_lines
    
    # Defining proper energy axis
    energy = np.array(energy)
    energy_mid = 0.5 * (energy[:-1] + energy[1:])
    # for i in np.arange(np.size(energy)-1):
    #     energy_mid[i] = 0.5*(energy[i+1] + energy[i])
        

    
    weight = np.array(parameters)[:-1]
    xrf_struc= xrf_comp_new_coolnew(energy_solar,counts_solar,i_angle,e_angle,at_no,weight,xrf_lines,const_xrf)

            # Generating XRF spectrum
    bin_size = energy[1] - energy[0]
    ebin_left = energy_mid - 0.5*bin_size
    ebin_right = energy_mid + 0.5*bin_size
    
    no_elements = (np.shape(xrf_lines.lineenergy))[0]
    n_lines = (np.shape(xrf_lines.lineenergy))[1]
    n_ebins = np.size(energy_mid)

    # Defining the flux array required for XSPEC
    scaling_factor = (12.5*1e4*12.5*(round(exposure/8.0)+1)*1e4)/(exposure*4*np.pi*(altitude*1e4)**2)
    
    spectrum_xrf = np.zeros(n_ebins)
    
    # Flatten the line energies and corresponding intensities for vectorized operations
    line_energies = xrf_lines.lineenergy.ravel()
    xrf_intensities = xrf_struc.total_xrf.ravel()

    # Filter out invalid energy values
    valid_mask = line_energies > 0
    line_energies = line_energies[valid_mask]
    xrf_intensities = xrf_intensities[valid_mask]

    # Determine bin indices for all valid line energies
    bin_indices = np.digitize(line_energies, bins=ebin_left, right=False) - 1

    # Filter out indices outside the spectrum bounds
    valid_bins = (bin_indices >= 0) & (bin_indices < len(spectrum_xrf))
    bin_indices = bin_indices[valid_bins]
    xrf_intensities = xrf_intensities[valid_bins]

    np.add.at(spectrum_xrf, bin_indices, xrf_intensities)
    
    
    ### Defining the flux array required for XSPEC
    spectrum_xrf_scaled = scaling_factor*spectrum_xrf
    for i in range(0, n_ebins):
        flux[i] = spectrum_xrf_scaled[i]

    # ##print(len(flux))
    # ##print("\n\n\n\n")
        

def xrf_localmodel(energy : tuple, parameters : tuple, flux : list) -> None:
    ######profiling
    global total_local_time, start, end, reader, time_get_xrf_lines_1, time_get_xrf_constants_2,time_get_xrf_0, time_get_xrf_compute_3, time_get_xrf_fill_4
    # ##print(np.round(energy,2), len(energy),"\n\n\n\n\n")
    # plt.plot(energy_solar[:300], counts_solar[:300])
    # plt.plot(energy[:300],counts_solar[:300])
    # plt.show()
    # plt.legend(["1", "2"])
    # ##print(energy_solar, "\n\n",np.round(energy,2),"\n\n",counts_solar_new, "\n\n\n\n\n")
    # exit(0)
    z1 = time.time()

    st_local = time.time()
    start.append(st_local)

    z1 = time.time()
    const_xrf = get_constants_xrfcoolnew(energy_solar, at_no, xrf_lines)

    z2 = time.time()
    time_get_xrf_constants_2.append(z2-z1)

    ### convert weight parameters tuple to nparray
    weight = np.array(parameters)
    weight = weight[:-1]
    weight = weight/np.sum(weight)

    # ### Defining proper energy axis
    energy = np.array(energy)
    energy_mid = 0.5 * (energy[:-1] + energy[1:])
    ### Generating XRF spectrum
    bin_size = energy[1] - energy[0]
    ebin_left = energy_mid - 0.5 * bin_size
    # ebin_right = energy_mid + 0.5 * bin_size
    n_ebins = energy_mid.size
    # n_ebins = len(energy)
    z2 = time.time()
    time_get_xrf_0.append(z2-z1)
    

    ### Compute constants and XRF structure
    # z1 = time.time()
    # const_xrf = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)
    # z2 = time.time()
    # time_get_xrf_constants_2.append(z2-z1)

    # flux2 = rayleigh_compute(energy_solar , counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)
    z1 = time.time()
    # xrf_struc= xrf_comp_new_coolnew(energy_solar, counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)

    # const_xrf_old = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)

    # xrf_struc= xrf_comp_old(energy_solar,counts_solar,i_angle,e_angle,at_no,weight,xrf_lines,const_xrf_old)

    xrf_struc= xrf_comp_new_coolnew(energy_solar,counts_solar,i_angle,e_angle,at_no,weight,xrf_lines,const_xrf)

    # assert(np.isclose(xrf_struc.primary_xrf, xrf_struc_old.primary_xrf, 1e-9).all())
    # ##print((xrf_struc.secondary_xrf - xrf_struc_old.secondary_xrf))


    # xrf_struc= parallel_xrf_comp(energy_solar, counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)
    z2 = time.time()
    time_get_xrf_compute_3.append(z2-z1)

    # z1 = time.time()
    # xrf_struc= xrf_comp_new(energy_solar, counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)
    # # xrf_struc= parallel_xrf_comp(energy_solar, counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)
    # z2 = time.time()
    # time_get_xrf_compute_3.append(z2-z1)
    # xrf_struc = xrf_comp_new(energy_solar, counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)

    # result, message = are_attributes_exactly_equal(xrf_struc_old, xrf_struc)
    # ##print(f"Yo bitch: {result}")  # Output: True
    # ##print(message)  # Output: "All attributes are exactly equal."


    z1 = time.time()
    ### Initialize XRF spectrum array
    spectrum_xrf = np.zeros(n_ebins)
    # ### Fill in the bins with XRF lines
    # for i in range(no_elements):
    #     for j in range(n_lines):
    #         line_energy = xrf_lines.lineenergy[i, j]
    #         if line_energy > 0:  # Ensure valid energy value
    #             bin_index = np.where((ebin_left <= line_energy) & (line_energy < ebin_right))[0]
    #             if bin_index.size > 0:
    #                 spectrum_xrf[bin_index] += xrf_struc.total_xrf[i, j]

    # Assuming the following variables are pre-defined:
    # ebin_left, ebin_right: bin boundaries (1D arrays)
    # xrf_lines.lineenergy: 2D array of line energies, shape (no_elements, n_lines)
    # xrf_struc.total_xrf: 2D array of corresponding intensities, shape (no_elements, n_lines)
    # spectrum_xrf: 1D array to store the spectrum, shape matches len(ebin_left) or len(ebin_right) - 1

    # Flatten the line energies and corresponding intensities for vectorized operations
    line_energies = xrf_lines.lineenergy.ravel()
    xrf_intensities = xrf_struc.total_xrf.ravel()

    # Filter out invalid energy values
    valid_mask = line_energies > 0
    line_energies = line_energies[valid_mask]
    xrf_intensities = xrf_intensities[valid_mask]

    # Determine bin indices for all valid line energies
    bin_indices = np.digitize(line_energies, bins=ebin_left, right=False) - 1

    # Filter out indices outside the spectrum bounds
    valid_bins = (bin_indices >= 0) & (bin_indices < len(spectrum_xrf))
    bin_indices = bin_indices[valid_bins]
    xrf_intensities = xrf_intensities[valid_bins]

    # Accumulate intensities into the spectrum using NumPy's bincount
    np.add.at(spectrum_xrf, bin_indices, xrf_intensities)
    
    scaling_factor = (12.5*1e4*12.5*(round(exposure/8.0)+1)*1e4)/(exposure*4*np.pi*(altitude*1e4)**2)

    ### Defining the flux array required for XSPEC
    spectrum_xrf_scaled = scaling_factor * spectrum_xrf
    # ##print((n_ebins),"\n\n", spectrum_xrf_scaled.shape,"xrfstruc\n\n\n")

    ### Assigning scaled spectrum to flux(list object)
    flux[:] = spectrum_xrf_scaled
    # ##print(max(flux),"xrf\n\n\n")

    z2 = time.time()
    time_get_xrf_fill_4.append(z2-z1)

# Specifying parameter information
xrf_localmodel_ParInfo = ("Wt_Fe \"\" 5 1 1 20 20 1e-2","Wt_Ti \"\" 1 1e-6 1e-6 20 20 1e-2","Wt_Ca \"\" 9 5 5 20 20 1e-2","Wt_Si \"\" 21 15 15 35 35 1e-2","Wt_Al \"\" 14 5 5 20 20 1e-2","Wt_Mg \"\" 5 1e-6 1e-6 20 20 1e-2","Wt_Na \"\" 0.5 1e-6 1e-6 5 5 1e-2","Wt_O \"\" 45 30 30 60 60 1e-2")

# Creating the local model in PyXspec
AllModels.addPyMod(xrf_localmodel, xrf_localmodel_ParInfo, 'add')
    


"""

Test script to check if updated X2ABUNDANCE is working as expected.


before starting, change response path to pwd/test.

"""


# Specifying the input files
response_path = '/home/bond007/Desktop/altx2abund/test' #ENTER GLOBAL PATH
os.chdir(response_path)

class_l1_data = 'ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits'
bkg_file = 'ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits'
scatter_atable = 'tbmodel_20210827T210316000_20210827T210332000.fits'
solar_model_file = 'modelop_20210827T210316000_20210827T210332000.txt'

static_par_file = 'static_par_localmodel.txt'
xspec_log_file = 'log_x2abund_test.txt'
xspec_xcm_file = 'xcm_x2abund_test.xcm'
plot_file = 'plots_x2abund_test.pdf'

ignore_erange = ["0.9","4.2"]
ignore_string = '0.0-' + ignore_erange[0] + ' ' + ignore_erange[1] + '-**'

# PyXspec Initialisation
# Xset.openLog(xspec_log_file)
AllData.clear()
AllModels.clear()

spec_data = Spectrum(class_l1_data)
spec_data.background = bkg_file
spec_data.ignore(ignore_string)

# Defining model and fitting
spec_data.response.gain.slope = '1.0043000'
spec_data.response.gain.offset = '0.0316000'
spec_data.response.gain.slope.frozen = True
spec_data.response.gain.offset.frozen = True

full_model = 'atable{' + scatter_atable + '} + xrf_localmodel'


mo = Model(full_model)
mo(10).values = "45.0"
mo(10).frozen = True
mo(1).frozen = True
mo(6).link = '55 - (3+4+5+7+8+9)'

Fit.nIterations = 50
st_perform = time.time()
Fit.perform()
et_perform = time.time()

'''
MANUAL PROFILING CODE BELOW. UNCOMMENT TO CHECK TIME TAKEN FOR EACH COMPUTATION.

'''

##print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
##print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
##print("Time taken for Fit.perform():", et_perform - st_perform)
##print("Total time for xrf_local_model:", total_local_time)
##print("Actual fit time:", et_perform - st_perform - total_local_time)
##print("Total time:", et_perform - st_total)
##print("out of control time",  st_perform-st_total)

# ##print("time differences:",np.array(start[1:]) - np.array(end[:-1]))
# ##print("time reader:", reader)
# ##print("time_get_xrf_lines_1:", time_get_xrf_lines_1)
##print("sum_time_get_xrf_0:", np.sum(time_get_xrf_0))
##print("sum_time_get_xrf_lines_1:", np.sum(time_get_xrf_lines_1))
# ##print("time_get_xrf_constants_2:", time_get_xrf_constants_2)
##print("sum_time_get_xrf_constants_2:", np.sum(time_get_xrf_constants_2))
# ##print("time_get_xrf_compute_3:", time_get_xrf_compute_3)
##print("sum_time_get_xrf_compute_3:", np.sum(time_get_xrf_compute_3))
##print("sum_time_get_xrf_fill_4:", np.sum(time_get_xrf_fill_4))

##print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# ##print("sum_time_rayleigh_compute:", np.sum(time_rayleigh_compute))
##print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")



# # Plotting the fit outputs
# pdf_plot = PdfPages(plot_file)

# data_energy_tmp = spec_data.energies
# data_countspersec = spec_data.values
# data_background = spec_data.background.values
# data_backrem = np.array(data_countspersec) - np.array(data_background)

# data_energy = np.zeros(np.size(data_backrem))
# for k in range(0,np.size(data_energy)):
#     data_energy[k] = (data_energy_tmp[k])[0]
    
# folded_flux = mo.folded(1)
# delchi = (data_backrem - folded_flux)/np.sqrt(folded_flux)

# fig, (axis1, axis2) = plt.subplots(2, 1, gridspec_kw={'width_ratios':[1], 'height_ratios':[3,1]})
# fig.suptitle('Data Model Comparison')

# axis1.plot(data_energy,data_backrem)
# axis1.plot(data_energy,folded_flux)
# axis1.set_yscale("log")
        
# axis1.set_xlabel('Energy (keV)')
# axis1.set_ylabel('Counts/s')
# axis1.set_xlim([float(ignore_erange[0]),float(ignore_erange[1])])
# axis1.legend(['Data','Model'])
    
# axis2.plot(data_energy,delchi)
# axis2.set_xlabel('Energy (keV)')
# axis2.set_ylabel('Delchi')
# axis2.set_xlim([float(ignore_erange[0]),float(ignore_erange[1])])

# pdf_plot.savefig(fig,bbox_inches='tight',dpi=300)
# plt.close(fig)

# pdf_plot.close()

# # Closing PyXspec
# Xset.save(xspec_xcm_file)
# Xset.closeLog()
