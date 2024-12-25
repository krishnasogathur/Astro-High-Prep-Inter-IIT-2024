#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:33:43 2022

@author: netra

Test script to check if x2abund is working with all the updates
"""

# Importing necessary modules
from xspec import *
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages

# Specifying the input files
class_l1_data = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits'
bkg_file = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits'
scatter_atable = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/tbmodel_20210827T210316000_20210827T210332000.fits'
solar_model_file = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/modelop_20210827T210316000_20210827T210332000.txt'
response_path = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/'

static_par_file = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/static_par_localmodel.txt'
xspec_log_file = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/log_x2abund_test.txt'
xspec_xcm_file = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/xcm_x2abund_test.xcm'
plot_file = '/home/bond007/Desktop/ch2_class_x2abund_lmodel_v1.0/X2ABUND_LMODEL_V1/test/plots_x2abund_test.pdf'

ignore_erange = ["0.9","4.2"]
ignore_string = '0.0-' + ignore_erange[0] + ' ' + ignore_erange[1] + '-**'

# Getting the information for making the static parameter file
hdu_data = fits.open(class_l1_data)
hdu_header = hdu_data[1].header
hdu_data.close()

solar_zenith_angle = hdu_header['SOLARANG']
emiss_angle = hdu_header['EMISNANG']
sat_alt = hdu_header['SAT_ALT']
tint = hdu_header['EXPOSURE']

fid_statpar = open(static_par_file,'w')
fid_statpar.write(solar_model_file + '\n')
fid_statpar.write(str(solar_zenith_angle) + '\n')
fid_statpar.write(str(emiss_angle) + '\n')
fid_statpar.write(str(sat_alt) + '\n')
fid_statpar.write(str(tint) + '\n')
fid_statpar.close()

# PyXspec Initialisation
Xset.openLog(xspec_log_file)
AllData.clear()
AllModels.clear()

os.chdir(response_path)
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
mo(6).link = '100 - (3+4+5+7+8+9+10)'

Fit.nIterations = 100
Fit.perform()

# Plotting the fit outputs
pdf_plot = PdfPages(plot_file)

data_energy_tmp = spec_data.energies
data_countspersec = spec_data.values
data_background = spec_data.background.values
data_backrem = np.array(data_countspersec) - np.array(data_background)

data_energy = np.zeros(np.size(data_backrem))
for k in range(0,np.size(data_energy)):
    data_energy[k] = (data_energy_tmp[k])[0]
    
folded_flux = mo.folded(1)
delchi = (data_backrem - folded_flux)/np.sqrt(folded_flux)

fig, (axis1, axis2) = plt.subplots(2, 1, gridspec_kw={'width_ratios':[1], 'height_ratios':[3,1]})
fig.suptitle('Data Model Comparison')

axis1.plot(data_energy,data_backrem)
axis1.plot(data_energy,folded_flux)
axis1.set_yscale("log")
        
axis1.set_xlabel('Energy (keV)')
axis1.set_ylabel('Counts/s')
axis1.set_xlim([float(ignore_erange[0]),float(ignore_erange[1])])
axis1.legend(['Data','Model'])
    
axis2.plot(data_energy,delchi)
axis2.set_xlabel('Energy (keV)')
axis2.set_ylabel('Delchi')
axis2.set_xlim([float(ignore_erange[0]),float(ignore_erange[1])])

pdf_plot.savefig(fig,bbox_inches='tight',dpi=300)
plt.close(fig)

pdf_plot.close()

# Closing PyXspec
Xset.save(xspec_xcm_file)
Xset.closeLog()
