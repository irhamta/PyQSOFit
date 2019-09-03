#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:33:32 2019

@author: irhamta
"""

# =============================================================================
# 1. Initializing the input parameters

# Firstly, run the script below to produce the line list file, qsopar.fits,
# containing lines and their constraints, which will be needed in the
# following fitting program. From this file, you can change some specific
# parameters to suit your requirements, e.g., fitting range, line width,
# tie line center, tie line sigma, etc. If you want to fit extra lines,
# please append it to corresponding complex. Note that our line wavelength
# and sigma in the list are in Ln scale, like Lnlambda, Lnsigma.
# =============================================================================

import glob, os,sys,timeit
import matplotlib
import numpy as np
from PyQSOFit_dev import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

path='./j083/'

newdata = np.rec.array([
                        #(2798.75,'MgII',2700.,2900.,'MgII_br',1,5e-3,0.004,0.05,0.0017,0,0,0,0.05),\
                        #(2798.75,'MgII',2700.,2900.,'MgII_na',2,1e-3,5e-4,0.0017,0.01,1,1,0,0.002),\

                        (2798.75,'MgII',2700.,2900.,'MgII_br',1,5e-3,0.004,0.05,0.0017,0,0,0,0.05),\
                        #(2798.75,'MgII',2700.,2900.,'MgII_na',1,1e-3,5e-4,0.0017,0.01,1,1,0,0.002),\

                        #(1549.06,'CIV',1500.,1700.,'CIV_br',1,5e-3,0.004,0.05,0.015,0,0,0,0.05),\
                        #(1549.06,'CIV',1500.,1700.,'CIV_na',1,1e-3,5e-4,0.0017,0.01,1,1,0,0.002),\

                        #(1215.67,'Lya',1150.,1290.,'Lya_br',1,5e-3,0.004,0.05,0.02,0,0,0,0.05),\
                        #(1215.67,'Lya',1150.,1290.,'Lya_na',1,1e-3,5e-4,0.0017,0.01,0,0,0,0.002)\
                        ],\
                     formats='float32,a20,float32,float32,a20,float32,float32,float32,float32,\
                     float32,float32,float32,float32,float32',\
                     names='lambda,compname,minwav,maxwav,linename,ngauss,inisig,minsig,maxsig,voff,vindex,windex,findex,fvalue')
#------header-----------------
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'
hdr['inisig'] = 'Initial guess of linesigma [in lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'
hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'
#------save line info-----------
hdu = fits.BinTableHDU(data=newdata,header=hdr,name='data')
hdu.writeto(path+'qsopar.fits',overwrite=True)


# =============================================================================
# 2. Setup and read the input spectrum

# Setup the paths and read in your spectrum. Our code is written under the
# frame of SDSS spectral data format. Other data is also available as long as
# they include wavelength, flux, error, and redshift, and make sure
# the wavelength resolution is the same as SDSS spectrum
# (For SDSS the pixel scale is 1.e-4 in log space).
# =============================================================================


path1=path                  # the path of the source code file and qsopar.fits
path2=path + 'result/' # path of fitting results
path3=path + 'figures/'   # path of figure
path4=path + 'dustmaps/'             # path of dusp reddening map

#Requried
# an important note that all the data input must be finite, especically for the error !!!

filename = 'spec1d_coadd_j083_tellcorr_proc.csv'
data = pd.read_csv('../temp/' + filename)


lam = data.wavelength        # OBS wavelength [A]
flux = data.flux             # OBS flux [erg/s/cm^2/A]
err = data.flux_err  # 1 sigma error
z = 6.345                # Redshift

# =============================================================================
# somehow we need to normalize the flux to make the correct fit
nconst = 1e-17#flux.max()
err = err/nconst
flux = flux/nconst
# =============================================================================


#Optional
ra, dec = 83.8370655133264, 11.8482345307907

#plateid = data[0].header['plateid']   # SDSS plate ID
#mjd = data[0].header['mjd']           # SDSS MJD
#fiberid = data[0].header['fiberid']   # SDSS fiber ID


# =============================================================================
# 3. Fitting the spectrum with various models

# Use QSOFit to input the lam, flux, err, z, and other optinal parameters.
# Use function Fit to perform the fitting.
# Default settings cannot meet all needs.
# Please change settings for your own requirements.
# It depends on what science you need.
# The following example set dereddening, host decomposition to True and
# do not perform error measurement using Monte Carlo method.
# =============================================================================


# get data prepared
q = QSOFit(lam, flux, err, z, ra = ra, dec = dec, path = path1)

start = timeit.default_timer()

# do the fitting
q.Fit(name = None, nsmooth = 1, and_or_mask = False, deredden = True, reject_badpix = False, wave_range = [1100, 3050],\
      wave_mask =None, decomposition_host = False, Mi = None, npca_gal = 5, npca_qso = 20, \
      Fe_uv_op = True, poly = False, BC = False, rej_abs = False, initial_guess = None, MC = False, \
      n_trails = 5, linefit = True, tie_lambda = True, tie_width = True, tie_flux_1 = True, tie_flux_2 = True,\
      save_result = True, plot_fig = True,save_fig = True, plot_line_name = True, plot_legend = True, \
      dustmap_path = path4, save_fig_path = path3, save_fits_path = path2, save_fits_name = None)

end = timeit.default_timer()
print ('Fitting finished in : '+str(np.round(end-start))+'s')
# grey shade on the top is the continuum windows used to fit.


# =============================================================================
# 4. Show the result

# Get all models for the whole spectrum
# Continue to look at this section and below if you want to do some further
# calculations based on the fitting results.
# Here, we show how to extract different models from our fitting results,
# such as continuum model, emission line models and host galaxy component.
# Note that the emission regions of host galaxy template should be blocked,
# e.g., H$\alpha$ [6540, 6590].
# =============================================================================


fig=plt.figure(figsize=(15,8))
#plot the quasar rest frame spectrum after removed the host galaxy component
plt.plot(q.wave,q.flux,'grey')
plt.plot(q.wave,q.err,'r')


#To plot the whole model, we use Manygauss to reappear the line fitting results saved in gauss_result
plt.plot(q.wave,q.Manygauss(np.log(q.wave),q.gauss_result)+q.f_conti_model,'b',label='line',lw=2)
plt.plot(q.wave,q.f_conti_model,'c',lw=2)
plt.plot(q.wave,q.PL_poly_BC,'orange',lw=2)


plt.xlim(1100,3000)
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)',fontsize = 20)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)',fontsize = 20)


# =============================================================================
# 5. Extract the line parameters
# =============================================================================

fig=plt.figure(figsize=(10,6))
for p in range(int(len(q.gauss_result)/3)):
    if q.CalFWHM(q.gauss_result[3*p+1],q.gauss_result[3*p+2] ) < 1200.:  # < 1200 km/s narrow
        color = 'g'
    else:
        color = 'r'
    plt.plot(q.wave,q.Onegauss(np.log(q.wave),q.gauss_result[p*3:(p+1)*3]),color=color)
plt.plot(q.wave,q.Manygauss(np.log(q.wave),q.gauss_result),'b',lw=2)
plt.plot(q.wave,q.line_flux,'k')
plt.xlim(2750, 2850)
plt.ylim(-0.5, 3)
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)',fontsize = 20)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)',fontsize = 20)


# the line_prop function is used to calculate the broad line properties (Lnsigma > 0.0017 (1200 km/s) )
#fwhm,sigma,ew,peak,area = q.line_prop(q.linelist[6][0],q.gauss_result[0:27],'broad')
fwhm,sigma,ew,peak,area = q.line_prop(q.linelist[0][0],q.gauss_result[0:27],'broad')
print("MgII complex:")
print("FWHM (km/s)", fwhm)
print("Sigma (km/s)", sigma)
print("EW (A)",ew)
print("Peak (A)",peak)
print("area (10^(-17) erg/s/cm^2)",area)


# =============================================================================
# 6. Derive the physical parameters
# =============================================================================

print ()
print ('===== MgII result =====')

c = 299792.458 # speed of light in km/s
mgii_fwhm = fwhm

print ('MgII FWHM =', mgii_fwhm, 'km/s')


def calc_mass(cont_lum, fwhm):
    log_mass = 6.86 + 2*np.log10(fwhm/1e3) + 0.5*np.log10(cont_lum/1e44)
    return log_mass

L_3000 = 10**q.conti_result[-2]

M_BH = calc_mass(L_3000, mgii_fwhm)
print ('log M_BH/M_Sun = ', M_BH)

L_bol = 5.15*L_3000
print ('log L_bol =',  np.log10(L_bol))

L_Edd = 1.3*1e38 * (10**M_BH)
print ('L_bol/L_Edd = ', L_bol/L_Edd)

mgii_ew = ew
print('EW =', mgii_ew)