#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:33:32 2019

@author: irhamta
"""

# set basic figure parameters
import matplotlib as mpl
mpl_param = {'figure.figsize'   : [8.0, 6.0],
             'savefig.dpi'      : 300,
             'axes.titlesize'   : 'xx-large',
             'axes.labelsize'   : 'xx-large',
             'text.usetex'      : False,
             'font.family'      : 'serif'}
mpl.rcParams.update(mpl_param)


import pandas as pd
import timeit
import numpy as np

from PyQSOFit_dev import QSOFit

from astropy.io import fits
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# make cosmology model
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)

# plot line names
def plot_line_names(wave, flux, name=False):

    # the center wavelengths and names
    line_cen = np.array([6564.60,   6549.85,  6585.27,  6718.29,  6732.66,  4862.68,  5008.24,  4687.02,\
           4341.68,   3934.78,  3728.47,  3426.84,  2798.75,  1908.72,  1816.97,\
           1750.26,   1718.55,  1549.06,  1640.42,  1402.06,  1396.76,  1335.30,\
           1215.67])

    line_name = np.array(['',  '', 'Ha+NII', '', 'SII6718,6732', 'Hb', '[OIII]',\
            'HeII4687','Hr','CaII3934', 'OII3728', 'NeV3426', 'MgII','CIII]',\
            'SiII1816', 'NIII1750', 'NIV1718', 'CIV', 'HeII1640','',\
            'SiIV+OIV', 'CII1335','Lya'])


    for ll in range(len(line_cen)):
        if  wave.min() < line_cen[ll] < wave.max():
            plt.plot([line_cen[ll],line_cen[ll]],[flux.min()-1,flux.max()*1.5],'k:')

            # adjust the text position
            if name==True:
                plt.text(line_cen[ll]+10, np.nanmedian(flux)+3.5*np.std(flux), line_name[ll], rotation = 90, fontsize = 8)

def process():

# =============================================================================
# 1. Setup and read the input spectrum

# Setup the paths and read in your spectrum. Our code is written under the
# frame of SDSS spectral data format. Other data is also available as long as
# they include wavelength, flux, error, and redshift, and make sure
# the wavelength resolution is the same as SDSS spectrum
# (For SDSS the pixel scale is 1.e-4 in log space).
# =============================================================================


    path='./j083/'

    path1=path                  # the path of the source code file and qsopar.fits
    path2=path + 'result/'      # path of fitting results
    path3=path + 'figures/'     # path of figure
    path4=path + 'dustmaps/'    # path of dust reddening map

    # Required
    # an important note that all the data input must be finite, especically for the error !!!

    # The coordinate of the target
    ra, dec = 83.8370655133264, 11.8482345307907

    filename = 'spec1d_coadd_j083_tellcorr_proc.csv'
    data = pd.read_csv('../temp/' + filename)

    lam = data.wavelength        # OBS wavelength [A]
    flux = data.flux_corr             # OBS flux [erg/s/cm^2/A]
    err = data.flux_err          # 1 sigma error
    z = 6.345                    # Redshift

    # randomize the flux based on its 1-sigma error
    flux = np.random.normal(loc=flux, scale=err)

# =============================================================================
    # somehow we need to normalize the flux to make the correct fit
    nconst = 1e-17

    # there is bug in PyQSOFit where we need to manually add (1+z) factor
    # when shifting the spectrum to the rest-frame
    err = err*(1+z)/nconst
    flux = flux/nconst
# =============================================================================
# 2. Fitting the spectrum with various models

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

    # do the fitting
    q.Fit(name = None, nsmooth = 1, and_or_mask = False, deredden = False, reject_badpix = False, wave_range = [1100, 3050],\
          wave_mask =None, decomposition_host = False, Mi = None, npca_gal = 5, npca_qso = 20, \
          Fe_uv_op = True, poly = False, BC = False, rej_abs = True, initial_guess = None, MC = False, \
          n_trails = 5, linefit = True, tie_lambda = True, tie_width = True, tie_flux_1 = True, tie_flux_2 = True,\
          save_result = True, plot_fig = True,save_fig = True, plot_line_name = True, plot_legend = True, \
          dustmap_path = path4, save_fig_path = path3, save_fits_path = path2, save_fits_name = None)

    return q


def plot_fit(q):
    '''
    This function plots the spectral fitting result

    Parameters
    ----------
    q: PyQSOFit output
        The input is from PyQSOFit fitting result

    Returns
    ----------
    A plot containing spectral fitting result

    '''
    # =============================================================================
    # 6. Plot the result
    # =============================================================================

    # the upper panel containing a whole spectral range
    plt.figure(2, figsize=(15, 8))
    ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=3)

    plt.plot(q.wave, q.flux, 'k-', label='Observed flux', linewidth=1)
    plt.plot(q.wave, q.err, 'k-', label='Noise spectrum', linewidth=1, alpha=0.5)

    plt.plot(q.wave, q.f_conti_model+q.Manygauss(np.log(q.wave),q.gauss_result), 'r-')
    plt.plot(q.wave,q.f_fe_uv_model+q.f_pl_model,'orange', label='FeII')

    plt.plot(q.wave, q.f_pl_model, 'b-', label='PL continuum')
#    plt.plot(x, [-3e-18]*len(x), 'bs', label='Continuum window', markersize=2)


    plot_line_names(q.wave, q.flux, name=True)
    plt.ylim(np.nanmedian(q.flux) - 3*np.nanstd(q.flux),
             np.nanmedian(q.flux) + 5*np.nanstd(q.flux))
    plt.xlim(1125, 3000)
    plt.plot([1100, 3000], [0, 0], 'k--')
    plt.title('PSO J083+11 \t $z=%.3f$' %6.345)


    # MgII
    mgii = [(q.wave >= 2750) & (q.wave <= 2850)]
    mgii_flux = (q.flux-q.f_conti_model)[mgii]

    ax2 = plt.subplot2grid((5, 4), (3, 2), colspan=1, rowspan=2)

    plt.subplots_adjust(wspace = 0.5, hspace = 0.6)

    plt.plot(q.wave, q.flux-q.f_conti_model, 'k-', label='')
    plt.plot(q.wave, q.Manygauss(np.log(q.wave),q.gauss_result), 'r-', label='Fitted lines')
    plt.ylim(np.nanmedian(mgii_flux) - 2*np.std(mgii_flux),
             np.nanmedian(mgii_flux) + 4*np.std(mgii_flux))
    plt.xlim(2700, 2900)
    plt.text(0.7,0.9, 'MgII', fontsize = 15, transform = ax2.transAxes)
    plt.figlegend(loc='lower right', fontsize='xx-large')


    # Ly-a
    lya = [(q.wave >= 1160) & (q.wave <= 1290)]
    lya_flux = (q.flux-q.f_conti_model)[lya]

    ax3 = plt.subplot2grid((5, 4), (3, 0), colspan=1, rowspan=2)

    plt.subplots_adjust(wspace = 0.5, hspace = 0.6)

    plt.plot(q.wave, q.flux-q.f_conti_model, 'k-')
    plt.plot([1150, 1290], [0, 0], 'r-')

    plt.ylim(np.nanmedian(lya_flux) - 2*np.std(lya_flux),
             np.nanmedian(lya_flux) + 4*np.std(lya_flux))
    plt.xlim(1150, 1290)
    plt.text(0.7,0.9, r'Ly$\alpha$', fontsize = 15, transform = ax3.transAxes)

    # CIV
    civ = [(q.wave >= 1500) & (q.wave <= 1700)]
    civ_flux = (q.flux-q.f_conti_model)[civ]

    ax4 = plt.subplot2grid((5, 4), (3, 1), colspan=1, rowspan=2)

    plt.subplots_adjust(wspace = 0.5, hspace = 0.6)

    plt.plot(q.wave, q.flux-q.f_conti_model, 'k-')
    plt.plot([1500, 1600], [0, 0], 'r-')
    plt.ylim(np.nanmedian(civ_flux) - 2*np.std(civ_flux),
             np.nanmedian(civ_flux) + 4*np.std(civ_flux))
    plt.xlim(1500, 1600)
    plt.text(0.6,0.9, r'CIV (?)', fontsize = 15, transform = ax4.transAxes)

    plt.text(0.35, -0.95, r'Rest Wavelength [$\rm \AA$]', fontsize = 16, transform = ax1.transAxes)
    plt.text(-0.1,0.0, r'F$_\lambda \ \rm [10^{-17} \ erg \ cm^{-2} \ s^{-1} \ \AA^{-1}]$',fontsize = 16, transform = ax1.transAxes, rotation = 90)


if __name__ == '__main__':

    start = timeit.default_timer()

    # =============================================================================
    # 3. Initializing the input parameters

    # Firstly, run the script below to produce the line list file, qsopar.fits,
    # containing lines and their constraints, which will be needed in the
    # following fitting program. From this file, you can change some specific
    # parameters to suit your requirements, e.g., fitting range, line width,
    # tie line center, tie line sigma, etc. If you want to fit extra lines,
    # please append it to corresponding complex. Note that our line wavelength
    # and sigma in the list are in Ln scale, like Lnlambda, Lnsigma.
    # =============================================================================


    path='./j083/'

    newdata = np.rec.array([
                            #(2798.75,'MgII',2700.,2900.,'MgII_br',1,5e-3,0.004,0.05,0.0017,0,0,0,0.05),\
                            #(2798.75,'MgII',2700.,2900.,'MgII_na',2,1e-3,5e-4,0.0017,0.01,1,1,0,0.002),\

                            # we limit the maximum FWHM to 9*1200 km/s
                            (2798.75,'MgII',2700.,2900.,'MgII_br',1,1e-3,5e-4,9*0.0017,0.01,1,1,0,0.002),\
                            #(2798.75,'MgII',2700.,2900.,'MgII_br',1,5e-3,0.004,0.05,0.0017,0,0,0,0.05),\
                            #(2798.75,'MgII',2700.,2900.,'MgII_na',1,1e-3,5e-4,5*0.0017,0.01,1,1,0,0.002),\

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



    # =============================================================================
    # 4. Begin the fitting process and append the result to DataFrame
    # =============================================================================

    # whether you want to reuse the computed output or not
    read_from_file = False
    z = 6.345

    # here we do teh Monte-Carlo simulation to propagate the flux error
    for j in range(1):
        if read_from_file == True:
            result = pd.read_csv(path + 'result/result_temp.csv')
            break

        print ()
        print ('===== Processing object number: %i =====' %j)

        # do the fitting process
        q = process()

        # the line_prop function is used to calculate the broad line properties (Lnsigma > 0.0017 (1200 km/s) )
        # need to check why choose [0:27]
        fwhm, sigma, ew, peak, area = q.line_prop(q.linelist[0][0],
                                              q.gauss_result[0:27],
                                              'broad')
        print('MgII complex:')
        print('FWHM (km/s)', fwhm)
        print('Sigma (km/s)', sigma)
        print('EW (A)', ew)
        print('Peak (A)', peak)
        print('area (10^(-17) erg/s/cm^2)', area)

        plt.close('all')

        # produce the spectral fitting result and save the figures
        plot_fit(q)
        plt.savefig(path + 'figures/fit_result_%i' %j)

# =============================================================================
# Lya measurement
# =============================================================================

        lya_mask = [(q.wave >= 1160) & (q.wave <= 1290)]
        lya = pd.DataFrame({'wave'      : q.wave[lya_mask],
                            'flux'      : (q.flux-q.f_conti_model)[lya_mask],
                            'pl_cont'   : q.f_conti_model[lya_mask]})

        lya_cut = lya[lya['flux'] > 0].copy()

        print ('===== Lya result =====')

        lya_ew = (1 - ((lya_cut['flux']+lya_cut['pl_cont'])/lya_cut['pl_cont']))
        lya_ew_value = abs(np.trapz(y=lya_ew, x=lya_cut['wave']))

        print('Lya EW =', lya_ew_value)

# =============================================================================
# Calculate the m_1450 rest frame magnitude
# =============================================================================

        def power_law(x, amp, alpha):
            return amp * (x/3000)**alpha

#        plt.plot(q.wave, power_law(q.wave, q.conti_result[9], q.conti_result[10]))
#        plt.plot(q.wave, q.f_pl_model)

        # flux density in per Angstrom
        F_1450 = power_law(x=1450, amp=q.conti_result[9],
                           alpha=q.conti_result[10])*1e-17

        # flux density in per Angstrom, need to be converted to per Hz
        F_1450 = F_1450 * (3.34e4 * 1450**2)

        # Flux density in Jansky, converted to AB magniude
        m_1450 = -2.5*np.log10(F_1450) + 8.90

        # calculate the distance
        distance = cosmo.luminosity_distance(z).value # Mpc

        # Absoulte magnitude of M_1450
        M_1450 = m_1450 - (-5 + 5*np.log10(distance*1e6))

        print ('M_1450 =', M_1450)


# =============================================================================

        # save the result to DataFrame, initiate at the first loop
        if j == 0:
            result = pd.DataFrame({'mgii_fwhm'  : fwhm,
                                   'mgii_ew'    : ew,
                                   'mgii_flux'  : area,
                                   'pl_slope'   : q.conti_result[-7],
                                   'L3000'      : 10**q.conti_result[-2],

                                   'lya_ew'     : lya_ew_value,
                                   'M_1450'     : M_1450},
                                index=[j])

        # append the result to DataFrame
        else:
            result_temp = pd.DataFrame({'mgii_fwhm'  : fwhm,
                                        'mgii_ew'    : ew,
                                        'mgii_flux'  : area,
                                        'pl_slope'   : q.conti_result[-7],
                                        'L3000'      : 10**q.conti_result[-2],

                                        'lya_ew'     : lya_ew_value,
                                        'M_1450'     : M_1450},
                                index=[j])

            result = result.append(result_temp)

        # save to .csv file
        if j%10 == 0:
            result.to_csv(path + 'result/result_temp.csv', index=False)

    # =============================================================================
    # 5. Derive the physical parameters
    # =============================================================================

    c = 299792.458 # speed of light in km/s

    # mass calculation
    def calc_mass(cont_lum, fwhm):
        log_mass = 6.86 + 2*np.log10(fwhm/1e3) + 0.5*np.log10(cont_lum/1e44)
        return log_mass

    # insert the calculated value to result DataFrame
    result['M_BH'] = calc_mass(result['L3000'], result['mgii_fwhm'])
    result['L_bol'] = np.log10(5.15*result['L3000'])
    result['L_Edd'] = np.log10(1.3*1e38 * (10**result['M_BH']))
    result['L_bol/L_Edd'] = 10**(result['L_bol']-result['L_Edd'])

    # PyQSOFit store the line flux in 1e-17 unit, so we need to multiply it
    result['mgii_flux'] = result['mgii_flux']*1e-17

    # update and save the result again
    result.to_csv(path + 'result/result_temp.csv', index=False)

    # =============================================================================
    # 6. Plot the parameters distribution
    # =============================================================================

    def plot_dist(ind, par, label):
        '''
        This function plots the calculated physical parameters distribution

        Parameters
        ----------
        ind: tuple
            Axes position/index in the subplot
        par: str
            Column name in the DataFrame that will be used
        label: str
            The x-axis label for the plot
        Returns
        ----------
        Histograms of calculated parameters

        '''

        # plot the histogram
        axs[ind[0], ind[1]].hist(result[par], bins=25,
                                 facecolor='None', edgecolor='black')

        # plot the vlines for median and percentile errors
        axs[ind[0], ind[1]].axvline(result[par].median(),
                                       color='red', linestyle='--')
        axs[ind[0], ind[1]].axvline(result[par].quantile(0.16), color='red')
        axs[ind[0], ind[1]].axvline(result[par].quantile(0.84), color='red')

        # set the labels
        axs[ind[0], ind[1]].set_xlabel(label)

        # calculate the upper and lower error
        upper_err = result[par].quantile(0.84) - result[par].median()
        lower_err = result[par].median() - result[par].quantile(0.16)

        # set the title
        axs[ind[0], ind[1]].set_title(r'$%.3f^{+%.3f}_{-%.3f}$' \
           %(result[par].median(),
             upper_err,
             lower_err))

        print (par + r': %.3f +%.3f -%.3f' \
                   %(result[par].median(),
                     upper_err,
                     lower_err))


    # make the histograms of calculated parameters
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(wspace=0.3, hspace=0.7)

    print ()
    plot_dist((0, 0), 'mgii_fwhm', 'MgII FWHM (km/s)')
    plot_dist((0, 1), 'M_BH', r'$\log (M_{\rm BH}/M_{\odot})$')
    plot_dist((0, 2), 'L_bol/L_Edd', r'$L_{\rm bol}/L_{\rm Edd}$')
#    plot_dist((1, 0), 'L_bol', r'$\log L_{\rm bol}$')
    plot_dist((1, 0), 'M_1450', r'$M_{1450}$')
    plot_dist((1, 1), 'pl_slope', 'PL Slope')
    plot_dist((1, 2), 'lya_ew', r'Ly$\alpha$ EW ($\AA$)')

    plt.savefig(path + 'figures/par_estimate')

    end = timeit.default_timer()
    print('Processes finished in : '+str(np.round(end-start))+'s')

    plt.close('all')