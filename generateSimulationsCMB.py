"""
Generate lensed T, Q and U maps using lenspyx and save them in npy files.

Author: Miguel Ruiz Granda
"""

# *********************** PACKAGES ************************************************************************************
import numpy as np
from astropy.io import ascii
import healpy as hp
import os
os.chdir('/home/miguel/Desktop/TFM_2')
import lenspyx
interactiveMode = True
interactiveModeMaps = False
if interactiveMode:
    import matplotlib.pyplot as plt

# *********************** CONSTANTS ***********************************************************************************
T_CMB = 2.7255e6  # in muK
nside = 2048  #  2048
npix = hp.nside2npix(nside)
resol = hp.nside2resol(nside, arcmin=True)  # in arcmin
lmax = 2500
lmax_lensquest = 500
dlmax = 1000  # lmax of the unlensed fields is lmax + dlmax (for accurate lensing)
# The deflected map is constructed by interpolation of the undeflected map, built
# at target resolution approx. 0.7∗2**facres arcmin
facres = -1

FWHM_Planck = np.radians(5/60)  # radians
TNoisePlanck = 20 / resol  # muK  33/resol
PNoisePlanck = 40 / resol  # muK  70.2/resol
FWHM_LB = np.radians(30/60)  # radians
TPNoiseLB = 2.2 / resol  # muK
# *********************************************************************************************************************

# STEP 1: GENERATE CORRELATED T,Q,U (LENSED) AND PHI MAPS *************************************************************

# #####################################################################################################################
# Step 1.1: Use CLASS to generate theoretical unlensed and lensed CMB TT, EE, BB,
# TE, phi-phi, Tphi and Ephi power spectra. The cosmological parameters are obtained
# from Planck 2018 TT,TE,EE+lowE+lensing results and r = 0 situation is assumed.
# The file cl_ref.pre file was used to set the precision parameters to calculate
# precisely the CMB power spectra setting k_max_tau0_over_l_max=10.
# #####################################################################################################################

# Read the data files and convert them into numpy arrays. The column names are:
# colnames = ['1:l', '2:TT', '3:EE', '4:TE', '5:BB', '6:phiphi', '7:TPhi', '8:Ephi']
cls = ascii.read("/home/miguel/Desktop/TFM_2/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2.dat",
                 format="commented_header", header_start=10).as_array()
clsLensed = ascii.read("/home/miguel/Desktop/TFM_2/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2_lensed.dat",
                       format="commented_header", header_start=10).as_array()

if interactiveMode:
    # Plot CAMB and CLASS unlensed and lensed CMB power spectra
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].plot(cls['1:l'], cls['2:TT'] * T_CMB ** 2, color='k', label="Unlensed")
    ax[0, 0].plot(clsLensed['1:l'], clsLensed['2:TT'] * T_CMB ** 2, color='r', label="Lensed")
    ax[0, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TT}\ /\ \mu K^2$', size=18)
    ax[0, 1].plot(cls['1:l'], cls['3:EE'] * T_CMB ** 2, color='k', label="Unlensed")
    ax[0, 1].plot(clsLensed['1:l'], clsLensed['3:EE'] * T_CMB ** 2, color='r', label="Lensed")
    ax[0, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{EE}\ /\ \mu K^2$', size=18)
    ax[1, 0].plot(cls['1:l'], cls['5:BB'] * T_CMB ** 2, color='k', label="Unlensed")
    ax[1, 0].plot(clsLensed['1:l'], clsLensed['5:BB'] * T_CMB ** 2, color='r', label="Lensed")
    ax[1, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{BB}\ /\ \mu K^2$', size=18)
    ax[1, 1].plot(cls['1:l'], cls['4:TE'] * T_CMB ** 2, color='k', label="Unlensed")
    ax[1, 1].plot(clsLensed['1:l'], clsLensed['4:TE'] * T_CMB ** 2, color='r', label="Lensed")
    ax[1, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TE}\ /\ \mu K^2$', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax+dlmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig('ClsTTTEEEBB.pdf')

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(cls['1:l'], cls['6:phiphi'] * cls['1:l'] * (cls['1:l'] + 1), color='k') # , label="Unlensed")
    ax[0].set_ylabel(r'$\frac{[\ell(\ell +1)]^2}{2\pi}C_\ell^{\phi\phi}$', size=18)
    ax[0].set_xscale('log')
    ax[1].plot(cls['1:l'], cls['7:TPhi'] * (cls['1:l'] * (cls['1:l'] + 1)) ** 0.5 * T_CMB, color='k') #, label="Unlensed")
    # ax[1].plot(clsLensed['1:l'], clsLensed['7:TPhi'] * (clsLensed['1:l'] * (clsLensed['1:l'] + 1)) ** 0.5 * T_CMB,
    #           color='r', label="Lensed")
    ax[1].set_ylabel(r'$\frac{[\ell(\ell +1)]^{3/2}}{2\pi}C_\ell^{T\phi}\ /\ \mu K$', size=18)
    ax[1].set_xscale('log')
    # ax[1].legend(fontsize=14)
    ax[2].plot(cls['1:l'], cls['8:Ephi'] * (cls['1:l'] * (cls['1:l'] + 1)) ** 0.5 * T_CMB, color='k') #, label="Unlensed")
    #ax[2].plot(clsLensed['1:l'], clsLensed['8:Ephi'] * (clsLensed['1:l'] * (clsLensed['1:l'] + 1)) ** 0.5 * T_CMB,
    #           color='r', label="Lensed")
    ax[2].set_ylabel(r'$\frac{[\ell(\ell +1)]^{3/2}}{2\pi}C_\ell^{E\phi}\ /\ \mu K$', size=18)
    ax[2].set_xscale('log')
    # ax[2].legend(fontsize=14)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax+dlmax])
        ax.set_xlabel(r'$\ell$', size=18)
    fig.tight_layout()
    plt.show()
    plt.savefig('ClsPhiPhi.pdf')

# #####################################################################################################################
# Step 1.2: Generate samples of three uncorrelated Gaussian variables of zero mean
# and unit variance, h, j and k. To do so, using a constant power spectrum of value
# 1.0 is used to generate three sets of alm coefficients using the healpy function
# synalm.
# #####################################################################################################################
flatCls = np.ones(lmax + dlmax + 1)
hlm = hp.synalm(flatCls, lmax=lmax+dlmax, new=True)
jlm = hp.synalm(flatCls, lmax=lmax+dlmax, new=True)
klm = hp.synalm(flatCls, lmax=lmax+dlmax, new=True)

Clh = np.sqrt(hp.alm2cl(hlm, lmax=lmax+dlmax))
Clj = np.sqrt(hp.alm2cl(jlm, lmax=lmax+dlmax))
Clk = np.sqrt(hp.alm2cl(klm, lmax=lmax+dlmax))

# #####################################################################################################################
# Step 1.3: Obtain correlated tlm_unl, elm_unl and plm_unl coefficients using the
# unlensed theoretical CMB power spectra generated using CLASS and the hlm, jlm and
# klm coefficients. To do so, a cholesky decomposition C(l) = L(l)*L^T(l) of the
# covariance matrix of the power spectra is performed and, following, a matrix
# multiplication M(l,m, correlated) = L(l) * M(l, m, uncorrelated) which leads to
# the correlated t_lm, e_lm and phi_lm coefficients.
# #####################################################################################################################

# Initialize the arrays
tlm_unl = np.zeros_like(hlm)
elm_unl = np.zeros_like(hlm)
blm_unl = np.zeros_like(hlm)  # zero (no lensing)
plm = np.zeros_like(hlm)

# Fill the arrays using cholesky decomposition technique
for l in range(2, lmax + dlmax + 1):
    C = np.array([[cls['2:TT'][l-2], cls['4:TE'][l-2], cls['7:TPhi'][l-2]],
                  [cls['4:TE'][l-2], cls['3:EE'][l-2], cls['8:Ephi'][l-2]],
                  [cls['7:TPhi'][l-2], cls['8:Ephi'][l-2], cls['6:phiphi'][l-2]]])
    L = np.linalg.cholesky(C*2*np.pi/(l*(l+1)))
    for m in range(l + 1):
        ind = hp.Alm.getidx(lmax + dlmax, l, m)
        tlm_unl[ind], elm_unl[ind], plm[ind] = np.dot(L, np.array([hlm[ind], jlm[ind], klm[ind]]))
Cl_unlen = hp.alm2cl([tlm_unl, elm_unl, blm_unl], lmax=lmax+1000)

if interactiveMode:
    ls = np.arange(Cl_unlen.shape[1])
    # Plot CAMB and CLASS unlensed and lensed CMB power spectra
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].plot(ls, ls*(ls+1)*Cl_unlen[0, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="Lenspyx")
    ax[0, 0].plot(cls['1:l'], cls['2:TT'] * T_CMB ** 2, color='r', label="Theory")
    ax[0, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TT}\ /\ \mu K^2$', size=18)
    ax[0, 1].plot(ls, ls * (ls + 1) * Cl_unlen[1, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="Lenspyx")
    ax[0, 1].plot(cls['1:l'], cls['3:EE'] * T_CMB ** 2, color='r', label="Theory")
    ax[0, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{EE}\ /\ \mu K^2$', size=18)
    ax[1, 0].plot(ls, ls * (ls + 1) * Cl_unlen[2, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="Lenspyx")
    ax[1, 0].plot(cls['1:l'], cls['5:BB'] * T_CMB ** 2, color='r', label="Theory")
    ax[1, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{BB}\ /\ \mu K^2$', size=18)
    ax[1, 1].plot(ls, ls * (ls + 1) * Cl_unlen[3, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="Lenspyx")
    ax[1, 1].plot(cls['1:l'], cls['4:TE'] * T_CMB ** 2, color='r', label="Theory")
    ax[1, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TE}\ /\ \mu K^2$', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=10)
    fig.tight_layout()

np.save('teb_unlensed.npy', np.array([tlm_unl, elm_unl, blm_unl, plm]))

# #####################################################################################################################
# Step 1.4: Obtain T, Q and U lensed maps using lenspyx and save them in 'RawLensedMaps.npy'.
# #####################################################################################################################

# We transform the lensing potential into spin-1 deflection field.
dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + dlmax + 1, dtype=float) * np.arange(1, lmax + dlmax + 2)))

# We compute the position-space deflection.
Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))

# Computes the temperature deflected spin-0 healpix map from tlm_unl and deflection field dlm.
Tlen = lenspyx.alm2lenmap(tlm_unl, [Red, Imd], nside, facres=facres, verbose=True)
# Computes a deflected spin-weight healpix map from its gradient, elm_unl, and curl, blm_unl, modes
# and deflection field dlm.
Qlen, Ulen = lenspyx.alm2lenmap_spin([elm_unl, blm_unl], [Red, Imd], nside, 2, facres=facres, verbose=True)

np.save('RawLensedMaps', np.array([Tlen, Qlen, Ulen]))

if interactiveMode:
    fig, axes = plt.subplots(3, 1, figsize=(5, 10))
    plt.axes(axes[0])
    hp.mollview(Tlen, title='T', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[1])
    hp.mollview(Qlen, title='Q', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[2])
    hp.mollview(Ulen, title='U', cmap='Spectral_r', norm='hist', hold=True)
    plt.tight_layout()
    plt.savefig('RawMaps.pdf')

# #####################################################################################################################
# 4. Checking everything is right...
# #####################################################################################################################

tlm_len, elm_len, blm_len = hp.map2alm([Tlen, Qlen, Ulen], lmax=lmax, pol=True, use_pixel_weights=True)
np.save('teb_lensed.npy', np.array([tlm_len, elm_len, blm_len]))
Cl_len = hp.alm2cl([tlm_len, elm_len, blm_len], lmax=lmax)

if interactiveMode:
    ls = np.arange(Cl_len.shape[1])
    # Plot CAMB and CLASS unlensed and lensed CMB power spectra
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].plot(ls, ls*(ls+1)*Cl_len[0, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[0, 0].plot(clsLensed['1:l'], clsLensed['2:TT'] * T_CMB ** 2, color='r', label="Theory")
    ax[0, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TT}\ /\ \mu K^2$', size=18)
    ax[0, 1].plot(ls, ls * (ls + 1) * Cl_len[1, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[0, 1].plot(clsLensed['1:l'], clsLensed['3:EE'] * T_CMB ** 2, color='r', label="Theory")
    ax[0, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{EE}\ /\ \mu K^2$', size=18)
    ax[1, 0].plot(ls, ls * (ls + 1) * Cl_len[2, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[1, 0].plot(clsLensed['1:l'], clsLensed['5:BB'] * T_CMB ** 2, color='r', label="Theory")
    ax[1, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{BB}\ /\ \mu K^2$', size=18)
    ax[1, 1].plot(ls, ls * (ls + 1) * Cl_len[3, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[1, 1].plot(clsLensed['1:l'], clsLensed['4:TE'] * T_CMB ** 2, color='r', label="Theory")
    ax[1, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TE}\ /\ \mu K^2$', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig('ClsLensed_Theory.pdf')

# *********************************************************************************************************************
# STEP 2: GENERATE SIMULATIONS OF (T,Q,U) MAPS OBSERVED BY PLANCK, LITEBIRD AND THE
# COMBINATION OF PLANCK AND LITEBIRD **********************************************************************************

# #####################################################################################################################
# 1. PLANCK (T,Q,U) CMB MAPS
# #####################################################################################################################
TlenPl, QlenPl, UlenPl = hp.smoothing([Tlen, Qlen, Ulen], fwhm=FWHM_Planck, pol=True,
                                   lmax=lmax, use_pixel_weights=True)
TlenPl = TlenPl + np.random.normal(loc=0.0, scale=TNoisePlanck/T_CMB, size=npix)
QlenPl = QlenPl + np.random.normal(loc=0.0, scale=PNoisePlanck/T_CMB, size=npix)
UlenPl = UlenPl + np.random.normal(loc=0.0, scale=PNoisePlanck/T_CMB, size=npix)

tlmPl, elmPl, blmPl = hp.map2alm([TlenPl, QlenPl, UlenPl], lmax=lmax, pol=True, use_pixel_weights=True)
np.save('teb_Planck.npy', [tlmPl, elmPl, blmPl])

if interactiveModeMaps:
    fig, axes = plt.subplots(3, 1, figsize=(5, 10))
    plt.axes(axes[0])
    hp.mollview(TlenPl, title='T Planck', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[1])
    hp.mollview(QlenPl, title='Q Planck', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[2])
    hp.mollview(UlenPl, title='U Planck', cmap='Spectral_r', norm='hist', hold=True)
    plt.tight_layout()
    plt.savefig('PlanckMaps.pdf')

# #####################################################################################################################
# 2. LITEBIRD (T,Q,U) CMB MAPS
# #####################################################################################################################

TlenLB, QlenLB, UlenLB = hp.smoothing([Tlen, Qlen, Ulen], fwhm=FWHM_LB, pol=True,
                                   lmax=lmax, use_pixel_weights=True)

TlenLB = TlenLB + np.random.normal(loc=0.0, scale=TPNoiseLB/T_CMB, size=npix)
QlenLB = QlenLB + np.random.normal(loc=0.0, scale=TPNoiseLB/T_CMB, size=npix)
UlenLB = UlenLB + np.random.normal(loc=0.0, scale=TPNoiseLB/T_CMB, size=npix)

tlmLB, elmLB, blmLB = hp.map2alm([TlenLB, QlenLB, UlenLB], lmax=lmax, pol=True, use_pixel_weights=True)
np.save('teb_LiteBIRD.npy', np.array([tlmLB, elmLB, blmLB]))

if interactiveModeMaps:
    fig, axes = plt.subplots(3, 1, figsize=(5, 10))
    plt.axes(axes[0])
    hp.mollview(TlenLB, title='T LiteBIRD', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[1])
    hp.mollview(QlenLB, title='Q LiteBIRD', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[2])
    hp.mollview(UlenLB, title='U LiteBIRD', cmap='Spectral_r', norm='hist', hold=True)
    plt.tight_layout()
    plt.savefig('LBMaps.pdf')

# #####################################################################################################################
# 3. PLANCK + LITEBIRD (T,Q,U) CMB MAPS
# #####################################################################################################################

beamPl = hp.gauss_beam(fwhm=FWHM_Planck, lmax=lmax, pol=True)
beamLB = hp.gauss_beam(fwhm=FWHM_LB, lmax=lmax, pol=True)

wP = 1 / (4*np.pi*np.array([TNoisePlanck**2, PNoisePlanck**2, PNoisePlanck**2])*(beamPl[:,0:3]**-2)/npix)
wL = 1 / (4*np.pi*np.array([TPNoiseLB**2, TPNoiseLB**2, TPNoiseLB**2])*(beamLB[:,0:3]**-2)/npix)

sum = wP + wL

wP = wP/sum
wL = wL/sum
np.save('weights.npy', np.array([wP, wL]))

tlmPLB = hp.almxfl(tlmPl, wP[:, 0]) + hp.almxfl(tlmLB, wL[:, 0])
elmPLB = hp.almxfl(elmPl, wP[:, 1]) + hp.almxfl(elmLB, wL[:, 1])
blmPLB = hp.almxfl(blmPl, wP[:, 2]) + hp.almxfl(blmLB, wL[:, 2])

np.save('teb_Planck_LiteBIRD.npy', np.array([tlmPLB, elmPLB, blmPLB]))

combinationMaps = hp.alm2map([tlmPLB, elmPLB, blmPLB], nside, lmax=lmax)

if interactiveModeMaps:
    fig, axes = plt.subplots(3, 1, figsize=(5, 10))
    plt.axes(axes[0])
    hp.mollview(combinationMaps[0, :], title='T Planck+LiteBIRD', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[1])
    hp.mollview(combinationMaps[1, :], title='Q Planck+LiteBIRD', cmap='Spectral_r', norm='hist', hold=True)
    plt.axes(axes[2])
    hp.mollview(combinationMaps[2, :], title='U Planck+LiteBIRD', cmap='Spectral_r', norm='hist', hold=True)
    plt.tight_layout()
    plt.savefig('PlanckLBMaps.pdf')

# #####################################################################################################################
# 4. Checking everything is right... Plotting Planck, LiteBird and Planck+LiteBird power spectra. Also, the weights and
#    deconvolution factors for the Planck and LiteBIRD combination are plotted.
# #####################################################################################################################

tlm_len, elm_len, blm_len = hp.map2alm([Tlen, Qlen, Ulen], lmax=lmax, pol=True, use_pixel_weights=True)
Cl_len = hp.alm2cl([tlm_len, elm_len, blm_len], lmax=lmax)
ClPlanck = hp.alm2cl([tlmPl, elmPl, blmPl], lmax=lmax)
ClLB = hp.alm2cl([tlmLB, elmLB, blmLB], lmax=lmax)
ClPlLB = hp.alm2cl([tlmPLB, elmPLB, blmPLB], lmax=lmax)

deconvolutionPlLB = np.array([(wP[:, 0]*beamPl[:, 0])**2+(wL[:, 0]*beamLB[:, 0])**2+2*wP[:, 0]*beamPl[:, 0]*wL[:, 0]*beamLB[:, 0],
                              (wP[:, 1]*beamPl[:, 1])**2+(wL[:, 1]*beamLB[:, 1])**2+2*wP[:, 1]*beamPl[:, 1]*wL[:, 1]*beamLB[:, 1],
                              (wP[:, 2]*beamPl[:, 2])**2+(wL[:, 2]*beamLB[:, 2])**2+2*wP[:, 2]*beamPl[:, 2]*wL[:, 2]*beamLB[:, 2],
                              wP[:, 0]*wP[:, 1]*beamPl[:, 0]*beamPl[:, 1]+wL[:, 0]*wL[:, 1]*beamLB[:, 0]*beamLB[:, 1]+
                              wP[:, 0]*wL[:, 1]*beamPl[:, 0]*beamLB[:, 1]+wP[:, 1]*wL[:, 0]*beamPl[:, 1]*beamLB[:, 0]])

if interactiveMode:
    # Weights.
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(wP[:, 0]**2, color='k', label=r'$w_{\ell}^{P,T}$')
    ax[0].plot(wL[:, 0]**2, color='r', label=r'$w_{\ell}^{L,T}$')
    ax[0].set_ylabel('Weights', size=18)
    ax[1].plot(wP[:, 1]**2, color='k', label=r'$w_{\ell}^{P,E}=w_{\ell}^{P,B}$')
    ax[1].plot(wL[:, 1]**2, color='r', label=r'$w_{\ell}^{L,E}=w_{\ell}^{L,B}$')
    ax[1].plot(wP[:, 1]**2+wL[:, 1]**2, color='r', label=r'$w_{\ell}^{L,E}=w_{\ell}^{L,B}$')
    ax[1].set_ylabel('Weights', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig('weights.pdf')

    N_T_planck = 4*np.pi*TNoisePlanck**2/npix
    N_P_planck = 4 * np.pi * PNoisePlanck ** 2 / npix
    N_T_LB = 4 * np.pi * TPNoiseLB ** 2 / npix
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(N_T_planck*np.ones_like(wP[:, 0]), color='k', label=r'$N_{\ell}^{TT,P}$')
    ax[0].plot(N_T_LB*np.ones_like(wP[:, 0]), color='r', label=r'$N_{\ell}^{TT,L}$')
    ax[0].plot(wP[:, 0]**2*N_T_planck+wL[:, 0]**2*N_T_LB, color='g', label=r'$N_{\ell}^{TT,C}$')
    ax[0].set_ylabel(r'Noise temperature / $\mu$K$^2$', size=18)
    ax[1].plot(N_P_planck*np.ones_like(wP[:, 0]), color='k', label=r'$N_{\ell}^{EE,P}=N_{\ell}^{BB,P}$')
    ax[1].plot(N_T_LB*np.ones_like(wP[:, 0]), color='r', label=r'$N_{\ell}^{EE,L}=N_{\ell}^{BB,L}$')
    ax[1].plot(wP[:, 1]**2*N_P_planck+wL[:, 1]**2*N_T_LB, color='g', label=r'$N_{\ell}^{EE,C}=N_{\ell}^{BB,C}$')
    ax[1].set_ylabel(r'Noise polarization / $\mu$K$^2$', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig('noise.pdf')

    # Deconvolved noise
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(N_T_planck*beamPl[:, 0] ** (-2), color='b', label=r'$N_{\ell}^{TT,P}{b_{\ell,TT}^P}^{-2}$')
    ax[0].plot(N_T_LB*beamLB[:, 0] ** (-2), color='r', label=r'$N_{\ell}^{TT,L}{b_{\ell,TT}^L}^{-2}$')
    ax[0].plot((wP[:, 0]**2*N_T_planck+wL[:, 0]**2*N_T_LB)/deconvolutionPlLB[0, :], color='g', label=r'$N_{\ell}^{TT,C}{b_{\ell,TT}^C}^{-2}$')
    ax[0].plot(wP[:, 0] ** 2 * N_T_planck * beamPl[:, 1] ** (-2) + wL[:, 0] ** 2 * N_T_LB * beamLB[:, 1] ** (-2),
               color='k',
               label=r'Bueno')
    ax[0].set_ylabel(r'Deconvolved noise temp. / $\mu$K$^2$', size=18)
    ax[1].plot(N_P_planck*beamPl[:, 1] ** (-2), color='b', label=r'$N_{\ell}^{EE,P}{b_{\ell,EE}^P}^{-2}=N_{\ell}^{BB,P}{b_{\ell,BB}^P}^{-2}$')
    ax[1].plot(N_T_LB*beamLB[:, 1] ** (-2), color='r', label=r'$N_{\ell}^{EE,L}{b_{\ell,EE}^L}^{-2}=N_{\ell}^{BB,L}{b_{\ell,BB}^L}^{-2}$')
    ax[1].plot((wP[:, 1]**2*N_P_planck+wL[:, 1]**2*N_T_LB)/deconvolutionPlLB[1, :], color='g', label=r'$N_{\ell}^{EE,C}{b_{\ell,EE}^C}^{-2}=N_{\ell}^{BB,C}{b_{\ell,BB}^C}^{-2}$')
    # ax[1].plot(1/(1/(N_T_LB * beamLB[:, 1] ** (-2))+1/(N_P_planck*beamPl[:, 1] ** (-2))), color='k',
    #            label=r'$N_{\ell}^{EE,L}{b_{\ell,EE}^L}^{-2}=N_{\ell}^{BB,L}{b_{\ell,BB}^L}^{-2}$')
    ax[1].plot(wP[:, 1]**2*N_P_planck*beamPl[:, 1] ** (-2) + wL[:, 1]**2*N_T_LB*beamLB[:, 1] ** (-2), color='k',
               label=r'Bueno')
    ax[1].set_ylabel(r'Deconvolved noise pol. / $\mu$K$^2$', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.semilogy()
        ax.set_ylim([1e-7, 4e-3])
        ax.legend(fontsize=10)
    fig.tight_layout()
    plt.savefig('DeconvolutionNoise.pdf')

    # Power spectra before deconvolution.
    ls = np.arange(Cl_len.shape[1])
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].plot(ls, ls*(ls+1)*Cl_len[0, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="Theory")
    ax[0, 0].plot(ls, ls*(ls+1)*ClPlanck[0, :] * T_CMB ** 2 / (2 * np.pi), color='b', label="Planck")
    ax[0, 0].plot(ls, ls * (ls + 1) * ClLB[0, :] * T_CMB ** 2 / (2 * np.pi), color='r', label="LiteBIRD")
    ax[0, 0].plot(ls, ls * (ls + 1) * ClPlLB[0, :] * T_CMB ** 2 / (2 * np.pi), color='g', label="Planck+LiteBIRD")
    ax[0, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TT}\ /\ \mu K^2$', size=18)
    ax[0, 1].plot(ls, ls * (ls + 1) * ClPlanck[1, :] * T_CMB ** 2 / (2 * np.pi), color='b', label="Planck")
    ax[0, 1].plot(ls, ls * (ls + 1) * ClLB[1, :] * T_CMB ** 2 / (2 * np.pi), color='r', label="LiteBIRD")
    ax[0, 1].plot(ls, ls * (ls + 1) * ClPlLB[1, :] * T_CMB ** 2 / (2 * np.pi), color='g', label="Planck+LiteBIRD")
    ax[0, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{EE}\ /\ \mu K^2$', size=18)
    ax[1, 0].plot(ls, ls * (ls + 1) * Cl_len[2, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="Theory")
    ax[1, 0].plot(ls, ls * (ls + 1) * ClPlanck[2, :] * T_CMB ** 2 / (2 * np.pi), color='b', label="Planck")
    ax[1, 0].plot(ls, ls * (ls + 1) * ClLB[2, :] * T_CMB ** 2 / (2 * np.pi), color='r', label="LiteBIRD")
    ax[1, 0].plot(ls, ls * (ls + 1) * ClPlLB[2, :] * T_CMB ** 2 / (2 * np.pi), color='g', label="Planck+LiteBIRD")
    ax[1, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{BB}\ /\ \mu K^2$', size=18)
    ax[1, 1].plot(ls, ls * (ls + 1) * Cl_len[3, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="Theory")
    ax[1, 1].plot(ls, ls * (ls + 1) * ClPlanck[3, :] * T_CMB ** 2 / (2 * np.pi), color='b', label="Planck")
    ax[1, 1].plot(ls, ls * (ls + 1) * ClLB[3, :] * T_CMB ** 2 / (2 * np.pi), color='r', label="LiteBIRD")
    ax[1, 1].plot(ls, ls * (ls + 1) * ClPlLB[3, :] * T_CMB ** 2 / (2 * np.pi), color='g', label="Planck+LiteBIRD")
    ax[1, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TE}\ /\ \mu K^2$', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=10)
    fig.tight_layout()
    plt.savefig('ClsRecovered.pdf')

    # Power spectra after deconvolution.
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].plot(ls, ls * (ls + 1) * Cl_len[0, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[0, 0].plot(ls, ls * (ls + 1) * ClPlanck[0, :] * T_CMB ** 2 / (2 * np.pi * beamPl[:, 0]**2), color='b', label="Planck")
    ax[0, 0].plot(ls, ls * (ls + 1) * ClLB[0, :] * T_CMB ** 2 / (2 * np.pi * beamLB[:, 0]**2), color='r', label="LiteBIRD")
    ax[0, 0].plot(ls, ls * (ls + 1) * ClPlLB[0, :] * T_CMB ** 2 / (2 * np.pi*deconvolutionPlLB[0, :]), color='g', label="Planck+LiteBIRD")
    ax[0, 0].set_ylim([-100, 10000])
    ax[0, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TT}\ /\ \mu K^2$', size=18)
    ax[0, 1].plot(ls, ls * (ls + 1) * Cl_len[1, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[0, 1].plot(ls, ls * (ls + 1) * ClPlanck[1, :] * T_CMB ** 2 / (2 * np.pi * beamPl[:, 1]**2), color='b', label="Planck")
    ax[0, 1].plot(ls, ls * (ls + 1) * ClLB[1, :] * T_CMB ** 2 / (2 * np.pi * beamLB[:, 1]**2), color='r', label="LiteBIRD")
    ax[0, 1].plot(ls, ls * (ls + 1) * ClPlLB[1, :] * T_CMB ** 2 / (2 * np.pi*deconvolutionPlLB[1, :]), color='g', label="Planck+LiteBIRD")
    ax[0, 1].set_ylim([-1, 150])
    ax[0, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{EE}\ /\ \mu K^2$', size=18)
    ax[1, 0].plot(ls, ls * (ls + 1) * Cl_len[2, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[1, 0].plot(ls, ls * (ls + 1) * ClPlanck[2, :] * T_CMB ** 2 / (2 * np.pi * beamPl[:, 2]**2), color='b', label="Planck")
    ax[1, 0].plot(ls, ls * (ls + 1) * ClLB[2, :] * T_CMB ** 2 / (2 * np.pi * beamLB[:, 2]**2), color='r', label="LiteBIRD")
    ax[1, 0].plot(ls, ls * (ls + 1) * ClPlLB[2, :] * T_CMB ** 2 / (2 * np.pi*deconvolutionPlLB[2, :]), color='g', label="Planck+LiteBIRD")
    ax[1, 0].set_ylim([0, 0.15])
    ax[1, 0].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{BB}\ /\ \mu K^2$', size=18)
    ax[1, 1].plot(ls, (ls * (ls + 1) * ClLB[3, :] * T_CMB ** 2 / (2 * np.pi * beamLB[:, 0] * beamLB[:, 1])), color='r', label="LiteBIRD")
    ax[1, 1].plot(ls, ls * (ls + 1) * ClPlanck[3, :] * T_CMB ** 2 / (2 * np.pi * beamPl[:, 0] * beamPl[:, 1]), color='b', label="Planck")
    ax[1, 1].plot(ls, ls * (ls + 1) * ClPlLB[3, :] * T_CMB ** 2 / (2 * np.pi*deconvolutionPlLB[3, :]), color='g', label="Planck+LiteBIRD")
    ax[1, 1].plot(ls, ls * (ls + 1) * Cl_len[3, :] * T_CMB ** 2 / (2 * np.pi), color='k', label="lenspyx")
    ax[1, 1].set_ylim([-200, 200])
    ax[1, 1].set_ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell^{TE}\ /\ \mu K^2$', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=10)
    fig.tight_layout()
    plt.savefig('ClsDeconvolved.pdf')

    # Deconvolution factors
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(beamPl[:, 0]**2, color='b', label=r'${b_{\ell,TT}^P}^2$')
    ax[0].plot(beamLB[:, 0]**2, color='r', label=r'${b_{\ell,TT}^L}^2$')
    ax[0].plot(deconvolutionPlLB[0, :], color='g', label=r'${b_{\ell,TT}^C}^2$')
    ax[0].set_ylabel('Beam TT', size=18)
    ax[1].plot(beamPl[:, 1]**2, color='b', label=r'${b_{\ell,EE}^P}^2={b_{\ell,BB}^P}^2$')
    ax[1].plot(beamLB[:, 1]**2, color='r', label=r'${b_{\ell,EE}^L}^2={b_{\ell,BB}^L}^2$')
    ax[1].plot(deconvolutionPlLB[1, :], color='g', label=r'${b_{\ell,EE}^C}^2={b_{\ell,BB}^C}^2$')
    ax[1].set_ylabel('Beam EE/BB', size=18)
    ax[2].plot(beamPl[:, 0]*beamPl[:, 1], color='b', label=r'${b_{\ell,TE}^P}^2$')
    ax[2].plot(beamLB[:, 0]*beamLB[:, 1], color='r', label=r'${b_{\ell,TE}^L}^2$')
    ax[2].plot(deconvolutionPlLB[3, :], color='g', label=r'${b_{\ell,TE}^C}^2$')
    ax[2].set_ylabel('Beam TE', size=18)
    for ax in ax.reshape(-1):
        ax.set_xlim([2, lmax])
        ax.set_xlabel(r'$\ell$', size=18)
        ax.legend(fontsize=10)
    fig.tight_layout()
    plt.savefig('Deconvolution.pdf')