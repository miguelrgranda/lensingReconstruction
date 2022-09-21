"""
Lensquest: Minimum Variance estimator calculation for the simulation of Planck experiment.

Author: Miguel Ruiz Granda
"""

# *********************** PACKAGES ************************************************************************************
import numpy as np
from astropy.io import ascii
import healpy as hp
# from ctypes import *
# lib = cdll.LoadLibrary('/gpfs/users/ruizm/Healpix_3.81/lib/libhealpix_cxx.so.3')
import lensquest
interactiveMode = True
interactiveModeMaps = False
if interactiveMode:
    import matplotlib.pyplot as plt

# *********************** CONSTANTS ***********************************************************************************
T_CMB = 2.7255e6  # in muK
nside = 2048
npix = hp.nside2npix(nside)
resol = hp.nside2resol(nside, arcmin=True)  # in arcmin
lmax = 2500
lmax_lensquest = 2500
dlmax = 1000  # lmax of the unlensed fields is lmax + dlmax (for accurate lensing)
# The deflected map is constructed by interpolation of the undeflected map, built
# at target resolution approx. 0.7∗2**facres arcmin
facres = -1

FWHM_Planck = np.radians(5/60)  # radians
TNoisePlanck = 20 / resol  # muK
PNoisePlanck = 40 / resol  # muK
# *********************************************************************************************************************

# Read the cl data files and convert them into numpy arrays. The column names are:
# colnames = ['1:l', '2:TT', '3:EE', '4:TE', '5:BB', '6:phiphi', '7:TPhi', '8:Ephi']
cls = ascii.read("/gpfs/users/ruizm/TFM/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2.dat",
                 format="commented_header", header_start=10).as_array()
clsLensed = ascii.read("/gpfs/users/ruizm/TFM/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2_lensed.dat",
                       format="commented_header", header_start=10).as_array()
# Read the (tlm,elm,blm) lensed coefficients generated using generateSimulationsCMB.py.
tlmPl, elmPl, blmPl = np.load('teb_Planck.npy')

# Beam functions of Planck
beamPl = hp.gauss_beam(fwhm=FWHM_Planck, lmax=lmax, pol=True)

# Angular power spectra
ClPlanck = hp.alm2cl([tlmPl, elmPl, blmPl], lmax=lmax)

# *********************************************************************************************************************

# STEP 3: LENSQUEST: CMB lensing reconstruction ***********************************************************************

# *********************************************************************************************************************

# colnames = ['1:l', '2:TT', '3:EE', '4:TE', '5:BB', '6:phiphi', '7:TPhi', '8:Ephi']
# Revise the length of Cl lensed... 3500 vs 2500
# Revise the noise power spectra (instrumental noise + beam)?)
noisePlTemp = (4*np.pi*(TNoisePlanck/T_CMB)**2/npix)*np.ones(2501)
noisePlPol = (4*np.pi*(PNoisePlanck/T_CMB)**2/npix)*np.ones(2501)
noisePl_TTDeconv = noisePlTemp/beamPl[:, 0]**2
noisePl_EEDeconv = noisePlPol/beamPl[:, 1]**2
noisePl_BBDeconv = noisePlPol/beamPl[:, 2]**2

ell = np.arange(2, lmax+1)
ClTT = np.concatenate(([0, 0], clsLensed['2:TT']*2*np.pi/(ell*(ell+1))))
ClEE = np.concatenate(([0, 0], clsLensed['3:EE']*2*np.pi/(ell*(ell+1))))
ClTE = np.concatenate(([0, 0], clsLensed['4:TE']*2*np.pi/(ell*(ell+1))))
ClBB = np.concatenate(([0, 0], clsLensed['5:BB']*2*np.pi/(ell*(ell+1))))

maps = [hp.almxfl(tlmPl, 1 / (ClPlanck[0, :]*beamPl[:, 0]**(-1))), hp.almxfl(elmPl, 1 / (np.concatenate(([1, 1], ClPlanck[1, 2:]*beamPl[2:, 1]**(-1))))),
        hp.almxfl(blmPl, 1 / np.concatenate(([1, 1], ClPlanck[2, 2:]*beamPl[2:, 2]**(-1))))]
wcl = [ClTT, ClEE, ClBB, ClTE]
dcl = [ClTT+noisePl_TTDeconv, ClEE+noisePl_EEDeconv, ClBB+noisePl_BBDeconv, ClTE]

questhelper = lensquest.quest(maps, wcl, dcl, lmin=2, lmax=lmax_lensquest, lminCMB=2, lmaxCMB=lmax_lensquest)
# norm
norm = lensquest.quest_norm(wcl, dcl, lmin=2, lmax=lmax_lensquest, lminCMB=2, lmaxCMB=lmax_lensquest, bias=True)

# returns a_lm^Phi XY, where XY=TT,TE,EE,TB,EB or BB
phiTT = questhelper.grad('TT', norm=norm[0]['TT'], store='True')
phiTE = questhelper.grad('TE', norm=norm[0]['TE'], store='True')
phiEE = questhelper.grad('EE', norm=norm[0]['EE'], store='True')
phiTB = questhelper.grad('TB', norm=norm[0]['TB'], store='True')
phiEB = questhelper.grad('EB', norm=norm[0]['EB'], store='True')
phiBB = questhelper.grad('BB')
#
clPhiTT = hp.alm2cl(phiTT)
clPhiTE = hp.alm2cl(phiTE)
clPhiEE = hp.alm2cl(phiEE)
clPhiTB = hp.alm2cl(phiTB)
clPhiEB = hp.alm2cl(phiEB)
clPhiBB = hp.alm2cl(phiBB[0])

questhelper.make_minvariance(norm[1])
clphiMV = hp.alm2cl(questhelper.queststorage['grad']['MV'])
np.save('reconstructionSpectraPlanck', questhelper.queststorage['grad'])
np.save('NormalizationPlanck', norm[1])

if interactiveMode:
    # Deconvolution factors
    ellP = np.arange(0, lmax_lensquest+1)
    ellP_factor = (ellP * (ellP + 1)) ** 2 / (2 * np.pi)
    plt.figure()
    plt.plot(cls['1:l'][:lmax_lensquest - 1],
             cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
             (cls['1:l'][:lmax_lensquest - 1] + 1), color='k', label=r'$C_L^{\phi\phi}$ Theory')
    plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
             (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * norm[0]['TT'][2:], color='r')
    plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
             (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * norm[0]['TE'][2:], color='r')
    plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
             (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * norm[0]['EE'][2:], color='r')
    plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
             (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * norm[0]['TB'][2:], color='r')
    plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
             (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * norm[0]['EB'][2:], color='r')
    plt.plot(ellP, ellP_factor * clPhiTT, color='brown', label=r'TT')
    plt.plot(ellP, ellP_factor * clPhiTE, color='black', label=r'TE')
    plt.plot(ellP, ellP_factor * clPhiEE, color='darkorange', label=r'EE')
    plt.plot(ellP, ellP_factor * clPhiTB, color='green', label=r'TB')
    plt.plot(ellP, ellP_factor * clPhiEB, color='m', label=r'EB')
    plt.plot(ellP, ellP_factor * clphiMV, color='b', label=r'MV')
    plt.xlabel(r'$\ell$', size=18)
    plt.ylabel(r'$\frac{[\ell(\ell +1)]^2}{2\pi}C_\ell^{\phi\phi}$', size=18)
    plt.legend(fontsize=10)
    plt.xlim([2, lmax_lensquest])
    plt.semilogy()
    plt.tight_layout()
    plt.savefig('phiphiMVPlanck.pdf', dpi=1200, bbox_inches='tight')
