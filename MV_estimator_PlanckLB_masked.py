"""
Lensquest: Minimum Variance calculation of the lensing potential for simulated masked Planck and LiteBIRD combination maps.

Author: Miguel Ruiz Granda
"""

# *********************** PACKAGES ************************************************************************************
import numpy as np
from astropy.io import ascii
import healpy as hp
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
lmax_lensquest = lmax
dlmax = 1000  # lmax of the unlensed fields is lmax + dlmax (for accurate lensing)
# The deflected map is constructed by interpolation of the undeflected map, built
# at target resolution approx. 0.7∗2**facres arcmin
facres = -1

FWHM_Planck = np.radians(5/60)  # radians
TNoisePlanck = 20 / resol  # muK
PNoisePlanck = 40 / resol  # muK
FWHM_LB = np.radians(30/60)  # radians
TPNoiseLB = 2.2 / resol  # muK
# *********************************************************************************************************************

# Read the cl data files and convert them into numpy arrays. The column names are:
# colnames = ['1:l', '2:TT', '3:EE', '4:TE', '5:BB', '6:phiphi', '7:TPhi', '8:Ephi']
cls = ascii.read("base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2.dat",
                 format="commented_header", header_start=10).as_array()
clsLensed = ascii.read("base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2_lensed.dat",
                       format="commented_header", header_start=10).as_array()
ell = np.arange(2, lmax+1)
ClTT = np.concatenate(([0, 0], clsLensed['2:TT']*2*np.pi/(ell*(ell+1))))
ClEE = np.concatenate(([0, 0], clsLensed['3:EE']*2*np.pi/(ell*(ell+1))))
ClTE = np.concatenate(([0, 0], clsLensed['4:TE']*2*np.pi/(ell*(ell+1))))
ClBB = np.concatenate(([0, 0], clsLensed['5:BB']*2*np.pi/(ell*(ell+1))))

beamPlanck = hp.gauss_beam(fwhm=FWHM_Planck, lmax=lmax, pol=True)

# Read the (tlm,elm,blm) lensed coefficients generated using generateSimulationsCMB.py.
tlmPLB, elmPLB, blmPLB = np.load('teb_Planck_LiteBIRD_masked.npy')

# Angular power spectra
ClPlLB = hp.alm2cl([tlmPLB, elmPLB, blmPLB])

Cl_Theo_M = np.load('cls_theo_masked.npy')
Cl_WM_PlLB = np.load('Planck_LB_cls_masked.npy')

mask = np.load('Apodized_mask.npy')
fsky=np.mean(mask**2)**2/np.mean(mask**4)
print(fsky)

# *********************************************************************************************************************

# STEP 3: LENSQUEST: CMB lensing reconstruction ***********************************************************************

# *********************************************************************************************************************

maps = [hp.almxfl(tlmPLB, 1 / ClPlLB[0, :]), hp.almxfl(elmPLB, 1 / (np.concatenate(([1, 1], ClPlLB[1, 2:])))),
        hp.almxfl(blmPLB, 1 / np.concatenate(([1, 1], ClPlLB[2, 2:])))]
wcl = Cl_Theo_M
dcl = Cl_WM_PlLB

ell = np.arange(0, lmax+1)
plt.figure()
plt.plot(ell, ClPlLB[0, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TT')
plt.plot(ell, Cl_WM_PlLB[0, :lmax+1] * ell * (ell + 1) * T_CMB ** 2 / (2 * np.pi), label='NAMASTER TT')
plt.plot(ell, Cl_Theo_M[0, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Theo TT')
plt.plot(ell, ClPlLB[1, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map EE')
plt.plot(ell, Cl_WM_PlLB[1, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER EE')
plt.plot(ell, Cl_Theo_M[1, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Theo EE')
plt.plot(ell, ClPlLB[2, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map BB')
plt.plot(ell, Cl_WM_PlLB[2, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER BB')
plt.plot(ell, Cl_Theo_M[2, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Theo BB')
plt.plot(ell, ClPlLB[3, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TE')
plt.plot(ell, Cl_WM_PlLB[3, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER TE')
plt.plot(ell, Cl_Theo_M[3, :lmax+1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Theo TE')
plt.xlabel(r'$\ell$', size=18)
plt.ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell\ /\ \mu K^2$', size=18)
plt.legend()
# plt.semilogy()
plt.title('Planck+LiteBIRD')


questhelper = lensquest.quest(maps, wcl, dcl, lmin=2, lmax=lmax_lensquest, lminCMB=2, lmaxCMB=lmax_lensquest)
# norm
norm = lensquest.quest_norm(wcl, dcl, lmin=2, lmax=lmax_lensquest, lminCMB=2, lmaxCMB=lmax_lensquest, bias=True)

# returns a_lm^Phi XY, where XY=TT,TE,EE,TB,EB or BB
phiTT = questhelper.grad('TT', norm=norm[0]['TT']*np.sqrt(fsky), store='True')
phiTE = questhelper.grad('TE', norm=norm[0]['TE']*np.sqrt(fsky), store='True')
phiEE = questhelper.grad('EE', norm=norm[0]['EE']*np.sqrt(fsky), store='True')
phiTB = questhelper.grad('TB', norm=norm[0]['TB']*np.sqrt(fsky), store='True')
phiEB = questhelper.grad('EB', norm=norm[0]['EB']*np.sqrt(fsky), store='True')
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
np.save('reconstructionSpectraPlanckLBMasked', questhelper.queststorage['grad'])
np.save('NormalizationPlanckLBMasked', norm[1])

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
    plt.savefig('phiphiMVPlanckLBMasked.pdf', dpi=1200, bbox_inches='tight')
