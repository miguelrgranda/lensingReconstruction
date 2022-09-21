"""
Calculation of the coupling matrix from a given mask and estimation of the effect of the mask on the
angular power spectrum for the different experiments.

Author: Miguel Ruiz Granda
"""

# Import packages
import numpy as np
from astropy.io import ascii
import healpy as hp
# import os
# os.chdir('/home/miguel/Desktop/TFM')
import matplotlib.pyplot as plt
import pymaster as nmt

# Constants
nside = 2048
lmax = 2500
npix = hp.nside2npix(nside)
resol = hp.nside2resol(nside, arcmin=True)  # in arcmin
T_CMB = 2.7255e6  # in muK
FWHM_Planck = np.radians(5/60)  # radians
TNoisePlanck = 20 / resol  # muK  33/resol
PNoisePlanck = 40 / resol  # muK  70.2/resol
FWHM_LB = np.radians(30/60)  # radians
TPNoiseLB = 2.2 / resol  # muK

# Read the cl data files and convert them into numpy arrays. The column names are:
# colnames = ['1:l', '2:TT', '3:EE', '4:TE', '5:BB', '6:phiphi', '7:TPhi', '8:Ephi']
cls = ascii.read("base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2.dat",
                 format="commented_header", header_start=10).as_array()
clsLensed = ascii.read("base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2_lensed.dat",
                       format="commented_header", header_start=10).as_array()

# We convert the D_ell to C_ell
ell = np.arange(2, lmax+1)
ClTT_Theo = np.concatenate(([0, 0], clsLensed['2:TT'][:lmax-1]*2*np.pi/(ell*(ell+1))))
ClEE_Theo = np.concatenate(([0, 0], clsLensed['3:EE'][:lmax-1]*2*np.pi/(ell*(ell+1))))
ClTE_Theo = np.concatenate(([0, 0], clsLensed['4:TE'][:lmax-1]*2*np.pi/(ell*(ell+1))))
ClBB_Theo = np.concatenate(([0, 0], clsLensed['5:BB'][:lmax-1]*2*np.pi/(ell*(ell+1))))

# Noise terms for the different experiments

# Planck
beamPl = hp.gauss_beam(fwhm=FWHM_Planck, lmax=lmax, pol=True)

noisePlTemp = (4*np.pi*(TNoisePlanck/T_CMB)**2/npix)*np.ones(lmax+1)
noisePlPol = (4*np.pi*(PNoisePlanck/T_CMB)**2/npix)*np.ones(lmax+1)

noisePl_TTDeconv = noisePlTemp/beamPl[:, 0]**2
noisePl_EEDeconv = noisePlPol/beamPl[:, 1]**2
noisePl_BBDeconv = noisePlPol/beamPl[:, 2]**2

# LiteBIRD
beamLB = hp.gauss_beam(fwhm=FWHM_LB, lmax=lmax, pol=True)

noiseLB = (4*np.pi*(TPNoiseLB/T_CMB)**2/npix)*np.ones(lmax+1)

noiseLB_TTDeconv = noiseLB/beamLB[:, 0]**2
noiseLB_EEDeconv = noiseLB/beamLB[:, 1]**2
noiseLB_BBDeconv = noiseLB/beamLB[:, 2]**2

# Combination Planck + LiteBIRD
wP, wL = np.load('weights.npy')
beamPlLB = np.array([(wP[:lmax+1, 0]*beamPl[:, 0])**2+(wL[:lmax+1, 0]*beamLB[:, 0])**2+2*wP[:lmax+1, 0]*beamPl[:, 0]*wL[:lmax+1, 0]*beamLB[:, 0],
                              (wP[:lmax+1, 1]*beamPl[:, 1])**2+(wL[:lmax+1, 1]*beamLB[:, 1])**2+2*wP[:lmax+1, 1]*beamPl[:, 1]*wL[:lmax+1, 1]*beamLB[:, 1],
                              (wP[:lmax+1, 2]*beamPl[:, 2])**2+(wL[:lmax+1, 2]*beamLB[:, 2])**2+2*wP[:lmax+1, 2]*beamPl[:, 2]*wL[:lmax+1, 2]*beamLB[:, 2],
                              wP[:lmax+1, 0]*wP[:lmax+1, 1]*beamPl[:, 0]*beamPl[:, 1]+wL[:lmax+1, 0]*wL[:lmax+1, 1]*beamLB[:, 0]*beamLB[:, 1]+
                              wP[:lmax+1, 0]*wL[:lmax+1, 1]*beamPl[:, 0]*beamLB[:, 1]+wP[:lmax+1, 1]*wL[:lmax+1, 0]*beamPl[:, 1]*beamLB[:, 0]]).T

noisePLB_TTDeconv = (wP[:lmax+1, 0]**2*noisePlTemp+wL[:lmax+1, 0]**2*noiseLB)/beamPlLB[:, 0]
noisePLB_EEDeconv = (wP[:lmax+1, 1]**2*noisePlPol+wL[:lmax+1, 1]**2*noiseLB)/beamPlLB[:, 1]
noisePLB_BBDeconv = (wP[:lmax+1, 2]**2*noisePlPol+wL[:lmax+1, 2]**2*noiseLB)/beamPlLB[:, 2]

# Read mask and CMB maps and load the CMB maps from the different experiments
mask = np.load('2015_Galactic_GAL080.npy')
# mask_Temp = hp.read_map("COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits", dtype=int, verbose=False)
# mask_Pol = hp.read_map("COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits", dtype=int, verbose=False)
tlm_Pl, elm_Pl, blm_Pl = np.load('teb_Planck.npy')
tlm_LB, elm_LB, blm_LB = np.load('teb_LiteBIRD.npy')
tlm_PlLB, elm_PlLB, blm_PlLB = np.load('teb_Planck_LiteBIRD.npy')
# Deconvolve the beam
tlm_Pl, elm_Pl, blm_Pl = [hp.almxfl(tlm_Pl, 1/beamPl[:, 0]), hp.almxfl(elm_Pl, 1/beamPl[:, 1]), hp.almxfl(blm_Pl, 1/beamPl[:, 2])]
tlm_LB, elm_LB, blm_LB = [hp.almxfl(tlm_LB, 1/beamLB[:, 0]), hp.almxfl(elm_LB, 1/beamLB[:, 1]), hp.almxfl(blm_LB, 1/beamLB[:, 2])]
tlm_PlLB, elm_PlLB, blm_PlLB = [hp.almxfl(tlm_PlLB, 1/beamPlLB[:, 0]**0.5), hp.almxfl(elm_PlLB, 1/beamPlLB[:, 1]**0.5), hp.almxfl(blm_PlLB, 1/beamPlLB[:, 2]**0.5)]

Tlen_Pl, Qlen_Pl, Ulen_Pl = hp.alm2map([tlm_Pl, elm_Pl, blm_Pl], nside=nside)
Tlen_LB, Qlen_LB, Ulen_LB = hp.alm2map([tlm_LB, elm_LB, blm_LB], nside=nside)
Tlen_PlLB, Qlen_PlLB, Ulen_PlLB = hp.alm2map([tlm_PlLB, elm_PlLB, blm_PlLB], nside=nside)

# Degrading resolution (only when running in my computer). Degrading introduces an additional smoothing effect...not ideal
# for testing.
# mask_Temp = hp.ud_grade(mask_Temp, nside)
# mask_Pol = hp.ud_grade(mask_Pol, nside)

# Apodize the masks on a scale of ~1deg
mask = nmt.mask_apodization(mask, 10, apotype="C1")
np.save('Apodized_mask', mask)
# mask = np.load('Apodized_mask.npy')
fsky = np.sum(mask**4)/hp.nside2npix(nside)
print('Calculating fsky...')
print(fsky)
# mask_Temp = nmt.mask_apodization(mask_Temp, 1, apotype="C1")
# mask_Pol = nmt.mask_apodization(mask_Pol, 1, apotype="C1")

print('Apodization completed')

# Mask the CMB maps using the apodized masks
maskedPlanck = [Tlen_Pl*mask, Qlen_Pl*mask, Ulen_Pl*mask]
maskedLiteBIRD = [Tlen_LB*mask, Qlen_LB*mask, Ulen_LB*mask]
maskedPlanckLiteBIRD = [Tlen_PlLB*mask, Qlen_PlLB*mask, Ulen_PlLB*mask]
clsMasked_Planck = hp.anafast(maskedPlanck, lmax=lmax)
clsMasked_LiteBIRD = hp.anafast(maskedLiteBIRD, lmax=lmax)
clsMasked_PlanckLiteBIRD = hp.anafast(maskedPlanckLiteBIRD, lmax=lmax)
tlmM_Planck, elmM_Planck, blmM_Planck = hp.map2alm(maskedPlanck, lmax=lmax)
tlmM_LiteBIRD, elmM_LiteBIRD, blmM_LiteBIRD = hp.map2alm(maskedLiteBIRD, lmax=lmax)
tlmM_PlanckLiteBIRD, elmM_PlanckLiteBIRD, blmM_PlanckLiteBIRD = hp.map2alm(maskedPlanckLiteBIRD, lmax=lmax)
np.save('teb_Planck_masked.npy', [tlmM_Planck, elmM_Planck, blmM_Planck])
np.save('teb_LiteBIRD_masked.npy', [tlmM_LiteBIRD, elmM_LiteBIRD, blmM_LiteBIRD])
np.save('teb_Planck_LiteBIRD_masked.npy', [tlmM_PlanckLiteBIRD, elmM_PlanckLiteBIRD, blmM_PlanckLiteBIRD])

print('Masking completed')

#    Spin-0
f0 = nmt.NmtField(mask, None, spin=0, templates=None)
#    Spin-2
f2 = nmt.NmtField(mask, None, spin=2, templates=None)
# Create binning scheme. We will use 1 multipoles per bandpower.
b = nmt.NmtBin.from_lmax_linear(lmax, 1)
# We then generate an NmtWorkspace object that we use to compute and store the mode coupling matrix. Note that this
# matrix depends only on the masks of the two fields to correlate, but not on the maps themselves.

print('NMT fields are created')

# Two spin-0 fields: n_cls=1, [C_T1T2]
TT = nmt.NmtWorkspace()
TT.compute_coupling_matrix(f0, f0, b)

print('First computation of a coupling matrix.')

# One spin-0 field and one spin>0 field: n_cls=2, [C_TE,C_TB]
TETB = nmt.NmtWorkspace()
TETB.compute_coupling_matrix(f0, f2, b)

# Two spin>0 fields: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
EEEBBB = nmt.NmtWorkspace()
EEEBBB.compute_coupling_matrix(f2, f2, b)

print('Finished computing all the coupling matrices.')

# Multiply the angular power spectra by the coupling matrix to take into account the effects od the mask.
# Theoretical angular power spectra.
# It is not a problem to give a longer C_ell array, couple_cell will automatically shorten it

zeros = np.zeros_like(ClEE_Theo)
ClTT_Theo_M = TT.couple_cell([ClTT_Theo])
ClTETB_Theo_M = TETB.couple_cell([ClTE_Theo, zeros])
ClEEEBBB_Theo_M = EEEBBB.couple_cell([ClEE_Theo, zeros, zeros, ClBB_Theo])

# Power spectra used in the Wiener-filter of the input fields

ClTT_WM_Pl = TT.couple_cell([ClTT_Theo+noisePl_TTDeconv])
ClTETB_WM_Pl = TETB.couple_cell([ClTE_Theo, zeros])
ClEEEBBB_WM_Pl = EEEBBB.couple_cell([ClEE_Theo+noisePl_EEDeconv, zeros, zeros, ClBB_Theo+noisePl_BBDeconv])

ClTT_WM_LB = TT.couple_cell([ClTT_Theo+noiseLB_TTDeconv])
ClTETB_WM_LB = TETB.couple_cell([ClTE_Theo, zeros])
ClEEEBBB_WM_LB = EEEBBB.couple_cell([ClEE_Theo+noiseLB_EEDeconv, zeros, zeros, ClBB_Theo+noiseLB_BBDeconv])

ClTT_WM_PlLB = TT.couple_cell([ClTT_Theo+noisePLB_TTDeconv])
ClTETB_WM_PlLB = TETB.couple_cell([ClTE_Theo, zeros])
ClEEEBBB_WM_PlLB = EEEBBB.couple_cell([ClEE_Theo+noisePLB_EEDeconv, zeros, zeros, ClBB_Theo+noisePLB_BBDeconv])

# Save the cls in a npy file
np.save('cls_theo_masked.npy', [ClTT_Theo_M[0], ClEEEBBB_Theo_M[0], ClEEEBBB_Theo_M[3], ClTETB_Theo_M[0]])
np.save('Planck_cls_masked.npy', [ClTT_WM_Pl[0], ClEEEBBB_WM_Pl[0], ClEEEBBB_WM_Pl[3], ClTETB_WM_Pl[0]])
np.save('LB_cls_masked.npy', [ClTT_WM_LB[0], ClEEEBBB_WM_LB[0], ClEEEBBB_WM_LB[3], ClTETB_WM_LB[0]])
np.save('Planck_LB_cls_masked.npy', [ClTT_WM_PlLB[0], ClEEEBBB_WM_PlLB[0], ClEEEBBB_WM_PlLB[3], ClTETB_WM_PlLB[0]])



ell = np.arange(0, lmax+1)
plt.figure()
plt.plot(ell, clsMasked_Planck[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TT')
plt.plot(ell, ClTT_WM_Pl[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER TT')
plt.plot(ell, clsMasked_Planck[1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map EE')
plt.plot(ell, ClEEEBBB_WM_Pl[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER EE')
plt.plot(ell, clsMasked_Planck[2]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map BB')
plt.plot(ell, ClEEEBBB_WM_Pl[3]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER BB')
plt.plot(ell, clsMasked_Planck[3]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TE')
plt.plot(ell, ClTETB_WM_Pl[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER TE')
plt.xlabel(r'$\ell$', size=18)
plt.ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell\ /\ \mu K^2$', size=18)
plt.title('Planck')
plt.semilogy()
plt.legend()
plt.savefig('Planck_masked.pdf')

plt.figure()
plt.plot(ell, clsMasked_LiteBIRD[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TT')
plt.plot(ell, ClTT_WM_LB[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER TT')
plt.plot(ell, clsMasked_LiteBIRD[1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map EE')
plt.plot(ell, ClEEEBBB_WM_LB[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER EE')
plt.plot(ell, clsMasked_LiteBIRD[2]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map BB')
plt.plot(ell, ClEEEBBB_WM_LB[3]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER BB')
plt.plot(ell, clsMasked_LiteBIRD[3]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TE')
plt.plot(ell, ClTETB_WM_LB[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER TE')
plt.xlabel(r'$\ell$', size=18)
plt.ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell\ /\ \mu K^2$', size=18)
plt.legend()
plt.semilogy()
plt.title('LiteBIRD')
plt.savefig('LiteBIRD_masked.pdf')

plt.figure()
plt.plot(ell, clsMasked_PlanckLiteBIRD[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TT')
plt.plot(ell, ClTT_WM_PlLB[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER TT')
plt.plot(ell, clsMasked_PlanckLiteBIRD[1]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map EE')
plt.plot(ell, ClEEEBBB_WM_PlLB[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER EE')
plt.plot(ell, clsMasked_PlanckLiteBIRD[2]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map BB')
plt.plot(ell, ClEEEBBB_WM_PlLB[3]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER BB')
plt.plot(ell, clsMasked_PlanckLiteBIRD[3]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='Map TE')
plt.plot(ell, ClTETB_WM_PlLB[0]*ell*(ell+1)*T_CMB**2/(2*np.pi), label='NAMASTER TE')
plt.xlabel(r'$\ell$', size=18)
plt.ylabel(r'$\frac{\ell(\ell +1)}{2\pi}C_\ell\ /\ \mu K^2$', size=18)
plt.legend()
plt.semilogy()
plt.title('Planck+LiteBIRD')
plt.savefig('Planck_LiteBIRD_masked.pdf')

print('NAMASTER execution is finished.')
