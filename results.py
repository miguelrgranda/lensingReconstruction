'''
This script calculates the MV weights, the signal-to-noise ratio (SNR) and plots the reconstructed lensing power
spectrum and the Wiener-filtered maps for the different quadratic estimators. The data used are the full-sky simulations
for Planck, LiteBIRD and the combination of both of them.
Author: Miguel Ruiz Granda
'''
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import healpy as hp
import quadprog
import os
os.chdir('/home/miguel/Desktop/TFM_2')

# *********************** CONSTANTS ***********************************************************************************
T_CMB = 2.7255e6  # in muK
nside = 2048  #  2048
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
FWHM_LB = np.radians(30/60)  # radians
TPNoiseLB = 2.2 / resol  # muK
# *********************************************************************************************************************

# Functions to calculate the minimum variance estimator
def get_cross(spectra, i, j):
    nspec = len(spectra)
    return spectra[min(i, j)] + spectra[max(i, j)]


def getweights(nl, spectra):
    """
    This function calculates the weights for the minimum variance estimator using the equations of Hu and White. The
    only restriction imposed is sum(w_i)=1.
    """
    lmax = len(list(nl.values())[0])
    l = np.arange(lmax)
    nmv = np.zeros(lmax)

    weight = {}
    for arg in spectra:
        weight[arg] = np.zeros(lmax)

    nspec = len(spectra)
    for L in range(2, lmax):
        mat = np.zeros((nspec, nspec))
        for i in range(nspec):
            for j in range(nspec):
                try:
                    mat[i, j] = nl[get_cross(spectra, i, j)][L]  # +self.n1[dh.get_cross(self.spectra,i,j)][L] #(1.+1./self.num_sims_MF)*
                except KeyError:
                    mat[i, j] = 0.0
        try:
            mate = np.linalg.solve(mat, np.ones(nspec))
            nmv[L] = 1. / sum(mate)
        except:
            print('Matrix singular for L=' + str(L))
            nmv[L] = 0.
        for i in range(nspec):
            weight[spectra[i]][L] = nmv[L] * mate[i]

    return nmv, weight

def getweights_quad(nl, spectra):
    """
    This function calculates the weights for the minimum variance estimator using a quadratic programing solver and
    taking into account that the weights have to verify 0<=w_i<=1 and sum(w_i)=1.
    """
    lmax = len(list(nl.values())[0])
    l = np.arange(lmax)
    nmv = np.zeros(lmax)

    weight = {}
    for arg in spectra:
        weight[arg] = np.zeros(lmax)

    nspec = len(spectra)

    # Matrix defining the constraints under which we want to minimize the quadratic function
    constrains = np.zeros((2*nspec+1, nspec))
    constrains[0, :] = np.ones(nspec)
    for i in range(1, nspec+1):
        constrains[i, i-1] = 1
    constrains[nspec+1:, :] = -constrains[1:nspec+1, :]

    # Vector defining the constraints
    b = np.concatenate((np.array([1]), np.zeros(nspec), -np.ones(nspec)))

    for L in range(2, lmax):
        mat = np.zeros((nspec, nspec))
        for i in range(nspec):
            for j in range(nspec):
                try:
                    mat[i, j] = nl[get_cross(spectra, i, j)][L]  # +self.n1[dh.get_cross(self.spectra,i,j)][L] #(1.+1./self.num_sims_MF)*
                except KeyError:
                    mat[i, j] = 0.0
        try:
            sol = quadprog.solve_qp(2*mat, np.zeros(nspec), constrains.T, b, meq=1)
            nmv[L] = sol[1]
        except:
            print('Quadprog error for L=' + str(L))
            nmv[L] = 0.
        for i in range(nspec):
            weight[spectra[i]][L] = sol[0][i]

    return nmv, weight

def make_minvariance(dictQuest, weights):
    """
        dictQuest: dictionary containing alm lensing reconstruction for TT, EE, TE, TB and EB.
        weight: dictionary of weights for the lensing reconstruction TT, EE, TE, TB and EB.
    """
    spectra = list(dictQuest.keys())[:-1]
    almMV = 0
    for spec in spectra:
        almMV += hp.almxfl(dictQuest[spec], weights[spec])

    dictQuest["MV"] = almMV

def snr(Cl_rec, Nl_rec):
    """
        Cl_rec: lensing reconstruction power spectrum
        Nl_rec: reconstruction noise power spectrum
    """
    aux = np.zeros_like(Cl_rec)
    for l in range(2, len(Cl_rec)):
        aux[l] = aux[l - 1] + (l + 0.5) * (Cl_rec[l] - Nl_rec[l])**2 / Cl_rec[l]**2
    return np.sqrt(aux)


def snr_correlation(Cl_in, Cl_rec, Cl_recin):
    """
        Cl_in: input lensing angular power spectrum
        Cl_rec: lensing reconstruction power spectrum
        Cl_recin: cross-correlation between the input and reconstruction lensing power spectrum
    """
    aux = np.zeros_like(Cl_rec)
    for l in range(2, len(Cl_rec)):
        aux[l] = aux[l - 1] + (2 * l + 1) * Cl_recin[l] ** 2 / (Cl_rec[l] * Cl_in[l] + Cl_recin[l] ** 2)
    return np.sqrt(aux)


def noise_min_variance(nl, weights, spectra):
    nspec = len(spectra)
    lmax = len(list(nl.values())[0])
    weight_MV = np.zeros(lmax)

    for L in range(2, lmax):
        for i in range(nspec):
            for j in range(nspec):
                try:
                    weight_MV[L] += weights[spectra[i]][L]*weights[spectra[j]][L]*nl[get_cross(spectra, i, j)][L]
                except KeyError:
                    weight_MV[L] += 0.0

    return weight_MV

def alm_change_lmax(alm, new_lmax):
    lmmap = hp.Alm.getidx(hp.Alm.getlmax(len(alm)), *hp.Alm.getlm(new_lmax, np.arange(hp.Alm.getsize(new_lmax))))
    return alm[lmmap]

def wiener_filter(alm, Cl_theory, Cl_exp):
    return hp.almxfl(alm, np.concatenate(([1, 1], Cl_theory[2:]/Cl_exp[2:])))

def wiener_filter_plot(Cl_theory, Cl_exp):
    return np.concatenate(([0, 0], Cl_theory[2:]/Cl_exp[2:]))

# Read the data files and convert them into numpy arrays. The column names are:
# colnames = ['1:l', '2:TT', '3:EE', '4:TE', '5:BB', '6:phiphi', '7:TPhi', '8:Ephi']
cls = ascii.read("/home/miguel/Desktop/TFM_2/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl2.dat",
                 format="commented_header", header_start=10).as_array()
tlm_unl, elm_unl, blm_unl, plm = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/teb_unlensed.npy')
dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + dlmax + 1, dtype=float) * np.arange(1, lmax + dlmax + 2)))
dlm = alm_change_lmax(dlm, lmax)
plm = alm_change_lmax(plm, lmax)

# Planck experiment
dictQuestPlanck = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/reconstructionSpectraPlanck.npy', allow_pickle=True).item()
normPlanck = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/NormalizationPlanck.npy', allow_pickle=True).item()
nmvQPlanck, weightsQPlanck = getweights_quad(normPlanck, list(dictQuestPlanck.keys())[:-1])
nmvPlanck, weightsPlanck = getweights(normPlanck, list(dictQuestPlanck.keys())[:-1])
make_minvariance(dictQuestPlanck, weightsPlanck)

# LiteBIRD experiment
dictQuestLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/reconstructionSpectraLB.npy', allow_pickle=True).item()
normLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/NormalizationLB.npy', allow_pickle=True).item()
nmvQLB, weightsQLB = getweights_quad(normLB, list(dictQuestLB.keys())[:-1])
nmvLB, weightsLB = getweights(normLB, list(dictQuestLB.keys())[:-1])
make_minvariance(dictQuestLB, weightsLB)

# Combination of Planck and LiteBIRD
dictQuestPlLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/reconstructionSpectraPlLB.npy', allow_pickle=True).item()
normPlLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/NormalizationPlLB.npy', allow_pickle=True).item()
nmvQPlLB, weightsQPlLB = getweights_quad(normPlLB, list(dictQuestPlLB.keys())[:-1])
nmvPlLB, weightsPlLB = getweights(normPlLB, list(dictQuestPlLB.keys())[:-1])
make_minvariance(dictQuestPlLB, weightsPlLB)


# Plot weights using Hu and White minimum-variance estimator for the different experiments
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].plot(weightsPlanck['TT'], label='TT')
ax[0].plot(weightsPlanck['EE'], label='EE')
ax[0].plot(weightsPlanck['TE'], label='TE')
ax[0].plot(weightsPlanck['TB'], label='TB')
ax[0].plot(weightsPlanck['EB'], label='EB')
ax[0].title.set_text('Planck experiment')
ax[0].set_ylabel(r'Weights', size=18)
ax[1].plot(weightsLB['TT'], label='TT')
ax[1].plot(weightsLB['EE'], label='EE')
ax[1].plot(weightsLB['TE'], label='TE')
ax[1].plot(weightsLB['TB'], label='TB')
ax[1].plot(weightsLB['EB'], label='EB')
ax[1].title.set_text('LiteBIRD experiment')
plt.legend()
ax[2].plot(weightsPlLB['TT'], label='TT')
ax[2].plot(weightsPlLB['EE'], label='EE')
ax[2].plot(weightsPlLB['TE'], label='TE')
ax[2].plot(weightsPlLB['TB'], label='TB')
ax[2].plot(weightsPlLB['EB'], label='EB')
ax[2].title.set_text('Planck and LiteBIRD combination')
for ax in ax.reshape(-1):
    ax.set_xlim([2, lmax])
    ax.set_xlabel(r'$\ell$', size=18)
    ax.legend(fontsize=14, loc='center right')
fig.tight_layout()

# Plot weights using the minimum-variance estimator for the different experiments calculated using the quadratic solver
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].plot(weightsQPlanck['TT'], label='TT')
ax[0].plot(weightsQPlanck['EE'], label='EE')
ax[0].plot(weightsQPlanck['TE'], label='TE')
ax[0].plot(weightsQPlanck['TB'], label='TB')
ax[0].plot(weightsQPlanck['EB'], label='EB')
ax[0].title.set_text('Planck experiment')
ax[0].set_ylabel(r'Weights', size=18)
ax[1].plot(weightsQLB['TT'], label='TT')
ax[1].plot(weightsQLB['EE'], label='EE')
ax[1].plot(weightsQLB['TE'], label='TE')
ax[1].plot(weightsQLB['TB'], label='TB')
ax[1].plot(weightsQLB['EB'], label='EB')
ax[1].title.set_text('LiteBIRD experiment')
plt.legend()
ax[2].plot(weightsQPlLB['TT'], label='TT')
ax[2].plot(weightsQPlLB['EE'], label='EE')
ax[2].plot(weightsQPlLB['TE'], label='TE')
ax[2].plot(weightsQPlLB['TB'], label='TB')
ax[2].plot(weightsQPlLB['EB'], label='EB')
ax[2].title.set_text('Planck and LiteBIRD combination')
for ax in ax.reshape(-1):
    ax.set_xlim([2, lmax])
    ax.set_xlabel(r'$\ell$', size=18)
    ax.legend(fontsize=14, loc='center right')
fig.tight_layout()


clPhiPhi = hp.alm2cl(dlm)
ellP = np.arange(0, lmax_lensquest+1)
ellP_factor = (ellP * (ellP + 1)) ** 2 / (2 * np.pi)
# Planck
clPhiTTPlanck = hp.alm2cl(dictQuestPlanck['TT'])
clPhiTEPlanck = hp.alm2cl(dictQuestPlanck['TE'])
clPhiEEPlanck = hp.alm2cl(dictQuestPlanck['EE'])
clPhiTBPlanck = hp.alm2cl(dictQuestPlanck['TB'])
clPhiEBPlanck = hp.alm2cl(dictQuestPlanck['EB'])
clPhiMVPlanck = hp.alm2cl(dictQuestPlanck['MV'])

clCrossTTPlanck = hp.alm2cl(dictQuestPlanck['TT'], plm)
clCrossTEPlanck = hp.alm2cl(dictQuestPlanck['TE'], plm)
clCrossEEPlanck = hp.alm2cl(dictQuestPlanck['EE'], plm)
clCrossTBPlanck = hp.alm2cl(dictQuestPlanck['TB'], plm)
clCrossEBPlanck = hp.alm2cl(dictQuestPlanck['EB'], plm)
clCrossMVPlanck = hp.alm2cl(dictQuestPlanck['MV'], plm)

plt.figure()
plt.plot(cls['1:l'][:lmax_lensquest - 1],
         cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1), color='k', label=r'$C_\ell^{\phi\phi}$ Theory')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TTTT'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TETE'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['EEEE'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TBTB'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['EBEB'][2:], color='r')
plt.plot(ellP, ellP_factor * clPhiTTPlanck, color='b', label=r'TT')
plt.plot(ellP, ellP_factor * clPhiTEPlanck, color='red', label=r'TE')
plt.plot(ellP, ellP_factor * clPhiEEPlanck, color='darkorange', label=r'EE')
plt.plot(ellP, ellP_factor * clPhiTBPlanck, color='green', label=r'TB')
plt.plot(ellP, ellP_factor * clPhiEBPlanck, color='m', label=r'EB')
plt.plot(ellP, ellP_factor * clPhiMVPlanck, color='brown', label=r'MV')
plt.xlabel(r'$\ell$', size=30)
plt.ylabel(r'$\frac{[\ell(\ell +1)]^2}{2\pi}C_\ell^{\phi\phi}$', size=30)
plt.legend(fontsize=20, loc='lower right')
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.xlim([2, lmax_lensquest])
plt.ylim([1e-9, 1e-4])
plt.semilogx()
plt.semilogy()
plt.title('Planck experiment', fontsize=30)
plt.tight_layout()

fig, axs = plt.subplots(3, 2, figsize=(10, 6), constrained_layout=True)
fig.suptitle('Planck experiment', fontsize=16)
axs[0, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiTTPlanck[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TTTT'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TTTT'][2:]), color='brown', label=r'TT')
axs[1, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiEEPlanck[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['EEEE'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['EEEE'][2:]), color='brown', label=r'EE')
axs[2, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiTEPlanck[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TETE'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TETE'][2:]), color='brown', label=r'TE')
axs[0, 1].plot(ellP[2:], (ellP_factor[2:] * clPhiTBPlanck[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TBTB'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['TBTB'][2:]), color='brown', label=r'TB')
axs[1, 1].plot(ellP[2:], (ellP_factor[2:] * clPhiEBPlanck[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['EBEB'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlanck['EBEB'][2:]), color='brown', label=r'EB')
for ax in axs.flat:
    ax.set_ylim([-0.25, 0.25])
    ax.axhline(0., c='k')
    ax.legend()

# LiteBIRD
clPhiTTLB = hp.alm2cl(dictQuestLB['TT'])
clPhiTELB = hp.alm2cl(dictQuestLB['TE'])
clPhiEELB = hp.alm2cl(dictQuestLB['EE'])
clPhiTBLB = hp.alm2cl(dictQuestLB['TB'])
clPhiEBLB = hp.alm2cl(dictQuestLB['EB'])
clPhiMVLB = hp.alm2cl(dictQuestLB['MV'])

clCrossTTLB = hp.alm2cl(dictQuestLB['TT'], plm)
clCrossTELB = hp.alm2cl(dictQuestLB['TE'], plm)
clCrossEELB = hp.alm2cl(dictQuestLB['EE'], plm)
clCrossTBLB = hp.alm2cl(dictQuestLB['TB'], plm)
clCrossEBLB = hp.alm2cl(dictQuestLB['EB'], plm)
clCrossMVLB = hp.alm2cl(dictQuestLB['MV'], plm)

plt.figure()
plt.plot(cls['1:l'][:lmax_lensquest - 1],
         cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1), color='k', label=r'$C_\ell^{\phi\phi}$ Theory')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TTTT'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TETE'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['EEEE'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TBTB'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['EBEB'][2:], color='r')
plt.plot(ellP, ellP_factor * clPhiTTLB, color='b', label=r'TT')
plt.plot(ellP, ellP_factor * clPhiTELB, color='red', label=r'TE')
plt.plot(ellP, ellP_factor * clPhiEELB, color='darkorange', label=r'EE')
plt.plot(ellP, ellP_factor * clPhiTBLB, color='green', label=r'TB')
plt.plot(ellP, ellP_factor * clPhiEBLB, color='m', label=r'EB')
plt.plot(ellP, ellP_factor * clPhiMVLB, color='brown', label=r'MV')
plt.xlabel(r'$\ell$', size=30)
plt.ylabel(r'$\frac{[\ell(\ell +1)]^2}{2\pi}C_\ell^{\phi\phi}$', size=30)
plt.legend(fontsize=20, loc='center right')
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.xlim([2, lmax_lensquest])
plt.ylim([1e-9, 1e-4])
plt.semilogx()
plt.semilogy()
plt.title('LiteBIRD experiment', fontsize=30)
plt.tight_layout()

fig, axs = plt.subplots(3, 2, figsize=(10, 6), constrained_layout=True)
fig.suptitle('LiteBIRD experiment', fontsize=16)
axs[0, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiTTLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TTTT'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TTTT'][2:]), color='brown', label=r'TT')
axs[1, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiEELB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['EEEE'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['EEEE'][2:]), color='brown', label=r'EE')
axs[2, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiTELB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TETE'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TETE'][2:]), color='brown', label=r'TE')
axs[0, 1].plot(ellP[2:], (ellP_factor[2:] * clPhiTBLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TBTB'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['TBTB'][2:]), color='brown', label=r'TB')
axs[1, 1].plot(ellP[2:], (ellP_factor[2:] * clPhiEBLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['EBEB'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normLB['EBEB'][2:]), color='brown', label=r'EB')
for ax in axs.flat:
    ax.set_ylim([-0.25, 0.25])
    ax.axhline(0., c='k')
    ax.legend()

# Planck+LB combination
clPhiTTPlLB = hp.alm2cl(dictQuestPlLB['TT'])
clPhiTEPlLB = hp.alm2cl(dictQuestPlLB['TE'])
clPhiEEPlLB = hp.alm2cl(dictQuestPlLB['EE'])
clPhiTBPlLB = hp.alm2cl(dictQuestPlLB['TB'])
clPhiEBPlLB = hp.alm2cl(dictQuestPlLB['EB'])
clPhiMVPlLB = hp.alm2cl(dictQuestPlLB['MV'])

clCrossTTPlLB = hp.alm2cl(dictQuestPlLB['TT'], plm)
clCrossTEPlLB = hp.alm2cl(dictQuestPlLB['TE'], plm)
clCrossEEPlLB = hp.alm2cl(dictQuestPlLB['EE'], plm)
clCrossTBPlLB = hp.alm2cl(dictQuestPlLB['TB'], plm)
clCrossEBPlLB = hp.alm2cl(dictQuestPlLB['EB'], plm)
clCrossMVPlLB = hp.alm2cl(dictQuestPlLB['MV'], plm)

plt.figure()
plt.plot(cls['1:l'][:lmax_lensquest - 1],
         cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1), color='k', label=r'$C_\ell^{\phi\phi}$ Theory')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TTTT'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TETE'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['EEEE'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TBTB'][2:], color='r')
# plt.plot(cls['1:l'][:lmax_lensquest - 1], cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
#          (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['EBEB'][2:], color='r')
plt.plot(ellP, ellP_factor * clPhiTTPlLB, color='b', label=r'TT')
plt.plot(ellP, ellP_factor * clPhiTEPlLB, color='red', label=r'TE')
plt.plot(ellP, ellP_factor * clPhiEEPlLB, color='darkorange', label=r'EE')
plt.plot(ellP, ellP_factor * clPhiTBPlLB, color='green', label=r'TB')
plt.plot(ellP, ellP_factor * clPhiEBPlLB, color='m', label=r'EB')
plt.plot(ellP, ellP_factor * clPhiMVPlLB, color='brown', label=r'MV')
plt.xlabel(r'$\ell$', size=30)
plt.ylabel(r'$\frac{[\ell(\ell +1)]^2}{2\pi}C_\ell^{\phi\phi}$', size=30)
plt.legend(fontsize=20, loc='lower right')
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.xlim([2, lmax_lensquest])
plt.ylim([1e-9, 1e-4])
# plt.semilogx()
plt.semilogy()
plt.title('Planck and LiteBIRD combination', fontsize=30)
plt.tight_layout()

fig, axs = plt.subplots(3, 2, figsize=(10, 6), constrained_layout=True)
fig.suptitle('Planck and LiteBIRD combination', fontsize=16)
axs[0, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiTTPlLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TTTT'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TTTT'][2:]), color='brown', label=r'TT')
axs[1, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiEEPlLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['EEEE'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['EEEE'][2:]), color='brown', label=r'EE')
axs[2, 0].plot(ellP[2:], (ellP_factor[2:] * clPhiTEPlLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TETE'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TETE'][2:]), color='brown', label=r'TE')
axs[0, 1].plot(ellP[2:], (ellP_factor[2:] * clPhiTBPlLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TBTB'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['TBTB'][2:]), color='brown', label=r'TB')
axs[1, 1].plot(ellP[2:], (ellP_factor[2:] * clPhiEBPlLB[2:]-(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['EBEB'][2:]))/(cls['6:phiphi'][:lmax_lensquest - 1] * cls['1:l'][:lmax_lensquest - 1] *
         (cls['1:l'][:lmax_lensquest - 1] + 1) + ellP_factor[2:] * normPlLB['EBEB'][2:]), color='brown', label=r'EB')
for ax in axs.flat:
    ax.set_ylim([-0.25, 0.25])
    ax.axhline(0., c='k')
    ax.legend()

# SNR (NORMAL) (Cl_rec[l] - Nl_rec[l])**2 / Cl_rec[l]**2
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
# plt.figure()
# plt.plot(((clPhiTTPlanck-normPlanck['TTTT'])))
# plt.plot(((clpp*2*np.pi/(ellP*(ellP+1)))))
# plt.plot(((ellP+0.5)*(clpp*2*np.pi/(ellP*(ellP+1)))**2/(clpp*2*np.pi/(ellP*(ellP+1))+normPlanck['TTTT'])**2)[2:])
# plt.plot((ellP[2:]+0.5)*((clPhiTTPlanck-normPlanck['TTTT'])**2/clPhiTTPlanck**2)[2:])
# plt.plot(((ellP+0.5)*(clpp*2*np.pi/(ellP*(ellP+1)))**2/(clpp*2*np.pi/(ellP*(ellP+1))+normPlanck['TTTT'])**2)[2:])
ax[0].plot(snr(clPhiTTPlanck, normPlanck['TTTT']), label='TT')
ax[0].plot(snr(clPhiEEPlanck, normPlanck['EEEE']), label='EB')
ax[0].plot(snr(clPhiTEPlanck, normPlanck['TETE']), label='TE')
ax[0].plot(snr(clPhiTBPlanck, normPlanck['TBTB']), label='TB')
ax[0].plot(snr(clPhiEBPlanck, normPlanck['EBEB']), label='EB')
ax[0].plot(snr(clPhiMVPlanck, noise_min_variance(normPlanck, weightsQPlanck, list(dictQuestPlanck.keys())[:-1])), label='MV')
ax[0].title.set_text('Planck experiment')
ax[0].set_ylabel(r'$S/N$', size=18)
ax[1].plot(snr(clPhiTTLB, normLB['TTTT']), label='TT')
ax[1].plot(snr(clPhiEELB, normLB['EEEE']), label='EB')
ax[1].plot(snr(clPhiTELB, normLB['TETE']), label='TE')
ax[1].plot(snr(clPhiTBLB, normLB['TBTB']), label='TB')
ax[1].plot(snr(clPhiEBLB, normLB['EBEB']), label='EB')
ax[1].plot(snr(clPhiMVLB, noise_min_variance(normLB, weightsQLB, list(dictQuestLB.keys())[:-1])), label='MV')
ax[1].title.set_text('LiteBIRD experiment')
plt.legend()
ax[2].plot(snr(clPhiTTPlLB, normPlLB['TTTT']), label='TT')
ax[2].plot(snr(clPhiEEPlLB, normPlLB['EEEE']), label='EB')
ax[2].plot(snr(clPhiTEPlLB, normPlLB['TETE']), label='TE')
ax[2].plot(snr(clPhiTBPlLB, normPlLB['TBTB']), label='TB')
ax[2].plot(snr(clPhiEBPlLB, normPlLB['EBEB']), label='EB')
ax[2].plot(snr(clPhiMVPlLB, noise_min_variance(normPlLB, weightsQPlLB, list(dictQuestPlLB.keys())[:-1])), label='MV')
ax[2].title.set_text('Planck and LiteBIRD combination')
for ax in ax.reshape(-1):
    ax.set_xlim([2, lmax])
    ax.set_xlabel(r'$\ell$', size=18)
    # ax.set_ylabel(r'$S/N$', size=18)
    ax.legend(fontsize=14, loc='lower right')
fig.tight_layout()

clpp = np.concatenate(([0, 0], cls['6:phiphi'][:lmax_lensquest - 1]*2*np.pi/(cls['1:l'][:lmax_lensquest - 1]*(cls['1:l'][:lmax_lensquest - 1]+1))))
# SNR (NORMAL) theory
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
# plt.figure()
# plt.plot(clpp*2*np.pi/(ellP*(ellP+1))+normPlanck['TTTT'])
# plt.plot(clPhiTTPlanck)
ax[0].plot(snr(clpp+normPlanck['TTTT'], normPlanck['TTTT']), label='TT')
ax[0].plot(snr(clpp+normPlanck['EEEE'], normPlanck['EEEE']), label='EE')
ax[0].plot(snr(clpp+normPlanck['TETE'], normPlanck['TETE']), label='TE')
ax[0].plot(snr(clpp+normPlanck['TBTB'], normPlanck['TBTB']), label='TB')
ax[0].plot(snr(clpp+normPlanck['EBEB'], normPlanck['EBEB']), label='EB')
ax[0].plot(snr(clpp+noise_min_variance(normPlanck, weightsQPlanck, list(dictQuestPlanck.keys())[:-1]),
               noise_min_variance(normPlanck, weightsQPlanck, list(dictQuestPlanck.keys())[:-1])), label='MV')
ax[0].title.set_text('Planck experiment')
ax[0].set_ylabel(r'$S/N$', size=18)
ax[1].plot(snr(clpp+normLB['TTTT'], normLB['TTTT']), label='TT')
ax[1].plot(snr(clpp+normLB['EEEE'], normLB['EEEE']), label='EE')
ax[1].plot(snr(clpp+normLB['TETE'], normLB['TETE']), label='TE')
ax[1].plot(snr(clpp+normLB['TBTB'], normLB['TBTB']), label='TB')
ax[1].plot(snr(clpp+normLB['EBEB'], normLB['EBEB']), label='EB')
ax[1].plot(snr(clpp+noise_min_variance(normLB, weightsQLB, list(dictQuestLB.keys())[:-1]),
               noise_min_variance(normLB, weightsQLB, list(dictQuestLB.keys())[:-1])), label='MV')
ax[1].title.set_text('LiteBIRD experiment')
plt.legend()
ax[2].plot(snr(clpp+normPlLB['TTTT'], normPlLB['TTTT']), label='TT')
ax[2].plot(snr(clpp+normPlLB['EEEE'], normPlLB['EEEE']), label='EE')
ax[2].plot(snr(clpp+normPlLB['TETE'], normPlLB['TETE']), label='TE')
ax[2].plot(snr(clpp+normPlLB['TBTB'], normPlLB['TBTB']), label='TB')
ax[2].plot(snr(clpp+normPlLB['EBEB'], normPlLB['EBEB']), label='EB')
ax[2].plot(snr(clpp+noise_min_variance(normPlLB, weightsQPlLB, list(dictQuestPlLB.keys())[:-1]),
               noise_min_variance(normPlLB, weightsQPlLB, list(dictQuestPlLB.keys())[:-1])), label='MV')
ax[2].title.set_text('Planck and LiteBIRD combination')
for ax in ax.reshape(-1):
    ax.set_xlim([2, lmax])
    ax.set_ylim([-6, 136])
    ax.set_xlabel(r'$\ell$', size=18)
    # ax.set_ylabel(r'$S/N$', size=18)
    ax.legend(fontsize=12, loc='upper right')
fig.tight_layout()

# SNR (CORRELATION)
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].plot(snr_correlation(clPhiPhi, clPhiTTPlanck, clCrossTTPlanck), label='TT')
ax[0].plot(snr_correlation(clPhiPhi, clPhiEEPlanck, clCrossEEPlanck), label='EE')
ax[0].plot(snr_correlation(clPhiPhi, clPhiTEPlanck, clCrossTEPlanck), label='TE')
ax[0].plot(snr_correlation(clPhiPhi, clPhiTBPlanck, clCrossTBPlanck), label='TB')
ax[0].plot(snr_correlation(clPhiPhi, clPhiEBPlanck, clCrossEBPlanck), label='EB')
ax[0].plot(snr_correlation(clPhiPhi, clPhiMVPlanck, clCrossMVPlanck), label='MV')
ax[0].title.set_text('Planck experiment')
ax[0].set_ylabel(r'$S/N\ correlation$', size=18)
ax[1].plot(snr_correlation(clPhiPhi, clPhiTTLB, clCrossTTLB), label='TT')
ax[1].plot(snr_correlation(clPhiPhi, clPhiEELB, clCrossEELB), label='EE')
ax[1].plot(snr_correlation(clPhiPhi, clPhiTELB, clCrossTELB), label='TE')
ax[1].plot(snr_correlation(clPhiPhi, clPhiTBLB, clCrossTBLB), label='TB')
ax[1].plot(snr_correlation(clPhiPhi, clPhiEBLB, clCrossEBLB), label='EB')
ax[1].plot(snr_correlation(clPhiPhi, clPhiMVLB, clCrossMVLB), label='MV')
ax[1].title.set_text('LiteBIRD experiment')
plt.legend()
ax[2].plot(snr_correlation(clPhiPhi, clPhiTTPlLB, clCrossTTPlLB), label='TT')
ax[2].plot(snr_correlation(clPhiPhi, clPhiEEPlLB, clCrossEEPlLB), label='EE')
ax[2].plot(snr_correlation(clPhiPhi, clPhiTEPlLB, clCrossTEPlLB), label='TE')
ax[2].plot(snr_correlation(clPhiPhi, clPhiTBPlLB, clCrossTBPlLB), label='TB')
ax[2].plot(snr_correlation(clPhiPhi, clPhiEBPlLB, clCrossEBPlLB), label='EB')
ax[2].plot(snr_correlation(clPhiPhi, clPhiMVPlLB, clCrossMVPlLB), label='MV')
ax[2].title.set_text('Planck and LiteBIRD combination')
for ax in ax.reshape(-1):
    ax.set_xlim([2, lmax])
    ax.set_ylim([-0.1, 2.6])
    ax.set_xlabel(r'$\ell$', size=18)
    # ax.set_ylabel(r'$S/N\ correlation$', size=18)
    ax.legend(fontsize=12, loc='lower right')
fig.tight_layout()

# Patch sky for the minimum variance estimator
# originalMap = hp.alm2map(wiener_filter(dlm, clPhiPhi, (ellP * (ellP + 1))*clPhiMVPlanck), nside=nside)
# originalMap = hp.alm2map(wiener_filter(dlm, clpp, clpp+noise_min_variance(normPlanck, weightsQPlanck,
#     list(dictQuestPlanck.keys())[:-1])*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
clpp = np.concatenate(([0, 0], cls['6:phiphi'][:lmax_lensquest - 1]))

clpp = np.concatenate(([0, 0], cls['6:phiphi'][:lmax_lensquest - 1]*(2*np.pi)/(ellP[2:]*(ellP[2:]+1))))
# Wiener filter MV plots and lensing power spectrum after filtering
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(ellP, wiener_filter_plot(clpp, clpp+noise_min_variance(normPlanck, weightsQPlanck,
    list(dictQuestPlanck.keys())[:-1])), label='Planck')
axs[0].plot(ellP, wiener_filter_plot(clpp, clpp+noise_min_variance(normLB, weightsQLB,
    list(dictQuestLB.keys())[:-1])), label='LiteBIRD')
axs[0].plot(ellP, wiener_filter_plot(clpp, clpp+noise_min_variance(normPlLB, weightsQPlLB,
    list(dictQuestPlLB.keys())[:-1])), label='Planck+LiteBIRD')
axs[0].set_ylabel(r'Wiener filter', size=20)
axs[1].plot(ellP, ellP_factor*clpp, label='Theory', color='k')
axs[1].plot(ellP, ellP_factor*wiener_filter_plot(clpp**2, clpp+noise_min_variance(normPlanck, weightsQPlanck,
    list(dictQuestPlanck.keys())[:-1])), label='Planck')
axs[1].plot(ellP, ellP_factor*wiener_filter_plot(clpp**2, clpp+noise_min_variance(normLB, weightsQLB,
    list(dictQuestLB.keys())[:-1])), label='LiteBIRD')
axs[1].plot(ellP, ellP_factor*wiener_filter_plot(clpp**2, clpp+noise_min_variance(normPlLB, weightsQPlLB,
    list(dictQuestPlLB.keys())[:-1])), label='Planck+LiteBIRD')
axs[1].set_ylabel(r'$\frac{[\ell(\ell +1)]^2}{2\pi}C_\ell^{\phi\phi}$', size=20)
for ax in axs:
    ax.set_xlim([2,2500])
    ax.semilogx()
    ax.set_xlabel(r'$\ell$', size=20)
    ax.legend(fontsize=12)
plt.tight_layout()
