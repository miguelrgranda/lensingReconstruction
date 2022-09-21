'''
This script plots the Wiener-filtered maps for the different quadratic estimators. The data used are the full-sky
simulations for Planck, LiteBIRD and the combination of both of them.
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
nmvPlanck, weightsPlanck = getweights(normPlanck, list(dictQuestPlanck.keys())[:-1])
make_minvariance(dictQuestPlanck, weightsPlanck)


# LiteBIRD experiment
dictQuestLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/reconstructionSpectraLB.npy', allow_pickle=True).item()
normLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/NormalizationLB.npy', allow_pickle=True).item()
nmvLB, weightsLB = getweights(normLB, list(dictQuestLB.keys())[:-1])
make_minvariance(dictQuestLB, weightsLB)


# Combination of Planck and LiteBIRD
dictQuestPlLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/reconstructionSpectraPlLB.npy', allow_pickle=True).item()
normPlLB = np.load('/home/miguel/Desktop/altamira_TFM/DataTFM/NormalizationPlLB.npy', allow_pickle=True).item()
nmvPlLB, weightsPlLB = getweights(normPlLB, list(dictQuestPlLB.keys())[:-1])
make_minvariance(dictQuestPlLB, weightsPlLB)

ellP = np.arange(0, lmax_lensquest+1)
ellP_factor = (ellP * (ellP + 1)) ** 2 / (2 * np.pi)
clpp = np.concatenate(([0, 0], cls['6:phiphi'][:lmax_lensquest - 1]))
originalMap = hp.alm2map(dlm, nside=nside)
# Planck experiment maps
recoveredMapTTPl = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlanck['TT'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlanck['TTTT']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapEEPl = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlanck['EE'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlanck['EEEE']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapTEPl = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlanck['TE'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlanck['TETE']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapEBPl = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlanck['EB'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlanck['EBEB']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapTBPl = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlanck['TB'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlanck['TBTB']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapMVPl = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlanck['MV'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+noise_min_variance(normPlanck, weightsPlanck,
    list(dictQuestPlanck.keys())[:-1])*(ellP*(ellP+1)/(2*np.pi))), nside=nside)

# LiteBIRD experiment maps
recoveredMapTTLB = hp.alm2map(wiener_filter(hp.almxfl(dictQuestLB['TT'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normLB['TTTT']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapEELB = hp.alm2map(wiener_filter(hp.almxfl(dictQuestLB['EE'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normLB['EEEE']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapTELB = hp.alm2map(wiener_filter(hp.almxfl(dictQuestLB['TE'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normLB['TETE']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapTBLB = hp.alm2map(wiener_filter(hp.almxfl(dictQuestLB['TB'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normLB['TBTB']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapEBLB = hp.alm2map(wiener_filter(hp.almxfl(dictQuestLB['EB'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normLB['EBEB']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapMVLB = hp.alm2map(wiener_filter(hp.almxfl(dictQuestLB['MV'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+noise_min_variance(normLB, weightsLB,
    list(dictQuestLB.keys())[:-1])*(ellP*(ellP+1)/(2*np.pi))), nside=nside)

# Combination of Planck and LiteBIRD maps
recoveredMapTTComb = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlLB['TT'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlLB['TTTT']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapEBComb = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlLB['EB'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlLB['EBEB']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapTBComb = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlLB['TB'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlLB['TBTB']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapEEComb = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlLB['EE'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlLB['EEEE']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapTEComb = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlLB['TE'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+normPlLB['TETE']*(ellP*(ellP+1)/(2*np.pi))), nside=nside)
recoveredMapMVComb = hp.alm2map(wiener_filter(hp.almxfl(dictQuestPlLB['MV'],
                    np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2))), clpp,
                        clpp+noise_min_variance(normPlLB, weightsPlLB,
    list(dictQuestPlLB.keys())[:-1])*(ellP*(ellP+1)/(2*np.pi))), nside=nside)

fig, axs = plt.subplots(ncols=1, nrows=1)
plt.axes(axs)
hp.cartview(originalMap, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=True)
fig.tight_layout()

minv = -0.0014# -0.00100
maxv = 0.002 # 0.0015
fig, axs = plt.subplots(ncols=3, nrows=6) #, figsize=(12, 20))
plt.axes(axs[0, 0])
hp.cartview(recoveredMapTTPl, lonra=[-10, 10], latra=[-10, 10], hold=True, title='Planck', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[1, 0])
hp.cartview(recoveredMapEEPl, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[2, 0])
hp.cartview(recoveredMapTEPl, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[3, 0])
hp.cartview(recoveredMapTBPl, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[4, 0])
hp.cartview(recoveredMapEBPl, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[5, 0])
hp.cartview(recoveredMapMVPl, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[0, 1])
hp.cartview(recoveredMapTTLB, lonra=[-10, 10], latra=[-10, 10], hold=True, title='LiteBIRD', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[1, 1])
hp.cartview(recoveredMapEELB, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[2, 1])
hp.cartview(recoveredMapTELB, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[3, 1])
hp.cartview(recoveredMapTBLB, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[4, 1])
hp.cartview(recoveredMapEBLB, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[5, 1])
hp.cartview(recoveredMapMVLB, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[0, 2])
hp.cartview(recoveredMapTTComb, lonra=[-10, 10], latra=[-10, 10], hold=True, title='Planck & LiteBIRD', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[1, 2])
hp.cartview(recoveredMapEEComb, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[2, 2])
hp.cartview(recoveredMapTEComb, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[3, 2])
hp.cartview(recoveredMapTBComb, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[4, 2])
hp.cartview(recoveredMapEBComb, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
plt.axes(axs[5, 2])
hp.cartview(recoveredMapMVComb, lonra=[-10, 10], latra=[-10, 10], hold=True, title='', xsize=1000, cbar=False, min=minv, max=maxv)
fig.tight_layout()
im = fig.axes[0].get_images()[0]
plt.colorbar(im, ax=fig.axes)
# plt.savefig('mapsPlanckLBLensing.pdf', bbox_inches='tight')




