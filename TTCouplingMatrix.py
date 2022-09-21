"""
Calculation and plot of the TT coupling matrix from a given mask.

Author: Miguel Ruiz Granda
"""

# Import packages
import numpy as np
from astropy.io import ascii
import healpy as hp
import os
os.chdir('/home/miguel/Desktop/altamira_TFM/DataTFM')
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymaster as nmt

# Constants
nside = 2048
lmax = 2500
npix = hp.nside2npix(nside)
resol = hp.nside2resol(nside, arcmin=True)  # in arcmin
T_CMB = 2.7255e6  # in muK

# Apodize the masks on a scale of ~1deg
mask = np.load('/home/miguel/Desktop/TFM/2015_Galactic_GAL080.npy')
maskApo = np.load('Apodized_mask.npy')

#    Spin-0
f0 = nmt.NmtField(mask, None, spin=0, templates=None)
f0A = nmt.NmtField(maskApo, None, spin=0, templates=None)
# Create binning scheme. We will use 1 multipoles per bandpower.
b = nmt.NmtBin.from_lmax_linear(lmax, 1)
# We then generate an NmtWorkspace object that we use to compute and store the mode coupling matrix. Note that this
# matrix depends only on the masks of the two fields to correlate, but not on the maps themselves.

print('NMT fields are created')

# Two spin-0 fields: n_cls=1, [C_T1T2]
TT = nmt.NmtWorkspace()
TT.compute_coupling_matrix(f0, f0, b)
matrix = TT.get_coupling_matrix()
TT.compute_coupling_matrix(f0A, f0A, b)
matrixApo = TT.get_coupling_matrix()

minmin = np.min([np.min(matrix), np.min(matrixApo)])
maxmax = np.max([np.max(matrix), np.max(matrixApo)])
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
axes[0].set_title('Without apodization')
img1 = axes[0].imshow(matrix, cmap=plt.cm.jet, norm=mpl.colors.LogNorm(), vmin=minmin, vmax=maxmax)
axes[1].set_title('With apodization')
img2 = axes[1].imshow(matrixApo, cmap=plt.cm.jet, norm=mpl.colors.LogNorm(), vmin=minmin, vmax=maxmax)
# plt.colorbar(img2)
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(img2, cax=cbar_ax)