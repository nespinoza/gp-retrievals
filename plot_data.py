import numpy as np
import pickle
import matplotlib.pyplot as plt

import juliet
import transitspectroscopy as ts
import seaborn as sns
sns.set_style('ticks')

#out_name = 'data_multi_visit_gp_atmosphere'
out_name = 'data_multi_visit_gp_atmosphere_binnedR50'


binning = True
R = 50

# First things first, get data out; use it to steal the wavelengths:
wavelengths1, depths1, depths_error1 = np.loadtxt('data/espinoza/visit1.txt', unpack = True, usecols = (0, 1, 2))
wavelengths2, depths2, depths_error2 = np.loadtxt('data/espinoza/visit2.txt', unpack = True, usecols = (0, 1, 2))
wavelengths3, depths3, depths_error3 = np.loadtxt('data/espinoza/visit3.txt', unpack = True, usecols = (0, 1, 2))
wavelengths4, depths4, depths_error4 = np.loadtxt('data/espinoza/visit4.txt', unpack = True, usecols = (0, 1, 2))

w1, w2, w3, w4 = np.copy(wavelengths1), np.copy(wavelengths2), np.copy(wavelengths3), np.copy(wavelengths4)
if binning:

    wavelengths1, depths1, depths_error1 = ts.utils.bin_at_resolution(wavelengths1,
                                                                      depths1,
                                                                      depths_errors = depths_error1,
                                                                      R = R)

    wavelengths2, depths2, depths_error2 = ts.utils.bin_at_resolution(wavelengths2,
                                                                      depths2,
                                                                      depths_errors = depths_error2,
                                                                      R = R)

    wavelengths3, depths3, depths_error3 = ts.utils.bin_at_resolution(wavelengths3,
                                                                      depths3,
                                                                      depths_errors = depths_error3,
                                                                      R = R)

    wavelengths4, depths4, depths_error4 = ts.utils.bin_at_resolution(wavelengths4,
                                                                      depths4,
                                                                      depths_errors = depths_error4,
                                                                      R = R)

if 'binned' in out_name:

    w1, w2, w3, w4 = wavelengths1, wavelengths2, wavelengths3, wavelengths4

posterior_contam1 = np.load(out_name+'_binned_posterior_contamination1.npy')
posterior_spectrum1 = np.load(out_name+'_binned_posterior_spectrum1.npy')
posterior_contam2 = np.load(out_name+'_binned_posterior_contamination2.npy')
posterior_spectrum2 = np.load(out_name+'_binned_posterior_spectrum2.npy')
posterior_contam3 = np.load(out_name+'_binned_posterior_contamination3.npy')
posterior_spectrum3 = np.load(out_name+'_binned_posterior_spectrum3.npy')
posterior_contam4 = np.load(out_name+'_binned_posterior_contamination4.npy')
posterior_spectrum4 = np.load(out_name+'_binned_posterior_spectrum4.npy')

pcolor = '#0097a7'
contamcolor = '#4285f4'
datacolor = 'black'
combinedcolor = 'black'

def plotit(x, xb, y, yerr, posterior_contam, posterior_spectrum):

    plt.errorbar(x, y, yerr, fmt = '.', color = datacolor, zorder = 1)

    for i in range( posterior_contam.shape[0] ):

        plt.plot(xb,  posterior_contam[i,:] * posterior_spectrum[i,:], color = contamcolor, alpha = 0.01, zorder = 2)

    plt.plot(xb, np.nanmedian( posterior_contam * posterior_spectrum, axis = 0 ), color = 'blue', zorder = 3)

plt.figure(figsize = (8,8))
plt.subplot(411)
plt.title('Visit 1')
#plt.plot(wavelengths_model, true_spectrum1*model1, label = 'With stellar contamination')

plotit(wavelengths1, w1, depths1, depths_error1, posterior_contam1, posterior_spectrum1)
plt.xlim(1,5)
#plt.legend()

plt.subplot(412)
plt.title('Visit 2')

plotit(wavelengths2, w2, depths2, depths_error2, posterior_contam2, posterior_spectrum2)
plt.xlim(1,5)

plt.subplot(413)
plt.title('Visit 3')

plotit(wavelengths3, w3, depths3, depths_error3, posterior_contam3, posterior_spectrum3)

plt.xlim(1,5)

plt.subplot(414)
plt.title('Visit 4')

plotit(wavelengths4, w4, depths4, depths_error4, posterior_contam4, posterior_spectrum4)
plt.xlim(1,5)

plt.tight_layout()
plt.show()

out = pickle.load( open(out_name+'_results.pkl','rb') )
results = out['dynesty_results']

import corner
from dynesty.utils import resample_equal
weights = np.exp(results['logwt'] - results['logz'][-1])
posterior_samples = resample_equal(results.samples, weights)

samples = posterior_samples[:, :3]
print(samples.shape)
ndim = samples.shape[1]
# This is the true mean of the second mode that we used above:
#value1 = [250, 0.1, np.log10(1e-3)]
samples[:, -1] = np.log10(samples[:,-1])

for i in range(3):

    val, valup, valdown = juliet.utils.get_quantiles(samples[:, i])
    print('value =',val,'+/-', valup-val, val-valdown)
    val, valup, valdown = juliet.utils.get_quantiles(samples[:, i], alpha = 0.99)
    print('3-sigma upper limit = ',valup)
# Make the base corner plot
figure = corner.corner(samples)

# Extract the axes
"""
axes = np.array(figure.axes).reshape((ndim, ndim))

# Loop over the diagonal
for i in range(ndim):
    ax = axes[i, i]
    ax.axvline(value1[i], color="r")

# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(value1[xi], color="r")
        ax.axhline(value1[yi], color="r")
        ax.plot(value1[xi], value1[yi], "r")
"""
plt.show()
