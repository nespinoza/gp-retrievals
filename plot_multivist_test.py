import numpy as np
import pickle
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')

out_name = 'multi_visit_gp_atmosphere'

wavelengths_model, true_spectrum1, model1 = np.loadtxt(out_name+'_model1.txt', unpack = True, usecols = (0,1,2))
wavelengths_model, true_spectrum2, model2 = np.loadtxt(out_name+'_model2.txt', unpack = True, usecols = (0,1,2))
wavelengths_model, true_spectrum3, model3 = np.loadtxt(out_name+'_model3.txt', unpack = True, usecols = (0,1,2))
wavelengths_model, true_spectrum4, model4 = np.loadtxt(out_name+'_model4.txt', unpack = True, usecols = (0,1,2))

wavelengths1, binned_true_spectrum1, binned_contaminated_true_spectrum1, noise1 = np.loadtxt(out_name+'_data1.txt', unpack = True, usecols = (0,1,2,3))
wavelengths2, binned_true_spectrum2, binned_contaminated_true_spectrum2, noise2 = np.loadtxt(out_name+'_data2.txt', unpack = True, usecols = (0,1,2,3))
wavelengths3, binned_true_spectrum3, binned_contaminated_true_spectrum3, noise3 = np.loadtxt(out_name+'_data3.txt', unpack = True, usecols = (0,1,2,3))
wavelengths4, binned_true_spectrum4, binned_contaminated_true_spectrum4, noise4 = np.loadtxt(out_name+'_data4.txt', unpack = True, usecols = (0,1,2,3))

posterior_contam1 = np.load(out_name+'_binned_posterior_contamination1.npy')
posterior_spectrum1 = np.load(out_name+'_binned_posterior_spectrum1.npy')
posterior_contam2 = np.load(out_name+'_binned_posterior_contamination2.npy')
posterior_spectrum2 = np.load(out_name+'_binned_posterior_spectrum2.npy')
posterior_contam3 = np.load(out_name+'_binned_posterior_contamination3.npy')
posterior_spectrum3 = np.load(out_name+'_binned_posterior_spectrum3.npy')
posterior_contam4 = np.load(out_name+'_binned_posterior_contamination4.npy')
posterior_spectrum4 = np.load(out_name+'_binned_posterior_spectrum4.npy')

yerr1, yerr2, yerr3, yerr4 = np.ones(len(wavelengths1))*300,\
                             np.ones(len(wavelengths2))*300,\
                             np.ones(len(wavelengths3))*300,\
                             np.ones(len(wavelengths4))*300

pcolor = '#0097a7'
contamcolor = '#4285f4'
datacolor = 'black'
combinedcolor = 'black'

def plotit(wavelengths, binned_contaminated_true_spectrum, posterior_contam, posterior_spectrum, noise):

    plt.plot(wavelengths, binned_contaminated_true_spectrum, '-', color = combinedcolor, lw = 3, zorder = 3)
    plt.errorbar(wavelengths, binned_contaminated_true_spectrum + noise, yerr1, fmt = '.', color = datacolor, alpha = 0.1, zorder = 1)

    for i in range( posterior_contam.shape[0] ):

        plt.plot(wavelengths,  posterior_contam[i,:] * posterior_spectrum[i,:], color = contamcolor, alpha = 0.01, zorder = 2)

    plt.plot(wavelengths, np.nanmedian( posterior_contam * posterior_spectrum, axis = 0 ), color = 'blue', zorder = 3)


plt.figure(figsize = (8,8))
plt.subplot(411)
plt.title('Visit 1')
#plt.plot(wavelengths_model, true_spectrum1*model1, label = 'With stellar contamination')

plotit(wavelengths1, binned_contaminated_true_spectrum1, posterior_contam1, posterior_spectrum1, noise1)
plt.xlim(1,5)
#plt.legend()

plt.subplot(412)
plt.title('Visit 2')

plotit(wavelengths2, binned_contaminated_true_spectrum2, posterior_contam2, posterior_spectrum2, noise2)

plt.xlim(1,5)

plt.subplot(413)
plt.title('Visit 3')

plotit(wavelengths3, binned_contaminated_true_spectrum3, posterior_contam3, posterior_spectrum3, noise3)

plt.xlim(1,5)

plt.subplot(414)
plt.title('Visit 4')

plotit(wavelengths4, binned_contaminated_true_spectrum4, posterior_contam4, posterior_spectrum4, noise4)

plt.xlim(1,5)

plt.tight_layout()
plt.show()

out = pickle.load( open('multi_visit_gp_atmosphere_results.pkl','rb') )
results = out['dynesty_results']

import corner
from dynesty.utils import resample_equal
weights = np.exp(results['logwt'] - results['logz'][-1])
posterior_samples = resample_equal(results.samples, weights)

samples = posterior_samples[:, :3]
print(samples.shape)
ndim = samples.shape[1]
# This is the true mean of the second mode that we used above:
value1 = [250, 0.1, np.log10(1e-3)]
samples[:, -1] = np.log10(samples[:,-1])

# Make the base corner plot
figure = corner.corner(samples)

# Extract the axes
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

plt.show()
