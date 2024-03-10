# Import classic libraries:
import numpy as np
rstate= np.random.default_rng(56101)
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

# Import dynesty:
import dynesty
from dynesty.utils import resample_equal

# Import celerite2:
import celerite2
from celerite2 import terms

# Utilities function:
import utils

# First things first, get data out; use it to steal the wavelengths:
wavelengths1, depths1, depths_error1 = np.loadtxt('data/espinoza/visit1.txt', unpack = True, usecols = (0, 1, 2))
wavelengths2, depths2, depths_error2 = np.loadtxt('data/espinoza/visit2.txt', unpack = True, usecols = (0, 1, 2))
wavelengths3, depths3, depths_error3 = np.loadtxt('data/espinoza/visit3.txt', unpack = True, usecols = (0, 1, 2))
wavelengths4, depths4, depths_error4 = np.loadtxt('data/espinoza/visit4.txt', unpack = True, usecols = (0, 1, 2))

out_name = 'multi_visit_gp_atmosphere'

# Define star/planet atmospheric properties:
star_properties = {}

star_properties['R'] =  0.11697 # Rsun
star_properties['Teff'] = 2559.0
star_properties['FeH'] = 0.04
star_properties['logg'] = 5.21

planet_properties = {}

planet_properties['R'] = 0.917985 # Rearth
planet_properties['M'] = 0.6356
planet_properties['T_eq'] = 255 

# Atmosphere is hydrogen-dominated by default:
bulk_species = ['H2']
param_species = ['O2', 'CO2', 'CH4', 'H2O', 'O3', 'N2O']
true_T = 250.
true_CO2 = 0.1
# But! Has 30% CO2, 10% CH4 ahd 10% H2O:
#
#                                 O2  CO2  CH4  H2O  O3   N2O
true_log_X = np.log10( np.array([0.0, true_CO2, 0.0, 0.0, 0.0, 0.0]) )
# And, of course, has clouds:
#
# 
true_a, true_Pcloud = 1., 1e-3
log_a, gamma, log_P_cloud = np.log10(true_a), -4., np.log10(true_Pcloud)
# Add a small depth offset in ppm:
true_offset = 400.

# Generate planet:
if not os.path.exists(out_name+'_full_posterior_spectrum.npy'):

    planet = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 500)

else:

    planet = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 10)

wavelengths_model = planet.wl

# Set abundances, clouds properties:
planet.set_parameters(true_T, true_log_X, [log_a, gamma, log_P_cloud])
true_spectrum = gaussian_filter(true_offset + planet.get_spectrum() * 1e6, 5)

# All right, now, bin the model to match the wavelengths of the data:

# Generate noise samples and wavelengths:
true_rho = 0.5
true_gp_sigma = 0.05
true_noise = 300.
true_factor = 0.5

# Define priors (assumed uniform):
a_T, b_T = 100.,300.
a_CO2, b_CO2 = 0., 1.0
a_Pcloud, b_Pcloud = 1e-6, 1.
a_offset, b_offset = -1000,1000
a_rho, b_rho = 0.0, 100.
a_gp_sigma, b_gp_sigma = 0.0,10. 
a_sigma_w, b_sigma_w = 0.0, 1000

# Simulate some wavelengths, noise and errorbars (underestimated):
noise1 = np.random.normal(0., true_noise, len(wavelengths1))
noise2 = np.random.normal(0., true_noise, len(wavelengths2))
noise3 = np.random.normal(0., true_noise, len(wavelengths3))
noise4 = np.random.normal(0., true_noise, len(wavelengths4))

yerr1 = np.ones(len(wavelengths1)) * true_noise * true_factor # errorbars are smaller than they should
yerr2 = np.ones(len(wavelengths2)) * true_noise * true_factor 
yerr3 = np.ones(len(wavelengths3)) * true_noise * true_factor 
yerr4 = np.ones(len(wavelengths4)) * true_noise * true_factor 

# Generate a GP sample from celerite2:
kernel = terms.Matern32Term(sigma = true_gp_sigma, 
                            rho = true_rho, 
                            eps=0.01)

true_gp = celerite2.GaussianProcess(kernel, mean = 1.0)
true_gp.compute(wavelengths_model, yerr = 0.)#yerr + 50.)

# Sample four samples from the (noiseless) GP:
model1 = true_gp.sample()
model2 = true_gp.sample()
model3 = true_gp.sample()
model4 = true_gp.sample()

gp1 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp1.compute(wavelengths1, yerr = 0.)

gp2 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp2.compute(wavelengths2, yerr = 0.) 

gp3 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp3.compute(wavelengths3, yerr = 0.) 

gp4 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp4.compute(wavelengths4, yerr = 0.) 

if os.path.exists(out_name+'_model1.txt'):

    wavelengths_model, true_spectrum1, model1 = np.loadtxt(out_name+'_model1.txt', unpack = True, usecols = (0,1,2))
    wavelengths_model, true_spectrum2, model2 = np.loadtxt(out_name+'_model2.txt', unpack = True, usecols = (0,1,2))
    wavelengths_model, true_spectrum3, model3 = np.loadtxt(out_name+'_model3.txt', unpack = True, usecols = (0,1,2))
    wavelengths_model, true_spectrum4, model4 = np.loadtxt(out_name+'_model4.txt', unpack = True, usecols = (0,1,2))

else:

    fout1 = open(out_name+'_model1.txt', 'w')
    fout2 = open(out_name+'_model2.txt', 'w')
    fout3 = open(out_name+'_model3.txt', 'w')
    fout4 = open(out_name+'_model4.txt', 'w')
    for i in range(len(wavelengths_model)):

        fout1.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(wavelengths_model[i], true_spectrum[i], model1[i]))
        fout2.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(wavelengths_model[i], true_spectrum[i], model2[i]))
        fout3.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(wavelengths_model[i], true_spectrum[i], model3[i]))
        fout4.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(wavelengths_model[i], true_spectrum[i], model4[i]))

    fout1.close()
    fout2.close()
    fout3.close()
    fout4.close()

plt.title('True spectrum:')
plt.plot(wavelengths_model, true_spectrum)
plt.show()

# Bin models to match data:
binned_true_spectrum1 = utils.bin_to_data(wavelengths1, wavelengths_model, true_spectrum)
binned_true_spectrum2 = utils.bin_to_data(wavelengths2, wavelengths_model, true_spectrum)
binned_true_spectrum3 = utils.bin_to_data(wavelengths3, wavelengths_model, true_spectrum)
binned_true_spectrum4 = utils.bin_to_data(wavelengths4, wavelengths_model, true_spectrum)

binned_contaminated_true_spectrum1 = utils.bin_to_data(wavelengths1, wavelengths_model, true_spectrum * model1)
binned_contaminated_true_spectrum2 = utils.bin_to_data(wavelengths2, wavelengths_model, true_spectrum * model2)
binned_contaminated_true_spectrum3 = utils.bin_to_data(wavelengths3, wavelengths_model, true_spectrum * model3)
binned_contaminated_true_spectrum4 = utils.bin_to_data(wavelengths4, wavelengths_model, true_spectrum * model4)
print('Binned datapoints:',len(binned_true_spectrum1))
print('Model datapoints:', len(wavelengths_model))
# Save fake data:
if os.path.exists(out_name+'_data1.txt'):

    wavelengths1, binned_true_spectrum1, binned_contaminated_true_spectrum1, noise1 = np.loadtxt(out_name+'_data1.txt', unpack = True, usecols = (0,1,2,3))
    wavelengths2, binned_true_spectrum2, binned_contaminated_true_spectrum2, noise2 = np.loadtxt(out_name+'_data2.txt', unpack = True, usecols = (0,1,2,3))
    wavelengths3, binned_true_spectrum3, binned_contaminated_true_spectrum3, noise3 = np.loadtxt(out_name+'_data3.txt', unpack = True, usecols = (0,1,2,3))
    wavelengths4, binned_true_spectrum4, binned_contaminated_true_spectrum4, noise4 = np.loadtxt(out_name+'_data4.txt', unpack = True, usecols = (0,1,2,3))

else:

    fout = open(out_name+'_data1.txt', 'w')
    for i in range(len(wavelengths1)):

        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(wavelengths1[i], binned_true_spectrum1[i], binned_contaminated_true_spectrum1[i], noise1[i]))

    fout.close()

    fout = open(out_name+'_data2.txt', 'w')
    for i in range(len(wavelengths2)):

        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(wavelengths2[i], binned_true_spectrum2[i], binned_contaminated_true_spectrum2[i], noise2[i]))

    fout.close()

    fout = open(out_name+'_data3.txt', 'w')
    for i in range(len(wavelengths3)):

        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(wavelengths3[i], binned_true_spectrum3[i], binned_contaminated_true_spectrum3[i], noise3[i]))

    fout.close()

    fout = open(out_name+'_data4.txt', 'w')
    for i in range(len(wavelengths4)):

        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(wavelengths4[i], binned_true_spectrum4[i], binned_contaminated_true_spectrum4[i], noise4[i]))

    fout.close()

plt.subplot(411)
plt.title('Visit 1')
plt.plot(wavelengths_model, true_spectrum*model1, label = 'With stellar contamination')
plt.plot(wavelengths_model, true_spectrum, label = 'Original spectra')

plt.errorbar(wavelengths1, binned_contaminated_true_spectrum1, fmt = 'o-', label = 'Binned w/ stellar contamination')
plt.errorbar(wavelengths1, binned_contaminated_true_spectrum1 + noise1, yerr1, fmt = '.', label = 'Binned w/ stellar contamination + noise')
plt.legend()

plt.subplot(412)
plt.title('Visit 2')
plt.plot(wavelengths_model, true_spectrum*model2, label = 'With stellar contamination')
plt.plot(wavelengths_model, true_spectrum, label = 'Original spectra')

plt.errorbar(wavelengths2, binned_contaminated_true_spectrum2, fmt = 'o-', label = 'Binned w/ stellar contamination')
plt.errorbar(wavelengths2, binned_contaminated_true_spectrum2 + noise2, yerr2, fmt = '.', label = 'Binned w/ stellar contamination + noise')
plt.legend()

plt.subplot(413)
plt.title('Visit 3')
plt.plot(wavelengths_model, true_spectrum*model3)#, label = 'With stellar contamination')
plt.plot(wavelengths_model, true_spectrum)#, label = 'Original spectra')

plt.errorbar(wavelengths3, binned_contaminated_true_spectrum3, fmt = 'o-')#, label = 'Binned w/ stellar contamination')
plt.errorbar(wavelengths3, binned_contaminated_true_spectrum3 + noise3, yerr3, fmt = '.')#, label = 'Binned w/ stellar contamination + noise')
plt.legend()

plt.subplot(414)
plt.title('Visit 4')
plt.plot(wavelengths_model, true_spectrum*model4)#, label = 'With stellar contamination')
plt.plot(wavelengths_model, true_spectrum)#, label = 'Original spectra')

plt.errorbar(wavelengths4, binned_contaminated_true_spectrum4, fmt = 'o-')#, label = 'Binned w/ stellar contamination')
plt.errorbar(wavelengths4, binned_contaminated_true_spectrum4 + noise4, yerr4, fmt = '.')#, label = 'Binned w/ stellar contamination + noise')
plt.legend()
plt.show()

real_y1 = binned_contaminated_true_spectrum1 + noise1#true_spectrum*model + noise
logy1 = np.log( real_y1 )
logy_err1 = yerr1 / real_y1

real_y2 = binned_contaminated_true_spectrum2 + noise2#true_spectrum*model + noise
logy2 = np.log( real_y2 )
logy_err2 = yerr2 / real_y2

real_y3 = binned_contaminated_true_spectrum3 + noise3#true_spectrum*model + noise
logy3 = np.log( real_y3 )
logy_err3 = yerr3 / real_y3

real_y4 = binned_contaminated_true_spectrum4 + noise4#true_spectrum*model + noise
logy4 = np.log( real_y4 )
logy_err4 = yerr4 / real_y4

# Define the prior and the log-likelihood a-la-dynesty:
def loglike(theta):

    T, CO2, Pcloud, offset1, rho1, gp_sigma1, sigma_w1, \
                    offset2, rho2, gp_sigma2, sigma_w2, \
                    offset3, rho3, gp_sigma3, sigma_w3, \
                    offset4, rho4, gp_sigma4, sigma_w4 = theta

    total_variance1 = ( logy_err1**2 + (sigma_w1 / real_y1)**2 )
    total_variance2 = ( logy_err2**2 + (sigma_w2 / real_y2)**2 )
    total_variance3 = ( logy_err3**2 + (sigma_w3 / real_y3)**2 )
    total_variance4 = ( logy_err4**2 + (sigma_w4 / real_y4)**2 )
    
    # Generate transmission spectrum:
    #                           O2   CO2  CH4  H2O  O3   N2O
    log_X = np.log10( np.array([0.0, CO2, 0.0, 0.0, 0.0, 0.0]) )

    # Set abundances, clouds properties, generate spectrum:
    planet.set_parameters(T, log_X, [log_a, gamma, np.log10(Pcloud)])
    spectrum1 = gaussian_filter(offset1 + planet.get_spectrum() * 1e6, 5)
    spectrum2 = gaussian_filter(offset2 + planet.get_spectrum() * 1e6, 5)
    spectrum3 = gaussian_filter(offset3 + planet.get_spectrum() * 1e6, 5)
    spectrum4 = gaussian_filter(offset4 + planet.get_spectrum() * 1e6, 5)
    # Bin spectrum to the data res:
    binned_spectrum1 = utils.bin_to_data(wavelengths1, wavelengths_model, spectrum1)
    binned_spectrum2 = utils.bin_to_data(wavelengths2, wavelengths_model, spectrum2)
    binned_spectrum3 = utils.bin_to_data(wavelengths3, wavelengths_model, spectrum3)
    binned_spectrum4 = utils.bin_to_data(wavelengths4, wavelengths_model, spectrum4)
    
    # Subtract model from data:
    residuals1 = logy1 - np.log(binned_spectrum1)
    residuals2 = logy2 - np.log(binned_spectrum2)
    residuals3 = logy3 - np.log(binned_spectrum3)
    residuals4 = logy4 - np.log(binned_spectrum4)

    # Update GP hyperparmeters. First, re-set kernel:
    gp1.kernel = terms.Matern32Term(sigma = gp_sigma1,
                                   rho = rho1,
                                   eps=1e-6)

    gp2.kernel = terms.Matern32Term(sigma = gp_sigma2,
                                   rho = rho2,
                                   eps=1e-6)

    gp3.kernel = terms.Matern32Term(sigma = gp_sigma3,
                                   rho = rho3,
                                   eps=1e-6)

    gp4.kernel = terms.Matern32Term(sigma = gp_sigma4,
                                   rho = rho4,
                                   eps=1e-6)

    # Compute:
    gp1.compute(wavelengths1, diag=total_variance1, quiet=True)
    gp2.compute(wavelengths2, diag=total_variance2, quiet=True)
    gp3.compute(wavelengths3, diag=total_variance3, quiet=True)
    gp4.compute(wavelengths4, diag=total_variance4, quiet=True)

    #print('In loglike:', theta)
    #print('Residuals:', residuals)
    # Return log-likelihood if compliant with priors:
    if (T > a_T and T < b_T) and (CO2 > a_CO2 and CO2 < b_CO2) and (Pcloud > a_Pcloud and Pcloud < b_Pcloud) and \
       (offset1 > a_offset and offset1 < b_offset) and (rho1 > a_rho and rho1 < b_rho) and \
       (gp_sigma1 > a_gp_sigma and gp_sigma1 < b_gp_sigma) and \
       (sigma_w1 > a_sigma_w and sigma_w1 < b_sigma_w) and \
       (offset2 > a_offset and offset2 < b_offset) and (rho2 > a_rho and rho2 < b_rho) and \
       (gp_sigma2 > a_gp_sigma and gp_sigma2 < b_gp_sigma) and \
       (sigma_w2 > a_sigma_w and sigma_w2 < b_sigma_w) and \
       (offset3 > a_offset and offset3 < b_offset) and (rho3 > a_rho and rho3 < b_rho) and \
       (gp_sigma3 > a_gp_sigma and gp_sigma3 < b_gp_sigma) and \
       (sigma_w3 > a_sigma_w and sigma_w3 < b_sigma_w) and \
       (offset4 > a_offset and offset4 < b_offset) and (rho4 > a_rho and rho4 < b_rho) and \
       (gp_sigma4 > a_gp_sigma and gp_sigma4 < b_gp_sigma) and \
       (sigma_w4 > a_sigma_w and sigma_w4 < b_sigma_w):

        return gp1.log_likelihood(residuals1) + gp2.log_likelihood(residuals2) + gp3.log_likelihood(residuals3) + gp4.log_likelihood(residuals4)

    else:

        return -1e101

def prior_transform(utheta):

    uT, uCO2, uPcloud, uoffset1, urho1, ugp_sigma1, usigma_w1, \
                       uoffset2, urho2, ugp_sigma2, usigma_w2, \
                       uoffset3, urho3, ugp_sigma3, usigma_w3, \
                       uoffset4, urho4, ugp_sigma4, usigma_w4  = utheta

    # Convert from unitary to the priors:
    T = transform_uniform(uT, [a_T, b_T])
    CO2 = transform_uniform(uCO2, [a_CO2, b_CO2])
    Pcloud = transform_loguniform(uPcloud, [a_Pcloud, b_Pcloud])
    offset1 = transform_uniform(uoffset1, [a_offset, b_offset])
    offset2 = transform_uniform(uoffset2, [a_offset, b_offset])
    offset3 = transform_uniform(uoffset3, [a_offset, b_offset])
    offset4 = transform_uniform(uoffset4, [a_offset, b_offset])
    rho1 = transform_uniform(urho1, [a_rho, b_rho])
    gp_sigma1 = transform_uniform(ugp_sigma1, [a_gp_sigma, b_gp_sigma])
    sigma_w1 = transform_uniform(usigma_w1, [a_sigma_w, b_sigma_w])
    rho2 = transform_uniform(urho2, [a_rho, b_rho])
    gp_sigma2 = transform_uniform(ugp_sigma2, [a_gp_sigma, b_gp_sigma])
    sigma_w2 = transform_uniform(usigma_w2, [a_sigma_w, b_sigma_w])
    rho3 = transform_uniform(urho3, [a_rho, b_rho])
    gp_sigma3 = transform_uniform(ugp_sigma3, [a_gp_sigma, b_gp_sigma])
    sigma_w3 = transform_uniform(usigma_w3, [a_sigma_w, b_sigma_w])
    rho4 = transform_uniform(urho4, [a_rho, b_rho])
    gp_sigma4 = transform_uniform(ugp_sigma4, [a_gp_sigma, b_gp_sigma])
    sigma_w4 = transform_uniform(usigma_w4, [a_sigma_w, b_sigma_w])

    #print('In prior transform:', T, CO2, Pcloud, offset, rho, gp_sigma, sigma_w)
    return T, CO2, Pcloud, offset1, rho1, gp_sigma1, sigma_w1, offset2, rho2, gp_sigma2, sigma_w2, offset3, rho3, gp_sigma3, sigma_w3, offset4, rho4, gp_sigma4, sigma_w4 

def transform_uniform(x, hyperparameters):

    a, b = hyperparameters
    return a + (b-a)*x 

def transform_loguniform(x, hyperparameters):
    a, b = hyperparameters
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la + x * (lb - la))

# Run dynesty:
if not os.path.exists(out_name+'_results.pkl'):

    dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=19,
                                            bound='multi', sample='rwalk', rstate=rstate)

    dsampler.run_nested()
    results = dsampler.results
    out = {}

    out['dynesty_results'] = results
    pickle.dump(out, open(out_name+'_results.pkl', 'wb'))

else:

    out = pickle.load(open(out_name+'_results.pkl', 'rb'))
    results = out['dynesty_results']

# Plot some results:
from dynesty import plotting as dyplot

true_sigma_w = true_noise * np.sqrt( 1 - true_factor**2 )
truths = [true_T, true_CO2, true_Pcloud, true_offset, true_rho, true_gp_sigma, true_sigma_w, \
                                         true_offset, true_rho, true_gp_sigma, true_sigma_w, \
                                         true_offset, true_rho, true_gp_sigma, true_sigma_w, \
                                         true_offset, true_rho, true_gp_sigma, true_sigma_w]

labels = [r'$T$ (K)', r'$CO2$', r'$P_{cloud}$' , r'Offset 1 (ppm)', r'$\rho_1$', r'$\sigma_{GP1}$', r'$\sigma_w1$', \
                                                 r'Offset 2 (ppm)', r'$\rho_2$', r'$\sigma_{GP2}$', r'$\sigma_w2$', \
                                                 r'Offset 3 (ppm)', r'$\rho_3$', r'$\sigma_{GP3}$', r'$\sigma_w3$', \
                                                 r'Offset 4 (ppm)', r'$\rho_4$', r'$\sigma_{GP4}$', r'$\sigma_w4$']

print(results.samples.shape)

fig, axes = dyplot.traceplot(results, truths=truths, labels=labels,
                             fig=plt.subplots(19, 2, figsize=(18, 10)))
fig.tight_layout()  
plt.show()

fig, axes = dyplot.cornerplot(results, truths=truths, show_titles=True, 
                              title_kwargs={'y': 1.04}, labels=labels,
                              fig=plt.subplots(19, 19, figsize=(12, 12)))

plt.show()

weights = np.exp(results['logwt'] - results['logz'][-1])
posterior_samples = resample_equal(results.samples, weights)

print('Computing posterior model plot...')

# Plot:
ax1 = plt.subplot(411)
plt.title('Visit 1')
plt.plot(wavelengths_model, true_spectrum, label = 'Original spectra', color = 'black')
plt.plot(wavelengths_model, true_spectrum*model1, label = 'Spectra w/ stellar contamination', color = 'orangered')
plt.errorbar(wavelengths1, binned_contaminated_true_spectrum1 + noise1, yerr1, fmt = 'k.', label = 'Data')

ax2 = plt.subplot(412)
plt.plot(wavelengths_model, true_spectrum, color = 'black')
plt.plot(wavelengths_model, true_spectrum*model2, color = 'orangered')
plt.errorbar(wavelengths2, binned_contaminated_true_spectrum2 + noise2, yerr2, fmt = 'k.')

ax3 = plt.subplot(413)
plt.plot(wavelengths_model, true_spectrum, color = 'black')
plt.plot(wavelengths_model, true_spectrum*model3, color = 'orangered')
plt.errorbar(wavelengths3, binned_contaminated_true_spectrum3 + noise3, yerr3, fmt = 'k.')

ax4 = plt.subplot(414)
plt.plot(wavelengths_model, true_spectrum, color = 'black')
plt.plot(wavelengths_model, true_spectrum*model4, color = 'orangered')
plt.errorbar(wavelengths4, binned_contaminated_true_spectrum4 + noise4, yerr4, fmt = 'k.')

indexes = np.arange(posterior_samples.shape[0])
idx = np.random.choice(indexes, 1000, replace = False)

if not os.path.exists(out_name+'_full_posterior_spectrum4.npy'):

    counter = 0
    contamination_posteriors1 = np.zeros([len(idx), len(wavelengths1)])
    binned_spectrum_posteriors1 = np.zeros([len(idx), len(wavelengths1)])
    full_spectrum_posteriors1 = np.zeros([len(idx), len(wavelengths_model)])

    contamination_posteriors2 = np.zeros([len(idx), len(wavelengths2)])
    binned_spectrum_posteriors2 = np.zeros([len(idx), len(wavelengths2)])
    full_spectrum_posteriors2 = np.zeros([len(idx), len(wavelengths_model)])

    contamination_posteriors3 = np.zeros([len(idx), len(wavelengths3)])
    binned_spectrum_posteriors3 = np.zeros([len(idx), len(wavelengths3)])
    full_spectrum_posteriors3 = np.zeros([len(idx), len(wavelengths_model)])

    contamination_posteriors4 = np.zeros([len(idx), len(wavelengths4)])
    binned_spectrum_posteriors4 = np.zeros([len(idx), len(wavelengths4)])
    full_spectrum_posteriors4 = np.zeros([len(idx), len(wavelengths_model)])

    for sample in posterior_samples[idx,:]:

        T, CO2, Pcloud, offset1, rho1, gp_sigma1, sigma_w1, \
                        offset2, rho2, gp_sigma2, sigma_w2, \
                        offset3, rho3, gp_sigma3, sigma_w3, \
                        offset4, rho4, gp_sigma4, sigma_w4 = sample


        total_variance1 = ( logy_err1**2 + (sigma_w1 / real_y1)**2 )
        total_variance2 = ( logy_err2**2 + (sigma_w2 / real_y2)**2 )
        total_variance3 = ( logy_err3**2 + (sigma_w3 / real_y3)**2 )
        total_variance4 = ( logy_err4**2 + (sigma_w4 / real_y4)**2 )

        # Generate transmission spectrum:
        #                           O2   CO2  CH4  H2O  O3   N2O
        log_X = np.log10( np.array([0.0, CO2, 0.0, 0.0, 0.0, 0.0]) )

        # Set abundances, clouds properties, generate spectrum:
        planet.set_parameters(T, log_X, [log_a, gamma, np.log10(Pcloud)])
        spectrum1 = gaussian_filter(offset1 + planet.get_spectrum() * 1e6, 5)
        spectrum2 = gaussian_filter(offset2 + planet.get_spectrum() * 1e6, 5)
        spectrum3 = gaussian_filter(offset3 + planet.get_spectrum() * 1e6, 5)
        spectrum4 = gaussian_filter(offset4 + planet.get_spectrum() * 1e6, 5)
        # Bin spectrum to the data res:
        binned_spectrum1 = utils.bin_to_data(wavelengths1, wavelengths_model, spectrum1)
        binned_spectrum2 = utils.bin_to_data(wavelengths2, wavelengths_model, spectrum2)
        binned_spectrum3 = utils.bin_to_data(wavelengths3, wavelengths_model, spectrum3)
        binned_spectrum4 = utils.bin_to_data(wavelengths4, wavelengths_model, spectrum4)

        # Subtract model from data:
        residuals1 = logy1 - np.log(binned_spectrum1)
        residuals2 = logy2 - np.log(binned_spectrum2)
        residuals3 = logy3 - np.log(binned_spectrum3)
        residuals4 = logy4 - np.log(binned_spectrum4)

        # Update GP hyperparmeters. First, re-set kernel:
        gp1.kernel = terms.Matern32Term(sigma = gp_sigma1,
                                       rho = rho1,
                                       eps=1e-6)

        gp2.kernel = terms.Matern32Term(sigma = gp_sigma2,
                                       rho = rho2,
                                       eps=1e-6)

        gp3.kernel = terms.Matern32Term(sigma = gp_sigma3,
                                       rho = rho3,
                                       eps=1e-6)

        gp4.kernel = terms.Matern32Term(sigma = gp_sigma4,
                                       rho = rho4,
                                       eps=1e-6)

        # Compute:
        gp1.compute(wavelengths1, diag=total_variance1, quiet=True)
        gp2.compute(wavelengths2, diag=total_variance2, quiet=True)
        gp3.compute(wavelengths3, diag=total_variance3, quiet=True)
        gp4.compute(wavelengths4, diag=total_variance4, quiet=True)

        conditional1 = gp1.condition(residuals1, wavelengths1)
        conditional2 = gp2.condition(residuals2, wavelengths2)
        conditional3 = gp3.condition(residuals3, wavelengths3)
        conditional4 = gp4.condition(residuals4, wavelengths4)

        contamination_posteriors1[counter, :] = np.exp(conditional1.sample())
        binned_spectrum_posteriors1[counter, :] = binned_spectrum1
        full_spectrum_posteriors1[counter, :] = spectrum1

        contamination_posteriors2[counter, :] = np.exp(conditional2.sample())
        binned_spectrum_posteriors2[counter, :] = binned_spectrum2
        full_spectrum_posteriors2[counter, :] = spectrum2

        contamination_posteriors3[counter, :] = np.exp(conditional3.sample())
        binned_spectrum_posteriors3[counter, :] = binned_spectrum3
        full_spectrum_posteriors3[counter, :] = spectrum3

        contamination_posteriors4[counter, :] = np.exp(conditional4.sample())
        binned_spectrum_posteriors4[counter, :] = binned_spectrum4
        full_spectrum_posteriors4[counter, :] = spectrum4

        ax1.plot(wavelengths1, np.exp(conditional1.sample()) * binned_spectrum1, color = 'cornflowerblue', alpha = 0.05, zorder = 1) 
        ax2.plot(wavelengths2, np.exp(conditional2.sample()) * binned_spectrum2, color = 'cornflowerblue', alpha = 0.05, zorder = 1)
        ax3.plot(wavelengths3, np.exp(conditional3.sample()) * binned_spectrum3, color = 'cornflowerblue', alpha = 0.05, zorder = 1)
        ax4.plot(wavelengths4, np.exp(conditional4.sample()) * binned_spectrum4, color = 'cornflowerblue', alpha = 0.05, zorder = 1)

        counter += 1

    np.save(out_name+'_full_posterior_spectrum1.npy', full_spectrum_posteriors1)
    np.save(out_name+'_binned_posterior_spectrum1.npy', binned_spectrum_posteriors1)
    np.save(out_name+'_binned_posterior_contamination1.npy', contamination_posteriors1)

    np.save(out_name+'_full_posterior_spectrum2.npy', full_spectrum_posteriors2)
    np.save(out_name+'_binned_posterior_spectrum2.npy', binned_spectrum_posteriors2)
    np.save(out_name+'_binned_posterior_contamination2.npy', contamination_posteriors2)

    np.save(out_name+'_full_posterior_spectrum3.npy', full_spectrum_posteriors3)
    np.save(out_name+'_binned_posterior_spectrum3.npy', binned_spectrum_posteriors3)
    np.save(out_name+'_binned_posterior_contamination3.npy', contamination_posteriors3)

    np.save(out_name+'_full_posterior_spectrum4.npy', full_spectrum_posteriors4)
    np.save(out_name+'_binned_posterior_spectrum4.npy', binned_spectrum_posteriors4)
    np.save(out_name+'_binned_posterior_contamination4.npy', contamination_posteriors4)

else:

    contamination_posteriors1 = np.load(out_name+'_binned_posterior_contamination1.npy')
    binned_spectrum_posteriors1 = np.load(out_name+'_binned_posterior_spectrum1.npy')
    full_spectrum_posteriors1 = np.load(out_name+'_full_posterior_spectrum1.npy')

    contamination_posteriors2 = np.load(out_name+'_binned_posterior_contamination2.npy')
    binned_spectrum_posteriors2 = np.load(out_name+'_binned_posterior_spectrum2.npy')
    full_spectrum_posteriors2 = np.load(out_name+'_full_posterior_spectrum2.npy')

    contamination_posteriors3 = np.load(out_name+'_binned_posterior_contamination3.npy')
    binned_spectrum_posteriors3 = np.load(out_name+'_binned_posterior_spectrum3.npy')
    full_spectrum_posteriors3 = np.load(out_name+'_full_posterior_spectrum3.npy')

    contamination_posteriors4 = np.load(out_name+'_binned_posterior_contamination4.npy')
    binned_spectrum_posteriors4 = np.load(out_name+'_binned_posterior_spectrum4.npy')
    full_spectrum_posteriors4 = np.load(out_name+'_full_posterior_spectrum4.npy')

    for i in range(contamination_posteriors1.shape[0]):

        ax1.plot(wavelengths1, contamination_posteriors1[i, :] * binned_spectrum_posteriors1[i, :], color = 'cornflowerblue', alpha = 0.05, zorder = 1)
        ax2.plot(wavelengths2, contamination_posteriors2[i, :] * binned_spectrum_posteriors2[i, :], color = 'cornflowerblue', alpha = 0.05, zorder = 1)
        ax3.plot(wavelengths3, contamination_posteriors3[i, :] * binned_spectrum_posteriors3[i, :], color = 'cornflowerblue', alpha = 0.05, zorder = 1)
        ax4.plot(wavelengths4, contamination_posteriors4[i, :] * binned_spectrum_posteriors4[i, :], color = 'cornflowerblue', alpha = 0.05, zorder = 1)

    # Plot median models, too:
    ax1.plot(wavelengths1, np.nanmedian(contamination_posteriors1 * binned_spectrum_posteriors1, axis = 0), \
             color = 'blue', zorder = 2)

    ax2.plot(wavelengths2, np.nanmedian(contamination_posteriors2 * binned_spectrum_posteriors2, axis = 0), \
             color = 'blue', zorder = 2)

    ax3.plot(wavelengths3, np.nanmedian(contamination_posteriors3 * binned_spectrum_posteriors3, axis = 0), \
             color = 'blue', zorder = 2)

    ax4.plot(wavelengths4, np.nanmedian(contamination_posteriors4 * binned_spectrum_posteriors4, axis = 0), \
             color = 'blue', zorder = 2)


ax1.set_xlim(np.min(wavelengths1), np.max(wavelengths1))
ax2.set_xlim(np.min(wavelengths2), np.max(wavelengths2))
ax3.set_xlim(np.min(wavelengths3), np.max(wavelengths3))
ax4.set_xlim(np.min(wavelengths4), np.max(wavelengths4))
plt.show()
