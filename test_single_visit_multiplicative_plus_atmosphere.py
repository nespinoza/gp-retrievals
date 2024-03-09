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

out_name = 'single_visit_gp_atmosphere'

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
planet = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 500)

wavelengths_model = planet.wl

# Set abundances, clouds properties:
planet.set_parameters(true_T, true_log_X, [log_a, gamma, log_P_cloud])
true_spectrum = gaussian_filter(true_offset + planet.get_spectrum() * 1e6, 5)

# All right, now, bin the model to match the wavelengths of the data:

# Generate noise samples and wavelengths:
true_rho = 0.5
true_gp_sigma = 0.01
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
noise = np.random.normal(0., true_noise, len(wavelengths1))
yerr = np.ones(len(wavelengths1)) * true_noise * true_factor # errorbars are smaller than they should

# Generate a GP sample from celerite2:
kernel = terms.Matern32Term(sigma = true_gp_sigma, 
                            rho = true_rho, 
                            eps=0.01)

true_gp = celerite2.GaussianProcess(kernel, mean = 1.0)
true_gp.compute(wavelengths_model, yerr = 0.)#yerr + 50.)

# Sample from the (noiseless) GP:
model = true_gp.sample()

gp = celerite2.GaussianProcess(kernel, mean = 0.0)
gp.compute(wavelengths1, yerr = 0.)

if os.path.exists(out_name+'_model.txt'):

    wavelengths_model, true_spectrum, model = np.loadtxt(out_name+'_model.txt', unpack = True, usecols = (0,1,2))

else:

    fout = open(out_name+'_model.txt', 'w')
    for i in range(len(wavelengths_model)):

        fout.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(wavelengths_model[i], true_spectrum[i], model[i]))

    fout.close()

plt.title('True spectrum:')
plt.plot(wavelengths_model, true_spectrum)
plt.show()

plt.subplot(211)
plt.title('Original')
plt.plot(wavelengths_model, true_spectrum*model, label = 'With stellar contamination')
plt.plot(wavelengths_model, true_spectrum, label = 'Original spectra')

# Bin models to match data:
binned_true_spectrum = utils.bin_to_data(wavelengths1, wavelengths_model, true_spectrum)
binned_contaminated_true_spectrum = utils.bin_to_data(wavelengths1, wavelengths_model, true_spectrum * model)
print('Binned datapoints:',len(binned_true_spectrum))
print('Model datapoints:', len(wavelengths_model))
# Save fake data:
if os.path.exists(out_name+'_data.txt'):

    wavelengths1, binned_true_spectrum, binned_contaminated_true_spectrum, noise = np.loadtxt(out_name+'_data.txt', unpack = True, usecols = (0,1,2,3))

else:

    fout = open(out_name+'_data.txt', 'w')
    for i in range(len(wavelengths1)):

        fout.write('{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n'.format(wavelengths1[i], binned_true_spectrum[i], binned_contaminated_true_spectrum[i], noise[i]))

    fout.close()

plt.errorbar(wavelengths1, binned_contaminated_true_spectrum, yerr, fmt = '.', label = 'Binned w/ stellar contamination')
plt.errorbar(wavelengths1, binned_contaminated_true_spectrum + noise, yerr, fmt = '.', label = 'Binned w/ stellar contamination + noise')
plt.legend()

# Plot:
plt.subplot(212)
plt.title('Log-version:')
plt.plot(wavelengths_model, np.log(true_spectrum*model))
plt.plot(wavelengths_model, np.log(true_spectrum))
plt.errorbar(wavelengths1, np.log(binned_contaminated_true_spectrum + noise), yerr / (binned_contaminated_true_spectrum + noise), fmt = '.')
#plt.plot(wavelengths, np.log(true_mean*model + noise), '.')
plt.legend()
plt.show()

plt.plot(wavelengths1, np.log(binned_contaminated_true_spectrum + noise) - np.log(binned_true_spectrum), label = 'log-data - log(true mean) (binned)')
plt.plot(wavelengths_model, np.log(model), label = 'log-GP')
plt.legend()
plt.show()

# All right, let's do some inference! First, set the data:
real_y = binned_contaminated_true_spectrum + noise#true_spectrum*model + noise
logy = np.log( real_y )
logy_err = yerr / real_y

# Define the prior and the log-likelihood a-la-dynesty:
def loglike(theta):

    T, CO2, Pcloud, offset, rho, gp_sigma, sigma_w = theta
    total_variance = ( logy_err**2 + (sigma_w / real_y)**2 )
    
    # Generate transmission spectrum:
    #                           O2   CO2  CH4  H2O  O3   N2O
    log_X = np.log10( np.array([0.0, CO2, 0.0, 0.0, 0.0, 0.0]) )

    # Set abundances, clouds properties, generate spectrum:
    planet.set_parameters(T, log_X, [log_a, gamma, np.log10(Pcloud)])
    spectrum = gaussian_filter(offset + planet.get_spectrum() * 1e6, 5)
    # Bin spectrum to the data res:
    binned_spectrum = utils.bin_to_data(wavelengths1, wavelengths_model, spectrum)
    
    # Subtract model from data:
    residuals = logy - np.log(binned_spectrum)

    # Update GP hyperparmeters. First, re-set kernel:
    gp.kernel = terms.Matern32Term(sigma = gp_sigma,
                                   rho = rho,
                                   eps=1e-6)

    # Compute:
    gp.compute(wavelengths1, diag=total_variance, quiet=True)

    #print('In loglike:', theta)
    #print('Residuals:', residuals)
    # Return log-likelihood if compliant with priors:
    if (T > a_T and T < b_T) and (CO2 > a_CO2 and CO2 < b_CO2) and (Pcloud > a_Pcloud and Pcloud < b_Pcloud) and \
       (offset > a_offset and offset < b_offset) and (rho > a_rho and rho < b_rho) and \
       (gp_sigma > a_gp_sigma and gp_sigma < b_gp_sigma) and \
       (sigma_w > a_sigma_w and sigma_w < b_sigma_w):

        return gp.log_likelihood(residuals)

    else:

        return -1e101

def prior_transform(utheta):

    uT, uCO2, uPcloud, uoffset, urho, ugp_sigma, usigma_w = utheta

    # Convert from unitary to the priors:
    T = transform_uniform(uT, [a_T, b_T])
    CO2 = transform_uniform(uCO2, [a_CO2, b_CO2])
    Pcloud = transform_loguniform(uPcloud, [a_Pcloud, b_Pcloud])
    offset = transform_uniform(uoffset, [a_offset, b_offset])
    rho = transform_uniform(urho, [a_rho, b_rho])
    gp_sigma = transform_uniform(ugp_sigma, [a_gp_sigma, b_gp_sigma])
    sigma_w = transform_uniform(usigma_w, [a_sigma_w, b_sigma_w])

    #print('In prior transform:', T, CO2, Pcloud, offset, rho, gp_sigma, sigma_w)
    return T, CO2, Pcloud, offset, rho, gp_sigma, sigma_w

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

    dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=7,
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
truths = [true_T, true_CO2, true_Pcloud, true_offset, true_rho, true_gp_sigma, true_sigma_w]
labels = [r'$T$ (K)', r'$CO2$', r'$P_{cloud}$' , r'Offset (ppm)', r'$\rho$', r'$\sigma_{GP}$', r'$\sigma_w$']
fig, axes = dyplot.traceplot(results, truths=truths, labels=labels,
                             fig=plt.subplots(7, 2, figsize=(13, 8)))
fig.tight_layout()  
plt.show()

fig, axes = dyplot.cornerplot(results, truths=truths, show_titles=True, 
                              title_kwargs={'y': 1.04}, labels=labels,
                              fig=plt.subplots(7, 7, figsize=(8, 8)))

plt.show()

weights = np.exp(results['logwt'] - results['logz'][-1])
posterior_samples = resample_equal(results.samples, weights)

print('Computing posterior model plot...')

plt.plot(wavelengths_model, true_spectrum, color = 'black', label = 'True model (w/o stellar contamination)', zorder = 3)
plt.plot(wavelengths_model, true_spectrum*model, color = 'orangered', label = 'True model (w/ stellar contamination)', zorder = 3)
plt.plot(wavelengths1, binned_contaminated_true_spectrum, color = 'red', lw = 2, label = 'True model (w/ stellar contamination, binned)', zorder = 3)
plt.plot(wavelengths1, real_y, '.', color = 'black', label = 'Data', zorder = 4, ms = 5)

indexes = np.arange(posterior_samples.shape[0])
idx = np.random.choice(indexes, 1000, replace = False)

if not os.path.exists(out_name+'_full_posterior_spectrum.npy'):

    counter = 0
    contamination_posteriors1 = np.zeros([len(idx), len(wavelengths1)])
    binned_spectrum_posteriors1 = np.zeros([len(idx), len(wavelengths1)])
    full_spectrum_posteriors1 = np.zeros([len(idx), len(wavelengths_model)])
    for sample in posterior_samples[idx,:]:

        T, CO2, Pcloud, offset, rho, gp_sigma, sigma_w = sample
        total_variance = ( logy_err**2 + (sigma_w / real_y)**2 )
           
        # Generate transmission spectrum:
        #                           O2   CO2  CH4  H2O  O3   N2O
        log_X = np.log10( np.array([0.0, CO2, 0.0, 0.0, 0.0, 0.0]) )

        # Set abundances, clouds properties:
        planet.set_parameters(T, log_X, [log_a, gamma, np.log10(Pcloud)])
        spectrum = gaussian_filter(offset + planet.get_spectrum() * 1e6, 5)
        binned_spectrum = utils.bin_to_data(wavelengths1, wavelengths_model, spectrum)

        # Subtract model from data:
        residuals = logy - np.log(binned_spectrum)

        # Update GP hyperparmeters. First, re-set kernel:
        gp.kernel = terms.Matern32Term(sigma = gp_sigma,
                                       rho = rho,
                                       eps = 0.01)

        # Compute:
        gp.compute(wavelengths1, diag=total_variance, quiet=True)   
        conditional = gp.condition(residuals, wavelengths1)

        contamination_posteriors1[counter, :] = np.exp(conditional.sample())
        binned_spectrum_posteriors1[counter, :] = binned_spectrum
        full_spectrum_posteriors1[counter, :] = spectrum
        plt.plot(wavelengths1, np.exp(conditional.sample()) * binned_spectrum, color = 'cornflowerblue', alpha = 0.05, zorder = 1) 
        counter += 1

    np.save(out_name+'_full_posterior_spectrum.npy', full_spectrum_posteriors1)
    np.save(out_name+'_binned_posterior_spectrum.npy', binned_spectrum_posteriors1)
    np.save(out_name+'_binned_posterior_contamination.npy', contamination_posteriors1)

else:

    contamination_posteriors1 = np.load(out_name+'_binned_posterior_contamination.npy')
    binned_spectrum_posteriors1 = np.load(out_name+'_binned_posterior_spectrum.npy')
    full_spectrum_posteriors1 = np.load(out_name+'_full_posterior_spectrum.npy')

    for i in range(contamination_posteriors1.shape[0]):

        plt.plot(wavelengths1, contamination_posteriors1[i, :] * binned_spectrum_posteriors1[i, :], color = 'cornflowerblue', alpha = 0.05, zorder = 1)

plt.xlim(np.min(wavelengths1), np.max(wavelengths1))
plt.legend()
plt.show()
