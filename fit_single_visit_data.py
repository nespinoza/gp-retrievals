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
import dynesty.pool as dypool
from dynesty.utils import resample_equal

# Import celerite2:
import celerite2
from celerite2 import terms

# Utilities function:
import utils

show_plots = True
multiprocessing = False
nthreads = 30
# First things first, get data out; use it to steal the wavelengths:
visit = 'visit1'

wavelengths1, depths1, depths_error1 = np.loadtxt('data/espinoza/'+visit+'.txt', unpack = True, usecols = (0, 1, 2))

out_name = 'data_single_'+visit+'_gp_atmosphere'

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

# Haze properties fixed:
log_a, gamma = np.log10(1.), -4

# Generate planet:
if not os.path.exists(out_name+'_full_posterior_spectrum.npy'):

    planet = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 500)

else:

    planet = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 10)

wavelengths_model = planet.wl

# Define priors (assumed uniform):
a_T, b_T = 100.,300.
a_CO2, b_CO2 = 0., 1.0
a_Pcloud, b_Pcloud = 1e-6, 1.
a_offset, b_offset = -3000,3000
a_rho, b_rho = 0.0, 100.
a_gp_sigma, b_gp_sigma = 0.0,10. 
a_sigma_w, b_sigma_w = 0.0, 1000

# Initialize GP sample from celerite2; initialize GP objects for each dataset:
kernel = terms.Matern32Term(sigma = 1.0, 
                            rho = 0.1, 
                            eps=0.01)

gp1 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp1.compute(wavelengths1, yerr = 0.)

if show_plots:

    plt.errorbar(wavelengths1, depths1, depths_error1, fmt = 'k.')

    plt.show()

real_y1 = depths1
logy1 = np.log( real_y1 )
logy_err1 = depths_error1 / real_y1

# Define the prior and the log-likelihood a-la-dynesty:
def loglike(theta):

    T, CO2, Pcloud, offset1, rho1, gp_sigma1, sigma_w1 = theta

    total_variance1 = ( logy_err1**2 + (sigma_w1 / real_y1)**2 )
    
    # Generate transmission spectrum:
    #                           O2   CO2  CH4  H2O  O3   N2O
    log_X = np.log10( np.array([0.0, CO2, 0.0, 0.0, 0.0, 0.0]) )

    # Set abundances, clouds properties, generate spectrum:
    planet.set_parameters(T, log_X, [log_a, gamma, np.log10(Pcloud)])
    spectrum1 = gaussian_filter(offset1 + planet.get_spectrum() * 1e6, 5)
    # Bin spectrum to the data res:
    binned_spectrum1 = utils.bin_to_data(wavelengths1, wavelengths_model, spectrum1)
    
    # Subtract model from data:
    residuals1 = logy1 - np.log(binned_spectrum1)

    # Update GP hyperparmeters. First, re-set kernel:
    gp1.kernel = terms.Matern32Term(sigma = gp_sigma1,
                                   rho = rho1,
                                   eps=1e-6)

    # Compute:
    gp1.compute(wavelengths1, diag=total_variance1, quiet=True)

    #print('In loglike:', theta)
    #print('Residuals:', residuals)
    # Return log-likelihood if compliant with priors:
    if (T > a_T and T < b_T) and (CO2 > a_CO2 and CO2 < b_CO2) and (Pcloud > a_Pcloud and Pcloud < b_Pcloud) and \
       (offset1 > a_offset and offset1 < b_offset) and (rho1 > a_rho and rho1 < b_rho) and \
       (gp_sigma1 > a_gp_sigma and gp_sigma1 < b_gp_sigma) and \
       (sigma_w1 > a_sigma_w and sigma_w1 < b_sigma_w):

        return gp1.log_likelihood(residuals1)

    else:

        return -1e101

def prior_transform(utheta):

    uT, uCO2, uPcloud, uoffset1, urho1, ugp_sigma1, usigma_w1 = utheta

    # Convert from unitary to the priors:
    T = transform_uniform(uT, [a_T, b_T])
    CO2 = transform_uniform(uCO2, [a_CO2, b_CO2])
    Pcloud = transform_loguniform(uPcloud, [a_Pcloud, b_Pcloud])
    offset1 = transform_uniform(uoffset1, [a_offset, b_offset])
    rho1 = transform_uniform(urho1, [a_rho, b_rho])
    gp_sigma1 = transform_uniform(ugp_sigma1, [a_gp_sigma, b_gp_sigma])
    sigma_w1 = transform_uniform(usigma_w1, [a_sigma_w, b_sigma_w])

    #print('In prior transform:', T, CO2, Pcloud, offset, rho, gp_sigma, sigma_w)
    return T, CO2, Pcloud, offset1, rho1, gp_sigma1, sigma_w1

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

    if not multiprocessing:

        dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=7,
                                                bound='multi', sample='rwalk', rstate=rstate)

        dsampler.run_nested()

    else:

        with dypool.Pool(nthreads, loglike, prior_transform) as pool:

            dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim=7,
                                                bound='multi', sample='rwalk', pool = pool, rstate=rstate)

        dsampler.run_nested()

    results = dsampler.results
    out = {}

    out['dynesty_results'] = results
    pickle.dump(out, open(out_name+'_results.pkl', 'wb'))

else:

    out = pickle.load(open(out_name+'_results.pkl', 'rb'))
    results = out['dynesty_results']


print(results.samples.shape)

weights = np.exp(results['logwt'] - results['logz'][-1])
posterior_samples = resample_equal(results.samples, weights)

print('Computing posterior model plot...')

if show_plots:

    ax1 = plt.subplot()
    plt.errorbar(wavelengths1, depths1, depths_error1, fmt = 'k.')

indexes = np.arange(posterior_samples.shape[0])
idx = np.random.choice(indexes, 1000, replace = False)

if not os.path.exists(out_name+'_full_posterior_spectrum4.npy'):

    counter = 0
    contamination_posteriors1 = np.zeros([len(idx), len(wavelengths1)])
    binned_spectrum_posteriors1 = np.zeros([len(idx), len(wavelengths1)])
    full_spectrum_posteriors1 = np.zeros([len(idx), len(wavelengths_model)])

    for sample in posterior_samples[idx,:]:

        T, CO2, Pcloud, offset1, rho1, gp_sigma1, sigma_w1 = sample

        total_variance1 = ( logy_err1**2 + (sigma_w1 / real_y1)**2 )

        # Generate transmission spectrum:
        #                           O2   CO2  CH4  H2O  O3   N2O
        log_X = np.log10( np.array([0.0, CO2, 0.0, 0.0, 0.0, 0.0]) )

        # Set abundances, clouds properties, generate spectrum:
        planet.set_parameters(T, log_X, [log_a, gamma, np.log10(Pcloud)])
        spectrum1 = gaussian_filter(offset1 + planet.get_spectrum() * 1e6, 5)
        # Bin spectrum to the data res:
        binned_spectrum1 = utils.bin_to_data(wavelengths1, wavelengths_model, spectrum1)

        # Subtract model from data:
        residuals1 = logy1 - np.log(binned_spectrum1)

        # Update GP hyperparmeters. First, re-set kernel:
        gp1.kernel = terms.Matern32Term(sigma = gp_sigma1,
                                       rho = rho1,
                                       eps=1e-6)

        # Compute:
        gp1.compute(wavelengths1, diag=total_variance1, quiet=True)

        conditional1 = gp1.condition(residuals1, wavelengths1)

        contamination_posteriors1[counter, :] = np.exp(conditional1.sample())
        binned_spectrum_posteriors1[counter, :] = binned_spectrum1
        full_spectrum_posteriors1[counter, :] = spectrum1

        if show_plots:

            ax1.plot(wavelengths1, binned_spectrum1, color = 'orangered', alpha = 0.05, zorder = 1)

            ax1.plot(wavelengths1, np.exp(conditional1.sample()) * binned_spectrum1, color = 'cornflowerblue', alpha = 0.05, zorder = 1) 

        counter += 1

    np.save(out_name+'_full_posterior_spectrum1.npy', full_spectrum_posteriors1)
    np.save(out_name+'_binned_posterior_spectrum1.npy', binned_spectrum_posteriors1)
    np.save(out_name+'_binned_posterior_contamination1.npy', contamination_posteriors1)

else:

    contamination_posteriors1 = np.load(out_name+'_binned_posterior_contamination1.npy')
    binned_spectrum_posteriors1 = np.load(out_name+'_binned_posterior_spectrum1.npy')
    full_spectrum_posteriors1 = np.load(out_name+'_full_posterior_spectrum1.npy')

    if show_plots:

        for i in range(contamination_posteriors1.shape[0]):

            ax1.plot(wavelengths1, binned_spectrum_posteriors1[i, :], color = 'orangered', alpha = 0.05, zorder = 1)

            ax1.plot(wavelengths1, contamination_posteriors1[i, :] * binned_spectrum_posteriors1[i, :], color = 'cornflowerblue', alpha = 0.05, zorder = 1)

        # Plot median models, too:
        ax1.plot(wavelengths1, np.nanmedian(contamination_posteriors1 * binned_spectrum_posteriors1, axis = 0), \
                 color = 'blue', zorder = 2)

if show_plots:

    ax1.set_xlim(np.min(wavelengths1), np.max(wavelengths1))
    plt.show()
