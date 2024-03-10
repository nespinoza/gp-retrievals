# Import classic libraries:
import numpy as np
rstate= np.random.default_rng(56101)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.ndimage import gaussian_filter

# Import dynesty:
import dynesty
from dynesty.utils import resample_equal

# Import celerite2:
import celerite2
from celerite2 import terms

# Import utility functions:
import utils

# Generate noise samples and wavelengths:
true_mean = 500.
true_rho = 0.5
true_gp_sigma = 0.05
true_noise = 200.
true_factor = 0.5

# Set input atmospheric model:
star_properties = {}

star_properties['R'] =  0.11697 # Rsun
star_properties['Teff'] = 2559.0
star_properties['FeH'] = 0.04
star_properties['logg'] = 5.21

planet_properties = {}

planet_properties['R'] = 0.917985 # Rearth
planet_properties['M'] = 0.6356
planet_properties['T_eq'] = 255 

bulk_species = ['H2'] # Hydrogen dominated!
param_species = ['O2', 'CO2', 'CH4', 'H2O', 'O3', 'N2O']
X = np.array([0.21, 3.60E-02, 2.378E-05, 0.01355, 3.901E-09, 3.741E-07])
a, Pcloud = 1., 1e6 
log_a, gamma, log_P_cloud = np.log10(a), -4., np.log10(Pcloud)

# Generate model
planet = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 1000)
planet.set_parameters(250., np.log10(X), [log_a, gamma, log_P_cloud])
tic = time.time()
spectrum = planet.get_spectrum()
toc = time.time()
print('Took to generate spectra:',toc-tic)

tic = time.time()
gf_spectrum = gaussian_filter(spectrum*1e6, 5)
toc = time.time()
print('Took to GF spectra:',toc-tic)

plt.plot(planet.wl, spectrum*1e6, '-')
plt.plot(planet.wl, gf_spectrum, '-')
plt.show()

# Define priors (assumed uniform):
a_mean, b_mean = 0., 20000.
a_rho, b_rho = 0.0, 10.
a_gp_sigma, b_gp_sigma = 0.0, 10000.
a_sigma_w, b_sigma_w = 0., 1000

# Simulate some wavelengths, noise and errorbars (underestimated):
wavelengths = planet.wl#np.linspace(0.6, 5.0, 100)
noise = np.random.normal(0., true_noise, len(wavelengths))
yerr = np.ones(len(wavelengths)) * true_noise * true_factor # errorbars are smaller than they should

# Generate a GP sample from celerite2:
kernel = terms.Matern32Term(sigma = true_gp_sigma, 
                            rho = true_rho, 
                            eps=0.01)

gp = celerite2.GaussianProcess(kernel, mean = 0.0)
gp.compute(wavelengths, yerr = 0.)#yerr + 50.)

# Sample from the (noiseless) GP:
model = 1. + gp.sample()

# Plot:
plt.plot(wavelengths, true_mean + gf_spectrum, 'g-')
plt.plot(wavelengths, true_mean + gf_spectrum * model, 'r-')
plt.plot(wavelengths, true_mean + gf_spectrum * model + noise, '.')
plt.show()

sys.exit()
# All right, let's do some inference! First, set the data:
y = true_mean + model + noise

# Define the prior and the log-likelihood a-la-dynesty:
def loglike(theta):

    mean, rho, gp_sigma, sigma_w = theta
    total_yerr = np.sqrt( yerr**2 + sigma_w**2 )
    
    # Subtract model from data:
    residuals = y - mean

    # Update GP hyperparmeters. First, re-set kernel:
    gp.kernel = terms.Matern32Term(sigma = gp_sigma,
                                   rho = rho,
                                   eps=0.01)

    # Compute:
    gp.compute(wavelengths, diag=yerr**2 + sigma_w**2, quiet=True)

    # Return log-likelihood if compliant with priors:
    if (mean > a_mean and mean < b_mean) and (rho > a_rho and rho < b_rho) and \
       (gp_sigma > a_gp_sigma and gp_sigma < b_gp_sigma) and \
       (sigma_w > a_sigma_w and sigma_w < b_sigma_w):

        return gp.log_likelihood(residuals)

    else:

        return -1e101

def prior_transform(utheta):

    umean, urho, ugp_sigma, usigma_w = utheta

    # Convert from unitary to the priors:
    mean = transform_uniform(umean, [a_mean, b_mean])
    rho = transform_uniform(urho, [a_rho, b_rho])
    gp_sigma = transform_uniform(ugp_sigma, [a_gp_sigma, b_gp_sigma])
    sigma_w = transform_uniform(usigma_w, [a_sigma_w, b_sigma_w])

    return mean, rho, gp_sigma, sigma_w

def transform_uniform(x, hyperparameters):

    a, b = hyperparameters
    return a + (b-a)*x 

# Run dynesty:
dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=4,
                                        bound='multi', sample='rwalk', rstate=rstate)

dsampler.run_nested()
results = dsampler.results 

# Plot some results:
from dynesty import plotting as dyplot

true_sigma_w = true_noise * np.sqrt( 1 - true_factor**2 )
truths = [true_mean, true_rho, true_gp_sigma, true_sigma_w]
labels = [r'$m$', r'$\rho$', r'$\sigma_{GP}$', r'$\sigma_w$']
fig, axes = dyplot.traceplot(results, truths=truths, labels=labels,
                             fig=plt.subplots(4, 2, figsize=(13, 8)))
fig.tight_layout()  
plt.show()

fig, axes = dyplot.cornerplot(results, truths=truths, show_titles=True, 
                              title_kwargs={'y': 1.04}, labels=labels,
                              fig=plt.subplots(4, 4, figsize=(8, 8)))

plt.show()

weights = np.exp(results['logwt'] - results['logz'][-1])
posterior_samples = resample_equal(results.samples, weights)
print(posterior_samples.shape)

plt.plot(wavelengths, true_mean + model, color = 'orangered', label = 'True model', zorder = 3)
plt.plot(wavelengths, y, '.', color = 'black', label = 'Data', zorder = 4, ms = 5)

indexes = np.arange(posterior_samples.shape[0])
idx = np.random.choice(indexes, 1000, replace = False)

for sample in posterior_samples[idx,:]:

    mean, rho, gp_sigma, sigma_w = sample

    # Subtract model from data:
    residuals = y - mean

    # Update GP hyperparmeters. First, re-set kernel:
    gp.kernel = terms.Matern32Term(sigma = gp_sigma,
                                   rho = rho,
                                   eps=0.01)

    # Compute:
    gp.compute(wavelengths, diag=yerr**2 + sigma_w**2, quiet=True)   
    conditional = gp.condition(residuals, wavelengths)
    plt.plot(wavelengths, conditional.sample() + mean, color = 'cornflowerblue', alpha = 0.05, zorder = 1) 

plt.xlim(np.min(wavelengths), np.max(wavelengths))
plt.legend()
plt.show()
