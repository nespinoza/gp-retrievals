# Import classic libraries:
import numpy as np
rstate= np.random.default_rng(56101)
import matplotlib.pyplot as plt
import seaborn as sns

# Import dynesty:
import dynesty
from dynesty.utils import resample_equal

# Import celerite2:
import celerite2
from celerite2 import terms

# Generate noise samples and wavelengths:
true_mean = 7000.
true_rho = 0.5
true_gp_sigma = 0.05
true_noise = 200.
true_factor = 0.5

# Define priors (assumed uniform):
a_mean, b_mean = 0., 20000.
a_rho, b_rho = 0.0, 100.
a_gp_sigma, b_gp_sigma = 0.0,10. 
a_sigma_w, b_sigma_w = 0.0, 1000

# Simulate some wavelengths, noise and errorbars (underestimated):
wavelengths = np.linspace(0.6, 5.0, 100)
noise = np.random.normal(0., true_noise, len(wavelengths))
yerr = np.ones(len(wavelengths)) * true_noise * true_factor # errorbars are smaller than they should

# Generate a GP sample from celerite2:
kernel = terms.Matern32Term(sigma = true_gp_sigma, 
                            rho = true_rho, 
                            eps=0.01)

true_gp = celerite2.GaussianProcess(kernel, mean = 1.0)
true_gp.compute(wavelengths, yerr = 0.)#yerr + 50.)

# Sample from the (noiseless) GP:
model = true_gp.sample()

gp = celerite2.GaussianProcess(kernel, mean = 0.0)
gp.compute(wavelengths, yerr = 0.)

plt.subplot(211)
plt.title('Original')
plt.plot(wavelengths, true_mean*model)
#plt.plot(wavelengths, true_mean*model + noise, '.')
plt.errorbar(wavelengths, true_mean*model + noise, yerr ,fmt = '.')

# Plot:
plt.subplot(212)
plt.title('Log-version:')
plt.plot(wavelengths, np.log(true_mean*model))
plt.errorbar(wavelengths, np.log(true_mean*model + noise), yerr / (true_mean*model + noise), fmt = '.')
#plt.plot(wavelengths, np.log(true_mean*model + noise), '.')
plt.show()

plt.plot(wavelengths, np.log(true_mean*model + noise) - np.log(true_mean), label = 'log-data - log(true mean)')
plt.plot(wavelengths, np.log(model), label = 'log-GP')
plt.legend()
plt.show()

# All right, let's do some inference! First, set the data:
real_y = true_mean*model + noise
logy = np.log( real_y )
logy_err = yerr / real_y

# Define the prior and the log-likelihood a-la-dynesty:
def loglike(theta):

    mean, rho, gp_sigma, sigma_w = theta
    total_variance = ( logy_err**2 + (sigma_w / real_y)**2 )
    
    # Subtract model from data:
    residuals = logy - np.log(mean)

    # Update GP hyperparmeters. First, re-set kernel:
    gp.kernel = terms.Matern32Term(sigma = gp_sigma,
                                   rho = rho,
                                   eps=1e-6)

    # Compute:
    gp.compute(wavelengths, diag=total_variance, quiet=True)

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

def transform_loguniform(x, hyperparameters):
    a, b = hyperparameters
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la + x * (lb - la))

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

plt.plot(wavelengths, true_mean*model, color = 'orangered', label = 'True model', zorder = 3)
plt.plot(wavelengths, real_y, '.', color = 'black', label = 'Data', zorder = 4, ms = 5)

indexes = np.arange(posterior_samples.shape[0])
idx = np.random.choice(indexes, 1000, replace = False)

for sample in posterior_samples[idx,:]:

    mean, rho, gp_sigma, sigma_w = sample

    # Get residuals:
    total_variance = ( logy_err**2 + (sigma_w / real_y)**2 )
       
    # Subtract model from data:
    residuals = logy - np.log(mean)

    # Update GP hyperparmeters. First, re-set kernel:
    gp.kernel = terms.Matern32Term(sigma = gp_sigma,
                                   rho = rho,
                                   eps=0.01)

    # Compute:
    gp.compute(wavelengths, diag=total_variance, quiet=True)   
    conditional = gp.condition(residuals, wavelengths)
    plt.plot(wavelengths, np.exp(conditional.sample()) * mean, color = 'cornflowerblue', alpha = 0.05, zorder = 1) 

plt.xlim(np.min(wavelengths), np.max(wavelengths))
plt.legend()
plt.show()
