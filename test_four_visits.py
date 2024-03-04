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
true_gp_sigma = 1000
true_noise = 200.
true_factor = 0.5

# Define priors (assumed uniform):
a_mean, b_mean = 0., 20000.
a_rho, b_rho = 0.0, 10.
a_gp_sigma, b_gp_sigma = 0.0, 10000.
a_sigma_w, b_sigma_w = 0., 1000

# Simulate some wavelengths, noise and errorbars (underestimated):
wavelengths = np.linspace(0.6, 5.0, 100)
noise1 = np.random.normal(0., true_noise, len(wavelengths))
noise2 = np.random.normal(0., true_noise, len(wavelengths))
noise3 = np.random.normal(0., true_noise, len(wavelengths))
noise4 = np.random.normal(0., true_noise, len(wavelengths))

yerr1 = np.ones(len(wavelengths)) * true_noise * true_factor # errorbars are smaller than they should
yerr2 = yerr1
yerr3 = yerr1
yerr4 = yerr1

# Generate *four* GP samples from celerite2:
kernel = terms.Matern32Term(sigma = true_gp_sigma, 
                            rho = true_rho, 
                            eps=0.01)

gp = celerite2.GaussianProcess(kernel, mean = 0.0)
gp.compute(wavelengths, yerr = 0.)#yerr + 50.)

gp1 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp1.compute(wavelengths, yerr = 0.)

gp2 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp2.compute(wavelengths, yerr = 0.)

gp3 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp3.compute(wavelengths, yerr = 0.)

gp4 = celerite2.GaussianProcess(kernel, mean = 0.0)
gp4.compute(wavelengths, yerr = 0.)

# Sample from the (noiseless) GP:
model1 = gp.sample()
model2 = gp.sample()
model3 = gp.sample()
model4 = gp.sample()

# Plot:
plt.subplot(411)
plt.plot(wavelengths, true_mean + model1)
plt.plot(wavelengths, true_mean + model1 + noise1, '.')

plt.subplot(412)
plt.plot(wavelengths, true_mean + model2)
plt.plot(wavelengths, true_mean + model2 + noise2, '.')

plt.subplot(413)
plt.plot(wavelengths, true_mean + model3)
plt.plot(wavelengths, true_mean + model3 + noise3, '.')

plt.subplot(414)
plt.plot(wavelengths, true_mean + model4)
plt.plot(wavelengths, true_mean + model4 + noise4, '.')

plt.show()

# All right, let's do some inference! First, set the data:
y1 = true_mean + model1 + noise1
y2 = true_mean + model2 + noise2
y3 = true_mean + model3 + noise3
y4 = true_mean + model4 + noise4

# Define the prior and the log-likelihood a-la-dynesty:
def update_gp(gp, wavelengths, yerr, theta):

    gp_sigma, rho, sigma_w = theta

    gp.kernel = terms.Matern32Term(sigma = gp_sigma,
                               rho = rho,
                               eps=0.01)

    gp.compute(wavelengths, diag=yerr**2 + sigma_w**2, quiet=True)

    return gp

def loglike(theta):

    mean1, mean2, mean3, mean4, rho, \
    gp_sigma1, gp_sigma2, gp_sigma3, gp_sigma4, \
    sigma_w1, sigma_w2, sigma_w3, sigma_w4 = theta

    total_yerr1 = np.sqrt( yerr1**2 + sigma_w1**2 )
    total_yerr2 = np.sqrt( yerr2**2 + sigma_w2**2 )
    total_yerr3 = np.sqrt( yerr3**2 + sigma_w3**2 )
    total_yerr4 = np.sqrt( yerr4**2 + sigma_w4**2 )
    
    # Subtract model from data:
    residuals1 = y1 - mean1
    residuals2 = y2 - mean2
    residuals3 = y3 - mean3
    residuals4 = y4 - mean4

    gp1c = update_gp(gp1, wavelengths, yerr1, [gp_sigma1, rho, sigma_w1])
    gp2c = update_gp(gp2, wavelengths, yerr2, [gp_sigma2, rho, sigma_w2])
    gp3c = update_gp(gp3, wavelengths, yerr3, [gp_sigma3, rho, sigma_w3])
    gp4c = update_gp(gp4, wavelengths, yerr4, [gp_sigma4, rho, sigma_w4])

    # Return log-likelihood if compliant with priors:
    if (mean1 > a_mean and mean1 < b_mean) and (mean2 > a_mean and mean2 < b_mean) and \
       (mean3 > a_mean and mean3 < b_mean) and (mean4 > a_mean and mean4 < b_mean) and\
       (rho > a_rho and rho < b_rho) and \
       (gp_sigma1 > a_gp_sigma and gp_sigma1 < b_gp_sigma) and \
       (gp_sigma2 > a_gp_sigma and gp_sigma2 < b_gp_sigma) and \
       (gp_sigma3 > a_gp_sigma and gp_sigma3 < b_gp_sigma) and \
       (gp_sigma4 > a_gp_sigma and gp_sigma4 < b_gp_sigma) and \
       (sigma_w1 > a_sigma_w and sigma_w1 < b_sigma_w) and \
       (sigma_w2 > a_sigma_w and sigma_w2 < b_sigma_w) and \
       (sigma_w3 > a_sigma_w and sigma_w3 < b_sigma_w) and \
       (sigma_w4 > a_sigma_w and sigma_w4 < b_sigma_w) :

        return gp1c.log_likelihood(residuals1) + gp2c.log_likelihood(residuals2) + \
               gp3c.log_likelihood(residuals3) + gp4c.log_likelihood(residuals4) 

    else:

        return -1e101

def prior_transform(utheta):

    umean1, umean2, umean3, umean4, urho, \
    ugp_sigma1, ugp_sigma2, ugp_sigma3, ugp_sigma4, \
    usigma_w1, usigma_w2, usigma_w3, usigma_w4 = utheta

    # Convert from unitary to the priors:
    mean1 = transform_uniform(umean1, [a_mean, b_mean])
    mean2 = transform_uniform(umean2, [a_mean, b_mean])
    mean3 = transform_uniform(umean3, [a_mean, b_mean])
    mean4 = transform_uniform(umean4, [a_mean, b_mean])
    rho = transform_uniform(urho, [a_rho, b_rho])
    gp_sigma1 = transform_uniform(ugp_sigma1, [a_gp_sigma, b_gp_sigma])
    sigma_w1 = transform_uniform(usigma_w1, [a_sigma_w, b_sigma_w])
    gp_sigma2 = transform_uniform(ugp_sigma2, [a_gp_sigma, b_gp_sigma])
    sigma_w2 = transform_uniform(usigma_w2, [a_sigma_w, b_sigma_w])
    gp_sigma3 = transform_uniform(ugp_sigma3, [a_gp_sigma, b_gp_sigma])
    sigma_w3 = transform_uniform(usigma_w3, [a_sigma_w, b_sigma_w])
    gp_sigma4 = transform_uniform(ugp_sigma4, [a_gp_sigma, b_gp_sigma])
    sigma_w4 = transform_uniform(usigma_w4, [a_sigma_w, b_sigma_w])

    return mean1, mean2, mean3, mean4, rho, \
           gp_sigma1, gp_sigma2, gp_sigma3, gp_sigma4, \
           sigma_w1, sigma_w2, sigma_w3, sigma_w4


def transform_uniform(x, hyperparameters):

    a, b = hyperparameters
    return a + (b-a)*x 

# Run dynesty:
dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=13,
                                        bound='multi', sample='rwalk', rstate=rstate)

dsampler.run_nested()
results = dsampler.results 

# Plot some results:
from dynesty import plotting as dyplot

weights = np.exp(results['logwt'] - results['logz'][-1])
posterior_samples = resample_equal(results.samples, weights)
for i in range(posterior_samples.shape[1]):

    print('Parameter ',i,':', np.nanmedian(posterior_samples[:, i]), '+/-', np.sqrt(np.var(posterior_samples[:, i])))

plt.hist(posterior_samples[:, 0], bins = 300)
plt.hist(posterior_samples[:, 1], bins = 300)
plt.hist(posterior_samples[:, 2], bins = 300)
plt.hist(posterior_samples[:, 3], bins = 300)
plt.xlabel('mean')
plt.show()

plt.hist(posterior_samples[:, 4], bins = 300)
plt.xlabel(r'$\rho$')
plt.show()

plt.hist(posterior_samples[:, 5], bins = 300)
plt.hist(posterior_samples[:, 6], bins = 300)
plt.hist(posterior_samples[:, 7], bins = 300)
plt.hist(posterior_samples[:, 8], bins = 300)
plt.xlabel('gpsigma')
plt.show()

plt.hist(posterior_samples[:, 9], bins = 300)
plt.hist(posterior_samples[:, 10], bins = 300)
plt.hist(posterior_samples[:, 11], bins = 300)
plt.hist(posterior_samples[:, 12], bins = 300)
plt.xlabel('sigma_w')
plt.show()

# Plot:
ax1 = plt.subplot(411)
plt.plot(wavelengths, true_mean + model1)
plt.plot(wavelengths, true_mean + model1 + noise1, '.')

ax2 = plt.subplot(412)
plt.plot(wavelengths, true_mean + model2)
plt.plot(wavelengths, true_mean + model2 + noise2, '.')

ax3 = plt.subplot(413)
plt.plot(wavelengths, true_mean + model3)
plt.plot(wavelengths, true_mean + model3 + noise3, '.')

ax4 = plt.subplot(414)
plt.plot(wavelengths, true_mean + model4)
plt.plot(wavelengths, true_mean + model4 + noise4, '.')

indexes = np.arange(posterior_samples.shape[0])
idx = np.random.choice(indexes, 1000, replace = False)

for sample in posterior_samples[idx,:]:

    mean1, mean2, mean3, mean4, rho, \
    gp_sigma1, gp_sigma2, gp_sigma3, gp_sigma4, \
    sigma_w1, sigma_w2, sigma_w3, sigma_w4 = sample

    # Subtract model from data:
    residuals1 = y1 - mean1
    residuals2 = y2 - mean2
    residuals3 = y3 - mean3
    residuals4 = y4 - mean4    

    # Update GP hyperparmeters. First, re-set kernel:
    gp1 = update_gp(gp1, wavelengths, yerr1, [gp_sigma1, rho, sigma_w1])
    gp2 = update_gp(gp2, wavelengths, yerr2, [gp_sigma2, rho, sigma_w2])
    gp3 = update_gp(gp3, wavelengths, yerr3, [gp_sigma3, rho, sigma_w3])
    gp4 = update_gp(gp4, wavelengths, yerr4, [gp_sigma4, rho, sigma_w4])

    conditional1 = gp1.condition(residuals1, wavelengths)
    conditional2 = gp2.condition(residuals2, wavelengths)
    conditional3 = gp3.condition(residuals3, wavelengths)
    conditional4 = gp4.condition(residuals4, wavelengths)

    ax1.plot(wavelengths, conditional1.sample() + mean1, color = 'cornflowerblue', alpha = 0.05, zorder = 1) 
    ax2.plot(wavelengths, conditional2.sample() + mean2, color = 'cornflowerblue', alpha = 0.05, zorder = 1)
    ax3.plot(wavelengths, conditional3.sample() + mean3, color = 'cornflowerblue', alpha = 0.05, zorder = 1)
    ax4.plot(wavelengths, conditional4.sample() + mean4, color = 'cornflowerblue', alpha = 0.05, zorder = 1)


ax1.set_xlim(np.min(wavelengths), np.max(wavelengths))
ax2.set_xlim(np.min(wavelengths), np.max(wavelengths))
ax3.set_xlim(np.min(wavelengths), np.max(wavelengths))
ax4.set_xlim(np.min(wavelengths), np.max(wavelengths))
plt.legend()
plt.show()
