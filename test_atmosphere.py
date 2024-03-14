import numpy as np
import transitspectroscopy as ts
import matplotlib.pyplot as plt
import utils
import time
star_properties = {}

star_properties['R'] =  0.11697 # Rsun
star_properties['Teff'] = 2559.0
star_properties['FeH'] = 0.04
star_properties['logg'] = 5.21

planet_properties = {}

planet_properties['R'] = 0.917985 # Rearth
planet_properties['M'] = 0.6356
planet_properties['T_eq'] = 255

bulk_species = ['N2']
param_species = ['O2', 'CO2', 'CH4', 'H2O', 'O3', 'N2O']

w, d, derr = np.loadtxt('data/espinoza/visit1.txt', unpack = True, usecols = (0,1,2))

# Test at R = 10,000
atmosphere = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species)

# Generate spectra 10 times, average time it takes:
timing = []
for i in range(100):
    tic = time.time()
    spectrum = atmosphere.get_spectrum()
    toc = time.time()
    timing.append(toc-tic)
timing = np.array(timing)
print('Gen spectrum at R = 10000', np.mean(timing), '+/-', np.sqrt(np.var(timing)))

# Test at R = 1,000
atmosphere1k = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 1000)

timing = []
for i in range(100):
    tic = time.time()
    spectrum1k = atmosphere1k.get_spectrum()
    toc = time.time()
    timing.append(toc-tic)
timing = np.array(timing)
print('Gen spectrum at R = 1000', np.mean(timing), '+/-', np.sqrt(np.var(timing)))

# Test at R = 100
atmosphere100 = utils.generate_atmosphere(star_properties, planet_properties, param_species, bulk_species, R = 100)

timing = []
for i in range(10):
    tic = time.time()
    spectrum100 = atmosphere100.get_spectrum()
    toc = time.time()
    timing.append(toc-tic)
timing = np.array(timing)
print('Gen spectrum at R = 100', np.mean(timing), '+/-', np.sqrt(np.var(timing)))

# Bin all spectra at wavelengths of a dataset:
plt.plot(w, utils.bin_to_data(w, atmosphere.wl, spectrum), '-', color = 'orangered', label = 'R = 10k, binned to data')
plt.plot(w, utils.bin_to_data(w, atmosphere1k.wl, spectrum1k), '-', color = 'cornflowerblue', label = 'R = 1k, binned to data')
plt.plot(w, utils.bin_to_data(w, atmosphere100.wl, spectrum100), '-', color = 'grey', label = 'R = 100, binned to data')

plt.legend()
plt.show()
