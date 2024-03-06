import numpy as np
from POSEIDON.core import create_star, create_planet, define_model, make_atmosphere, read_opacities, wl_grid_constant_R, compute_spectrum
from POSEIDON.constants import R_Sun, R_E, M_E 

class generate_atmosphere:

    def set_parameters(self, T, log_X, cloud_parameters):

        self.T = T
        self.log_X = log_X
        self.cloud_parameters = cloud_parameters

        PT_params = np.array([T])

        self.atmosphere = make_atmosphere(self.planet, self.model, self.P, self.P_ref, self.R_p_ref,
                                          PT_params, log_X,  cloud_params=cloud_parameters)

    def get_spectrum(self):

        return compute_spectrum(self.planet, self.star, self.model, self.atmosphere, self.opac, self.wl, 
                                spectrum_type = 'transmission')
        

    def __init__(self, star_properties, planet_properties, param_species, bulk_species, 
                 PT_profile = 'isotherm', cloud_model = 'MacMad17', cloud_type = 'deck_haze', cloud_dim = 1, 
                 P_min = 1e-7, P_max = 10, N_layers = 100, P_surf = 1.0, wl_min = 0.5, wl_max = 5.2, R = 10000,     
                 planet_name = 'myplanet', model_name = 'mymodel'):

        # Load stellar properties:
        self.R_s = star_properties['R'] * R_Sun
        self.T_s = star_properties['Teff']
        self.Met_s = star_properties['FeH']
        self.log_g_s = star_properties['logg']

        # Load planet properties:
        self.R_p = planet_properties['R'] * R_E
        self.M_p = planet_properties['M'] * M_E
        self.T_eq = planet_properties['T_eq']

        # Save parameters for posteriety:
        self.param_species = param_species
        self.bulk_species = bulk_species

        # Create star and planet:
        self.star = create_star(self.R_s, self.T_s, self.log_g_s, self.Met_s)
        self.planet = create_planet(planet_name, self.R_p, mass = self.M_p, T_eq = self.T_eq)
        

        self.model = define_model(model_name, bulk_species, param_species, 
                                  PT_profile = PT_profile, 
                                  cloud_model = cloud_model,
                                  cloud_type = cloud_type,
                                  cloud_dim = cloud_dim)

        self.wl = wl_grid_constant_R(wl_min, wl_max, R)

        # Read opacity:
        opacity_treatment = 'opacity_sampling'

        # Define fine temperature grid (K)
        T_fine_min = 100     # 100 K lower limit covers the TRAPPIST-1e P-T profile
        T_fine_max = 300     # 300 K upper limit covers the TRAPPIST-1e P-T profile
        T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

        T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

        # Define fine pressure grid (log10(P/bar))
        log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
        log_P_fine_max = 0.0    # 1 bar is the surface pressure, so no need to go deeper
        log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

        log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), log_P_fine_step)

        self.opac = read_opacities(self.model, self.wl, opacity_treatment, T_fine, log_P_fine, opacity_database = 'Temperate')

        # Set atmosphere --- first, set initial values:
        self.P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)
        self.P_ref = P_surf
        self.R_p_ref = self.R_p

        T_init = 300.
        log_X_init = np.log10( np.array([0.21, 3.60E-02, 2.378E-05, 0.01355, 3.901E-09, 3.741E-07]) )
        a, Pcloud = 1., 1e6
        log_a, gamma, log_P_cloud = np.log10(a), -4., np.log10(Pcloud)
        self.log_a, self.gamma, self.log_P_cloud = np.log10(a), -4., np.log10(Pcloud)

        cloud_params_init = [self.log_a, self.gamma, self.log_P_cloud]

        self.set_parameters(T_init, log_X_init, cloud_params_init)
