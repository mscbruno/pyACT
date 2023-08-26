# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:56:51 2022

@author: LPC
"""

import numpy as np
import matplotlib.pyplot as plt
import pyMechACT as pma
from lmfit import minimize, Parameters, report_fit

# Call input file

input_filename = 'inputs/TiO2.csv'
savefile = 1                                       # 1 for yes, anything else for no
output_filename = 'outputs/TiO2_params.csv'
input_data = pma.read_input(input_filename)

# Define parameters
# Mark -1 if not using any of the listed mechanisms

OH_radicals = 1
h_reactions = 1
SingOx_reactions = 1
Special_sites = 1

opt_algo = 'nelder'

##############################################################################
# List of parameters

MM_ACT = 151.163                        # g/mol
ACT_0 = input_data[1][0]

x0 = [input_data[1][0]/MM_ACT*1e-3,                # initial concentrations
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0]

params = Parameters()
# Initial concentrations
params.add('A0', value=x0[0], vary=False)
params.add('OH0', value=x0[1], vary=False)
params.add('R20', value=x0[2], vary=False)
params.add('R30', value=x0[3], vary=False)
params.add('R40', value=x0[4], vary=False)
params.add('R50', value=x0[5], vary=False)
params.add('R60', value=x0[6], vary=False)
params.add('R70', value=x0[7], vary=False)
params.add('R80', value=x0[8], vary=False)
params.add('R90', value=x0[9], vary=False)
params.add('R100', value=x0[10], vary=False)
params.add('R110', value=x0[11], vary=False)
params.add('R120', value=x0[12], vary=False)
params.add('R130', value=x0[13], vary=False)
params.add('R140', value=x0[14], vary=False)
params.add('h0', value=x0[15], vary=False)
params.add('phi0', value=x0[16], vary=False)
params.add('SS_ad0', value=x0[17], vary=False)
params.add('el0', value=x0[18], vary=False)
params.add('OxSin0', value=x0[19], vary=False)

# Inputs to function
params.add('wavelength',  value=input_data[7][0], vary=False)
params.add('irradiance',  value=input_data[5][0], vary=False)
params.add('absorbance',  value=input_data[6][0], vary=False)
params.add('SSA',         value=input_data[2][0], vary=False)
params.add('G_SS',        value=input_data[3][0], vary=False)
params.add('Special_sites', value=Special_sites, vary=False)


# Optimization variables
params.add('k_eh',  value=1e-4, min=1e-12, max=1e-3)
params.add('k_fouling', value=3e2, min=1e-1, max=1e5)
params.add('k_hole_r', value=1e8, min=1e5, max=1e10)
params.add('k_loss', value=1e6, min=1e3, max=1e10)
params.add('k_wash', value=1e0, min=1e-3, max=1e3)
params.add('k_as', value=1e-5, min=1e-7, max=1e2)
params.add('QY_SOx', value=3e-2, min=3e-3, max=4e-1)

# Experimental data
x_exp = input_data[0]
y_exp = np.array(input_data[1])
y_norm = y_exp/y_exp[0]

x_sec = [time*60 for time in x_exp]
y_mol = [C/MM_ACT*1e-3 for C in y_exp]

t_sim = np.linspace(0, 7200, 7200)
t_sim_min = t_sim/60

# Fit model
result = minimize(pma.residual, params, args=(x_sec, y_mol), method=opt_algo)  

# check results of the fit
x = pma.g(t_sim, x0, result.params)
ACT_fit = x[:,0]/x[0,0]
report_fit(result)

# test second minimization
#result.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

#res = lmfit.minimize(pma.residual, args=(x_sec, y_mol), method='leastsq', nan_policy='omit', burn=300, steps=1000, thin=20,
 #                   params=result.params, is_weighted=False, progress=False)

# Plot the results
plt.figure()
plt.scatter(x_sec, y_norm, marker='o', color='b', label='measured data', s=75)
plt.plot(t_sim, ACT_fit, '-', linewidth=2, color='red', label='fitted data')
plt.style.use('bmh')

# =============================================================================
# plt.figure()
# plt.plot(res.acceptance_fraction, 'o')
# plt.xlabel('walker')
# plt.ylabel('acceptance fraction')
# plt.show()
# =============================================================================

figure, axis = plt.subplots(2,3)

axis[0,0].plot(t_sim_min,ACT_fit, label='ACT')
axis[0,0].plot(x_exp,y_norm, 'r*')

axis[1,0].plot(t_sim_min,x[:,1], label='OH')
axis[1,1].plot(t_sim_min,x[:,15], label='holes')
axis[1,2].plot(t_sim_min,x[:,17], label='occupancy')

axis[0,1].plot(t_sim_min,x[:,16], label='Phi1')

axis[0,2].plot(t_sim_min,x[:,2], label='3-OHACT')
axis[0,2].plot(t_sim_min,x[:,3], label='2-OHACT')
axis[0,2].plot(t_sim_min,x[:,4], label='HQ')

axis[0,0].legend(loc='upper right')
axis[0,1].legend(loc='upper right')
axis[0,2].legend(loc='upper right')
axis[1,0].legend(loc='upper right')
axis[1,1].legend(loc='upper right')
axis[1,2].legend(loc='upper right')

axis[1,1].set_xlabel ("Elapsed time [min]")
axis[0,0].set_ylabel ("C/C0")
axis[1,0].set_ylabel ("Concentration (mM)")

plt.show()

# Data processing and export

h_avg = np.mean(x, axis=0)[15]
OH_avg = np.mean(x, axis=0)[1]
OxSing_avg = np.mean(x, axis=0)[19]

res_dict = result.params.valuesdict()
relevant_keys = ['k_eh', 
                 'k_fouling', 
                 'k_hole_r', 
                 'k_loss', 
                 'k_wash', 
                 'k_as', 
                 'QY_SOx']
opt_params = { keys: res_dict[keys] for keys in relevant_keys }
opt_params.update({'h_avg':h_avg,'OH_avg':OH_avg,'1O2_avg':OxSing_avg})

fit_model = ['t_sim_min', 'ACT_fit']

if savefile == 1:
    outfile = pma.write_output(output_filename, opt_params)

#emcee_plot = corner.corner(res.flatchain, labels=res.var_names,truths=list(res.params.valuesdict().values()))
