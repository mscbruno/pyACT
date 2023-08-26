# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:38:25 2022
Kinetics of ACT degradation by TiO2 photocatalysis
@author: Bruno
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns
from lmfit import minimize, Parameters, Parameter, report_fit


# inputs
wavelenght = 365                    # in nm
irradiance = 5                    # in mW/cm2
absorbance = 1e-7                    # in cm2/g, of the catalyst 
SSA = 45                            # specific surface area, in m2/g
V_sol = 0.005                       # volume of solution, in L
m_cat = 10                          # mass of catalyst, in mg
Q_eh = 1                            # quantum yield, mol of e-h pairs/eistein
W_cat = m_cat/V_sol                 # catalyst load, in mg/dm3
ACT_0 = 5                           # ACT initial conc, mg/L

k_hoh = 1e10                          # instantaneous
k_trap = 1e2                          # guess
k_recomb = 5e-3                       # kinetics of recombination, in holes cm-3 s-1
OH_ss = 9.0                           # saturated OH conc at surface, in nm-2

OH_stot = OH_ss*(1e18)/6.02e23        # conc of surface-bound OH species, mol/m2

# CALCULATIONS
E_photon = 6.63e-34*3e8/(wavelenght*1e-9)        # photon energy, in J/photon
photon_flux = 1e4*1e-3*irradiance/(E_photon)    # photon flux, in photons/s*m2
photon_flow = photon_flux*m_cat*1e-3*SSA         # photon flow, in photons/s
R_eh = (Q_eh*photon_flux*absorbance*W_cat)/6.02e23                   # mol holes/m2*s

# PARAMETERS
k_recomb = 5e-13                        # kinetics of recombination, in holes cm-3 s-1
OH_ss = 9.0                             # saturated OH conc at surface, in nm-2
MM_ACT = 151.163                        # molar mass, ACT, in g/mol

OH_stot = OH_ss*(1e18)/6.02e23          # conc of surface-bound OH species, mol/m2
OH_vtot = OH_stot*SSA*m_cat*1e-3/V_sol  # volumetric conc of surface-bound, mol/L

x0 = [ACT_0/MM_ACT*1e-3,                # initial concentrations
      0,
      0,
      0,
      0,
      0,
      1,
      0]

# define time
# t = np.linspace(0,7200,600)

def mechanism(x, t, paras):
    
    t = t*60
    
    # inputs
    wavelenght = 365                    # in nm
    irradiance = 5                      # in mW/cm2
    absorbance = 1e3                    # in m-1, of the catalyst 
    SSA = 45                            # specific surface area, in m2/g
    V_sol = 0.005                       # volume of solution, in L
    m_cat = 10                          # mass of catalyst, in mg
    #Q_eh = 1                            # quantum yield, mol of e-h pairs/eistein
    W_cat = m_cat/V_sol                 # catalyst load, in mg/dm3 = g/m3
    
    k_hoh = 1e10                          # instantaneous
    k_SOx = 1e10                          # instantaneous
    OH_ss = 9.0                           # saturated OH conc at surface, in nm-2
    OH_stot = OH_ss*(1e18)/6.02e23        # conc of surface-bound OH species, mol/m2
    O2_ss = 4.5 
    O2_stot = O2_ss*(1e18)/6.02e23        # conc of surface-bound OH species, mol/m2
    
    try:
        k_eh = paras['k_eh'].value
        k_phi = paras['k_fouling'].value
        k_ah = paras['k_hole_r'].value
        k_losses = paras['k_loss'].value
        QY_SOx = paras['QY_SOx'].value
        
    except KeyError:
        absorbance, k_phi, k_ah = paras    
    
    # CALCULATIONS
    E_photon = 6.63e-34*3e8/(wavelenght*1e-9)               # photon energy, in J/photon
    photon_flux = 1e4*1e-3*irradiance/(E_photon)            # photon flux, in photons/s*m2
    R_eh = k_eh*photon_flux/6.02e23                         # mol holes/m3*s
    
    # assign the parameters
    
    k_ACT_OH = 8.41E+09
    k_ACT_SOx = 3.35e+5
    k_ACT_1O2 = 3.5e+5
        
    OxSin = QY_SOx*R_eh
    #SupOx = 18e-6                                   # Korosi, 2019
    
    # assign each species to a vector element
    A = x[0]
    OH = x[1]
    SupOx = x[2]
    OxSin = x[3]
    h = x[4]
    el = x[5]
    phi = x[6]                                      # site availability (1 = full, 0 = empty)
    P = x[7]
                 
    # define the ODEs:
    
    dAdt = - k_ACT_OH*A*OH - k_ah*A*h*phi - k_ACT_SOx*SupOx*A - k_ACT_1O2*OxSin*A
    dPdt = k_ACT_OH*A*OH + k_ah*A*h*phi + k_ACT_SOx*SupOx*A + k_ACT_1O2*OxSin*A
    dhdt = R_eh - k_losses*h - k_hoh*h*OH_stot*phi - k_ah*A*h*phi
    deldt = R_eh - k_losses*el - k_SOx*el*O2_stot*phi
    dphidt = -k_phi*phi*P
    #dphidt = 0
    dOHdt = k_hoh*h*OH_stot*phi - k_ACT_OH*A*OH
    dSupOxdt = k_SOx*el*O2_stot*phi - QY_SOx*k_SOx*el*O2_stot*phi - k_ACT_SOx*SupOx*A
    dOxSindt = QY_SOx*k_SOx*el*O2_stot*phi - k_ACT_1O2*OxSin*A
    
    results = [dAdt,
               dOHdt,
               dSupOxdt,
               dOxSindt,
               dhdt,
               deldt,
               dphidt,
               dPdt
               ]
    
    return results


def g(t, x0, paras):
    
    x = odeint(mechanism, x0, t, args=(paras,))
    return x

def residual(paras, t, data):
    
    x0 = paras['A0'].value, paras['OH0'].value, paras['SupOx0'].value, paras['OxSin0'].value, paras['h0'].value, paras['el0'].value, paras['phi0'].value, paras['P0'].value
    model = g(t, x0, paras)
    
    ACT_model = model[:, 0]
    
    return (ACT_model - data).ravel()

##############################################################################
# Set parameters

params = Parameters()
params.add('A0', value=x0[0], vary=False)
params.add('OH0', value=x0[1], vary=False)
params.add('SupOx0', value=x0[2], vary=False)
params.add('OxSin0', value=x0[3], vary=False)
params.add('h0', value=x0[4], vary=False)
params.add('el0', value=x0[5], vary=False)
params.add('phi0', value=x0[6], vary=False)
params.add('P0', value=x0[7], vary=False)
params.add('k_eh',  value=1e-7, min=1e-12, max=1e-3)
params.add('k_fouling', value=3e2, min=1e-1, max=1e5)
params.add('k_hole_r', value=1e6, min=1e3, max=1e10)
params.add('k_loss', value=1e6, min=1e3, max=1e10)
params.add('QY_SOx', value=3e-2, min=3e-3, max=4e-1)

# Experimental data
x_exp = [0, 15, 30, 45, 60, 120]
y_exp = [1, 0.557, 0.421, 0.379, 0.354, 0.282]

x_sec = [time*60 for time in x_exp]
y_mol = [C*x0[0] for C in y_exp]

t_sim = np.linspace(0, 7200, 7200)
t_sim_min = t_sim/60

# Fit model
result = minimize(residual, params, args=(x_sec, y_mol), method='nelder')  # leastsq nelder

# check results of the fit
x = g(t_sim, x0, result.params)
ACT_fit = x[:,0]/x[0,0]
report_fit(result)

# Plot the results
plt.figure()
plt.scatter(x_sec, y_exp, marker='o', color='b', label='measured data', s=75)
plt.plot(t_sim, ACT_fit, '-', linewidth=2, color='red', label='fitted data')

plt.style.use('bmh')

figure, axis = plt.subplots(2,3)

axis[0,0].plot(t_sim_min,ACT_fit, label='ACT')
axis[0,0].plot(x_exp,y_exp, 'r*')

axis[0,1].plot(t_sim_min,x[:,1], label='OH')
axis[1,0].plot(t_sim_min,x[:,4], label='holes')
axis[1,1].plot(t_sim_min,x[:,6], label='occupancy')
axis[0,2].plot(t_sim_min,x[:,2], label='SOx')
axis[1,2].plot(t_sim_min,x[:,3], label='OxSin')

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
