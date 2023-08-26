# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:38:25 2022
Kinetics of ACT degradation by TiO2 photocatalysis
@author: Bruno
"""
from scipy.integrate import odeint
import csv
import numdifftools

##############################################################################
def read_input (filename):
    
    with open (filename) as file:
        
        input_file = csv.reader (file, delimiter = ';')
        rows = [r for r in input_file]
        
        time = []
        C_ACT = []
        SSA = []
        G_surf = []
        m_cat = []
        irrad = []
        absorbance = []
        wavelength = []
        material = []
        
        for i in range(1,len(rows)-1):
            time.append(float(rows[i][0]))
            C_ACT.append (float (rows[i][1]))
       
        SSA.append    (float (rows[0][5]))
        G_surf.append (float (rows[1][5]))
        m_cat.append  (float (rows[2][5]))
        irrad.append  (float (rows[3][5]))
        absorbance.append(float (rows[4][5]))
        wavelength.append(float (rows[5][5]))
        material.append(rows[6][5])
        
    output = [time,
              C_ACT,
              SSA,
              G_surf,
              m_cat,
              irrad,
              absorbance,
              wavelength,
              material
              ]
    
    return output

##############################################################################

def mechanism(x, t, paras):
    
    t = t*60
    
    # inputs
    wavelength = paras['wavelength'].value              # in nm
    irradiance = paras['irradiance'].value              # in mW/cm2
    absorbance = paras['absorbance'].value              # in m-1, of the catalyst 
    SSA = paras['SSA'].value                            # specific surface area, in m2/g
    Special_sites = paras['Special_sites'].value
    
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
        k_wash = paras['k_wash'].value
        k_as = paras['k_as'].value
        QY_SOx = paras['QY_SOx'].value
        
    except KeyError:
        absorbance, k_phi, k_ah = paras    
    
    # CALCULATIONS
    E_photon = 6.63e-34*3e8/(wavelength*1e-9)               # photon energy, in J/photon
    photon_flux = 1e4*1e-3*irradiance/(E_photon)            # photon flux, in photons/s*m2
    R_eh = k_eh*photon_flux/6.02e23                         # mol holes/m2*s
    
    if Special_sites == -1:
        C_SS = 0
    else:
        C_SS = 1e-6*paras['G_SS'].value                         # mol additive/m2
    
    # assign the parameters
    
    k = [8.41E+09,                 # OH second-order constants
                6.02E+09,
                7.13E+09,
                1.17E+10,
                6.63E+08,
                3.10E+08,
                5.87E+09,
                4.83E+07,
                5.13E+08,
                1.72E+08,
                4.47E+08,
                4.08E+08,
                1.40E+06,
                1.00E+06,
                ]

    S = [0.73,                     # Selectivities
              0.18,
              0.09,
              0.78,
              0.21]
 
    k_ACT_1O2 = 2.50e+7
           
    # assign each species to a vector element
    A = x[0]
    OH = x[1]
    R2 = x[2]
    R3 = x[3]
    R4 = x[4]
    R5 = x[5]
    R6 = x[6]
    R7 = x[7]
    R8 = x[8]
    R9 = x[9]
    R10 = x[10]
    R11 = x[11]
    R12 = x[12]
    R13 = x[13]
    R14 = x[14]
    h = x[15]
    phi = x[16]             # site availability (1 = full, 0 = empty)
    SS_ad = x[17]
    el = x[18]
    OxSin = x[19]
    
    
    # dummy trial
    # OH = 1e-13
    
    # define the ODEs:
    
    dAdt = - k[0]*A*OH - k_ah*A*h*phi - k_ACT_1O2*A*OxSin - k_as*A*C_SS*SS_ad
    #dAdt = - k[0]*A*OH - k_ah*A*h*phi - k_ACT_1O2*OxSin
    #dAdt = - k[0]*A*OH - k_ah*A*h*phi
    dR2dt = S[0]*(k[0]*A*OH + k_ah*A*h*phi + k_ACT_1O2*A*OxSin + k_as*A*C_SS*SS_ad) - k[1]*R2*OH
    dR3dt = S[1]*(k[0]*A*OH + k_ah*A*h*phi + k_ACT_1O2*A*OxSin + k_as*A*C_SS*SS_ad) - k[2]*R3*OH
    dR4dt = S[2]*(k[0]*A*OH + k_ah*A*h*phi + k_ACT_1O2*A*OxSin + k_as*A*C_SS*SS_ad) - k[3]*R4*OH
    dR5dt = k[5]*R6*OH - k[4]*R5*OH
    dR6dt = S[3]*k[3]*R4*OH - k[5]*R6*OH
    dR7dt = S[4]*k[3]*R4*OH - k[6]*R7*OH
    dR8dt = 2*k[1]*R2*OH + 2*k[3]*R4*OH + k[5]*R6*OH - k[7]*R8*OH
    dR9dt = k[3]*R4*OH + 2*k[6]*R7*OH - k[8]*R9*OH
    dR10dt = k[5]*R6*OH + k[7]*R8*OH - k[9]*R10*OH
    dR11dt = k[11]*R12*OH - k[10]*R11*OH
    dR12dt = k[1]*R2*OH + k[2]*R3*OH + S[3]*k[0]*A*OH - k[11]*R12*OH
    dR13dt = k[7]*R8*OH + k[8]*R9*OH - k[12]*R13*OH
    dR14dt = 2*k[9]*R10*OH + 2*k[12]*R13*OH
    #dhdt = R_eh - k_trap*h - k_recomb*h - k_hoh*h*OH_stot*phi - k_ah*A*h
    dhdt = R_eh - k_losses*h - k_hoh*h*OH_stot*phi - k_ah*A*h*phi
    dphidt = -k_phi*phi*(R2+R3+R4+R5+R6+R7+R8+R9+R10+R11+R12+R13)
    #dphidt = 0
    dOxSindt = QY_SOx*k_SOx*el*O2_stot - k_ACT_1O2*OxSin*A
    dSS_addt = -k_wash*C_SS*SS_ad
    deldt = R_eh - k_losses*el - k_SOx*el*O2_stot    
    
    dOHdt = k_hoh*h*OH_stot*phi - (k[0]*A + k[1]*R2 + k[2]*R3 + k[3]*R4 + k[4]*R5 + k[5]*R6 +
                k[6]*R7 + k[7]*R8 + k[8]*R9 + k[9]*R10 + k[10]*R11 + k[11]*R12 +
                k[12]*R13)*OH
    
    results = [dAdt,
               dOHdt,
               dR2dt,
               dR3dt,
               dR4dt,
               dR5dt,
               dR6dt,
               dR7dt,
               dR8dt,
               dR9dt,
               dR10dt,
               dR11dt,
               dR12dt,
               dR13dt,
               dR14dt,
               dhdt,
               dphidt,
               dSS_addt,
               deldt,
               dOxSindt
               ]
    
    return results

##############################################################################

def g(t, x0, paras):
    
    x = odeint(mechanism, x0, t, args=(paras,))
    return x

##############################################################################

def residual(paras, t, data):
    
    x0 = paras['A0'].value, paras['OH0'].value, paras['R20'].value, paras['R30'].value, paras['R40'].value, paras['R50'].value,    paras['R60'].value, paras['R70'].value, paras['R80'].value, paras['R90'].value, paras['R100'].value,    paras['R110'].value, paras['R120'].value, paras['R130'].value, paras['R140'].value, paras['h0'].value, paras['phi0'].value, paras['SS_ad0'].value, paras['el0'].value, paras['OxSin0'].value
    
    model = g(t, x0, paras)
    
    ACT_model = model[:, 0]
    
    return (ACT_model - data).ravel()

##############################################################################

def write_output (filename, results):
    
    units = ['mol h/einstein irrad', 
             's/mol', 
             'm2/(mol*s)', 
             '1/s', 
             'm2/(mol*s)', 
             'm2/(mol*s)',
             'mol 1O2/mol SOx',
             'mol/m2',
             'mol/L',
             'mol/L']
    
    with open (filename, 'w', newline='') as file:
        
        output_file = csv.writer (file, delimiter = ';')
        output_file.writerow(results.keys())
        output_file.writerow(results.values())
        output_file.writerow(units)
        
    return output_file
            