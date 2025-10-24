##################################################################################################################
# Author: Lorenzo Martinelli
# This script contains all the work done in the notebook Ex2_notebook.ipynb 
# (plotting the SAR and EAR curves, fitting them with a power law).
# The results shown in the report are obtained through the usage of the notebook, rather than this script.
# Any inconsistency between the two versions is only the fault of the author.
# Additionally, if running this script any issue is encountered, it should usually be fixed by checking how the 
# same specific issue was handled in the notebook version.  
##################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(13)

##################### FUNCTIONS #####################

def SAR(a: float, nu: float, S: int):
    """
    This function generates a profile for the SAR, given a log-series 
    probability for the species distribution.

    Args:
        a (float): Size of the subdivision of area taken into account.
        nu (float): Spontaneous birth rate.
        S (int): Number of species
    
    Returns:
        sar (float): Species Area Relationship
    """
    if np.any(a) > 1 or np.any(a) < 0:
        raise ValueError(f"The size of the subdivision must be greater than zero or lower than one. Used value: {a}")
    if nu > 1 or nu < 0:
        raise ValueError(f"The spontaneous birth rate must be greater than zero or lower than one. Used value: {nu}")
    if not isinstance(S, int):
        raise TypeError(f"The number of species must be an integer. Type of S: {type(S)}")
    if S < 0:
        raise ValueError(f"There must be at least 1 species in the ecosystem.")

    c = np.abs(1 / np.log(nu))
    sar = S * (1 + c * np.log(nu * (1 - a) + a))
    return sar

def EAR(a: float, nu: float, S: int):
    """
    This function generates a profile for the EAR, given a log-series 
    probability for the species distribution.

    Args:
        a (float): Size of the subdivision of area taken into account.
        nu (float): Spontaneous birth rate.
        S (int): Number of species
    
    Returns:
        ear (float): Endemic Area Relationship
    """
    if np.any(a) > 1 or np.any(a) < 0:
        raise ValueError(f"The size of the subdivision must be greater than zero or lower than one. Used value: {a}")
    if nu > 1 or nu < 0:
        raise ValueError(f"The spontaneous birth rate must be greater than zero or lower than one. Used value: {nu}")
    if not isinstance(S, int):
        raise TypeError(f"The number of species must be an integer. Type of S: {type(S)}")
    if S < 0:
        raise ValueError(f"There must be at least 1 species in the ecosystem.")
    
    ear = -S * np.log(1 - a * (1 - nu)) / np.abs(np.log(nu))
    return ear

##################### PARAMETERS #####################
S = 500                                           # Number of species
nu = 0.01                                         # Intrinsic birth rate
a_domain = np.linspace(0, 1, num = 1000)          # Subdivisions of the area

sar = SAR(a_domain, nu, S)
ear = EAR(a_domain, nu, S)

##################### PLOTS #####################
plt.figure(figsize = (7, 7))
plt.plot(sar, color = 'black', label = 'SAR', linewidth = 2)
plt.plot(ear, color = 'red', label = 'EAR', linewidth = 2)
plt.vlines(x = 1000 * 2/3, ymin = 0, ymax = 520, color = 'green', linewidth = 2, linestyle = '--', label = 'a = 2/3A')
plt.legend(loc = 'best', fontsize = 'large')
plt.xlabel('Fraction of Area [A.U.]', fontsize = 'x-large')
plt.xticks(ticks = [0, 200, 400, 600, 800, 1000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 'large')
plt.ylim(bottom = 0, top = 520)
plt.ylabel('Number of Species', fontsize = 'x-large')
plt.yticks(fontsize = 'large')
plt.title('Comparison Between SAR and EAR', fontsize = 'xx-large')
plt.grid(True)
# plt.savefig('sar_vs_ear.png', dpi = 300, bbox_inches = 'tight')
plt.close()

sar_2_3 = SAR(2/3, nu, S)
ear_2_3 = EAR(2/3, nu, S)

print(f"Number of species in a = 2/3 A: {sar_2_3:.3f} out of {S}")
print(f"Number of endemic species in a = 2/3 A: {ear_2_3:.3f} out of {S}")

plt.figure(figsize = (7, 7))
plt.plot(sar, color = 'black', label = 'SAR', linewidth = 2)
plt.xlabel('Fraction of Area [A.U.]', fontsize = 'x-large')
plt.xscale('log')
plt.yscale('log')
plt.xticks(ticks = [1, 10, 100, 1000], labels = [0.001, 0.01, 0.1, 0.1], fontsize = 'large')
plt.ylabel('Number of Species', fontsize = 'x-large')
plt.yticks(ticks = [10, 100, 500], labels = [10, 100, 500], fontsize = 'large')
plt.title('SAR in log-log Scale', fontsize = 'xx-large')
plt.grid(True)
# plt.savefight('log_sar.png', dpi = 300, bbox_inches = 'tight')
plt.close()


####################################### POWER LAW FIT #######################################

##################### FUNCTIONS #####################
def SAR_pl(a: float, k: float, z: float):
    """
    This function produces the power-law version of the empirical
    SAR distribution:
    
    SAR_pl = k * a ** z
    """
    if np.any(a) > 1 or np.any(a) < 0:
        raise ValueError(f"The size of the subdivision must be greater than zero or lower than one. Used value: {a}")
    if k < 0:
        raise ValueError(f"The multiplicative factor must be greater than zero. Used value: {k}")
    if z < 0 or z > 1:
        raise ValueError(f"The exponent must be greater than zero or lower than one. Used value: {z}")
    sar = k * np.power(a, z)
    return sar

##################### PARAMETERS #####################
k = S
z_domain = [0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
sar_pl_all = np.zeros(shape = (len(z_domain), 1000))

for i, z in enumerate(z_domain):
    sar_pl_all[i, :] = SAR_pl(a_domain, k, z)

##################### PLOTS #####################
plt.figure(figsize = (7, 7))
plt.plot(sar, color = 'black', label = 'SAR', linewidth = 2)
for i in range(len(sar_pl_all)):
    plt.plot(sar_pl_all[i], label = f'z = {z_domain[i]}', linewidth = 2, linestyle = '--')
plt.legend(loc = 'best', fontsize = 'large')
plt.xlabel('Fraction of Area [A.U.]', fontsize = 'x-large')
plt.xticks(ticks = [0, 200, 400, 600, 800, 1000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 'large')
plt.ylabel('Number of Species', fontsize = 'x-large')
plt.yticks(fontsize = 'large')
plt.title('Power Law SAR (Different Exponents)', fontsize = 'xx-large')
plt.grid(True)
plt.show()
# plt.savefight('power_law_sar.png', dpi = 300, bbox_inches = 'tight')
plt.close()

popt, _ = curve_fit(SAR_pl, a_domain, sar, p0 = [500, 0.23])
k_best, z_best = popt[0], popt[1]
print(f'Best parameters: k = {k_best:.3f} and z = {z_best:.3f}')

sar_pl_guess = SAR_pl(a_domain, S, 0.23)
sar_pl_best  = SAR_pl(a_domain, k_best, z_best)
sar          = SAR(a_domain, nu, S)

plt.figure(figsize = (7, 7))
plt.plot(sar, color = 'black', label = 'Real SAR', linewidth = 2)
plt.plot(sar_pl_guess, color = 'deeppink', label = 'First Guess', linewidth = 2, linestyle = '--')
plt.plot(sar_pl_best, color = 'dodgerblue', label = 'Scipy Fit', linewidth = 2, linestyle = '--')
plt.legend(loc = 'best', fontsize = 'large')
plt.xlabel('Fraction of Area [A.U.]', fontsize = 'x-large')
plt.xticks(ticks = [0, 200, 400, 600, 800, 1000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 'large')
plt.ylabel('Number of Species', fontsize = 'x-large')
plt.yticks(fontsize = 'large')
plt.title('Comparison Between Fits', fontsize = 'xx-large')
plt.grid(True)
# plt.savefight('compare_fits.png', dpi = 300, bbox_inches = 'tight')
plt.close()

nu_list = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
sar_nus = np.zeros(shape = (len(nu_list), 1000))
for i, nu in enumerate(nu_list):
    sar_nus[i, :] = SAR(a_domain, nu, S)

##################### PLOTS #####################
plt.figure(figsize = (7, 7))
plt.plot(sar_pl_guess, color = 'deeppink', label = 'First Guess', linewidth = 2, linestyle = '--')
for i in range(len(sar_nus)):
    plt.plot(sar_nus[i], label = f'nu = {nu_list[i]}', linewidth = 2)
plt.legend(loc = 'best', fontsize = 'large')
plt.xlabel('Fraction of Area [A.U.]', fontsize = 'x-large')
plt.xticks(ticks = [0, 200, 400, 600, 800, 1000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 'large')
plt.ylabel('Number of Species', fontsize = 'x-large')
plt.yticks(fontsize = 'large')
plt.title('Power Law SAR (Different Birth Rates)', fontsize = 'xx-large')
plt.grid(True)
# plt.savefight('compare_birth_rates.png', dpi = 300, bbox_inches = 'tight')
plt.close()