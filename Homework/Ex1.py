##################################################################################################################
# Author: Lorenzo Martinelli
# This script contains all the work done in the notebook Ex1_notebook.ipynb 
# (simulations of a consumer-resource model with linear or Monod intake).
# The results shown in the report are obtained through the usage of the notebook, rather than this script.
# Any inconsistency between the two versions is only the fault of the author.
# Additionally, if running this script any issue is encountered, it should usually be fixed by checking how the 
# same specific issue was handled in the notebook version.  
##################################################################################################################


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13)


####################################### LINEAR INTAKE #######################################

##################### PARAMETERS #####################
gamma   = 0.8                       # Yield coefficient
d       = 0.4                       # Consumer death rate
q       = 0.2                       # Uptake constant
C       = 5                         # Resource supply

T       = 100000                    # Total time
dt      = 0.0001                    # Timestep

N0      = 2                         # Initial consumers
R0      = 5                         # Initial resources

N = np.zeros(shape = T + 1)
R = np.zeros(shape = T + 1)

N[0], R[0] = N0, R0


##################### SIMULATION LOOP #####################
for t in range(T):
    dN = ( gamma * q * R[t] - d ) * N[t] * dt
    dR = ( C - q * R[t] * N[t] ) * dt

    N[t+1] = N[t] + dN
    R[t+1] = R[t] + dR

# Plot the evolution
plt.figure(figsize = (12, 6))
plt.plot(N, label = 'Consumers', color = 'red', linewidth = 2)
plt.plot(R, label = 'Resource', color = 'black', linewidth = 2)
plt.hlines(y = 10, xmin = 0, xmax = T, color = 'blue', linewidth = 2, linestyle = '--', label = 'N*')
plt.hlines(y = 2.5, xmin = 0, xmax = T, color = 'orange', linewidth = 2, linestyle = '--', label = 'R*')
plt.legend(loc = 'best', fontsize = 17)
plt.xlabel('Time [A.U.]', fontsize = 20)
plt.ylabel('Population', fontsize = 20)
plt.title('Linear Intake', fontsize = 25)
plt.xticks(ticks = [0, 20000, 40000, 60000, 80000, 100000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(bottom = 0, top = 12)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.grid(True)
# plt.savefig('linear_intake.png', dpi = 300, bbox_inches = 'tight')
plt.close()

####################################### MONOD INTAKE #######################################
##################### PARAMETERS #####################
q_M     = 3                        # Max uptake rate
K       = 8                         # Half-saturation constant

T       = 100000                    # Total time
dt      = 0.0001                    # Timestep

N0      = 2                         # Initial consumers
R0      = 5                         # Initial resources

N_monod = np.zeros(shape = T + 1)
R_monod = np.zeros(shape = T + 1)

N_monod[0], R_monod[0] = N0, R0


##################### SIMULATION LOOP #####################
for t in range(T):
    dN = ( gamma * ( q_M * R_monod[t] ) / ( K + R_monod[t] ) - d ) * N_monod[t] * dt
    dR = ( C - ( q_M * R_monod[t] ) / ( K + R_monod[t] ) * N_monod[t] ) * dt

    N_monod[t+1] = N_monod[t] + dN
    R_monod[t+1] = R_monod[t] + dR

# Plot the evolution
plt.figure(figsize = (12, 6))
plt.plot(N_monod, label = 'Consumers', color = 'red', linewidth = 2)
plt.plot(R_monod, label = 'Resource', color = 'black', linewidth = 2)
plt.hlines(y = 10, xmin = 0, xmax = T, color = 'blue', linewidth = 2, linestyle = '--', label = 'N*')
plt.hlines(y = 1.6, xmin = 0, xmax = T, color = 'orange', linewidth = 2, linestyle = '--', label = 'R*')
plt.legend(loc = 'best', fontsize = 17)
plt.xlabel('Time [A.U.]', fontsize = 20)
plt.ylabel('Population', fontsize = 20)
plt.title('Monod Intake', fontsize = 25)
plt.xticks(ticks = [0, 20000, 40000, 60000, 80000, 100000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(bottom = 0, top = 12)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.grid(True)
# plt.savefig('monod_intake.png', dpi = 300, bbox_inches = 'tight')
plt.close()


# Comparison plot
plt.figure(figsize = (12, 6))
plt.plot(N, label = 'N(t) (Linear)', color = 'red', linewidth = 2)
plt.plot(R, label = 'R(t) (Linear)', color = 'deeppink', linewidth = 2)
plt.plot(N_monod, label = 'N(t) (Monod)', color = 'blue', linewidth = 2)
plt.plot(R_monod, label = 'R(t) (Monod)', color = 'orange', linewidth = 2)

plt.hlines(y = 10, xmin = 0, xmax = T, label = 'N* (both)', color = 'black', linestyle = '--', linewidth = 2)
plt.hlines(y = 2.5, xmin = 0, xmax = T, label = 'R* (Linear)', color = 'green', linestyle = '--', linewidth = 2)
plt.hlines(y = 1.6, xmin = 0, xmax = T, label = 'R* (Monod)', color = 'purple', linestyle = '--', linewidth = 2)

plt.legend(loc = 'best', fontsize = 17)
plt.xlabel('Time [A.U.]', fontsize = 20)
plt.ylabel('Population', fontsize = 20)
plt.title('Comparison Between Linear vs. Monod Intake', fontsize = 25)
plt.xticks(ticks = [0, 20000, 40000, 60000, 80000, 100000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(bottom = 0, top = 12)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.grid(True)
# plt.savefig('comparison_linear_vs_monod.png', dpi = 300, bbox_inches = 'tight')
plt.close()


# Phase portrait plot
plt.figure(figsize = (12, 6))
plt.plot(N, R, label = 'Linear Intake', color = 'red', linewidth = 2)
plt.plot(N_monod, R_monod, label = 'Monod Intake', color = 'blue', linewidth = 2)
plt.legend(loc = 'best', fontsize = 17)
plt.xlabel('N(t)', fontsize = 20)
plt.ylabel('R(t)', fontsize = 20)
plt.title('Phase Portrait', fontsize = 25)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.grid(True)
# plt.savefig('phase_portrait.png', dpi = 300, bbox_inches = 'tight')
plt.close()


# Linear intake, N0 > N* and N0 < N* comparison
M0      = 15                        # Initial consumers, larger than N*
Q0      = 5                         # Initial resources, identical to the previous example

M = np.zeros(shape = T + 1)
Q = np.zeros(shape = T + 1)        

M[0], Q[0] = M0, Q0


##################### SIMULATION LOOP #####################
for t in range(T):
    dM = ( gamma * q * Q[t] - d ) * M[t] * dt
    dQ = ( C - q * Q[t] * M[t] ) * dt

    M[t+1] = M[t] + dM
    Q[t+1] = Q[t] + dQ

plt.figure(figsize = (12, 6))
plt.plot(N, label = 'N(0) < N*', color = 'red', linewidth = 2)
plt.plot(M, label = 'N(0) > N*', color = 'black', linewidth = 2)
plt.plot(R, label = 'R(t) (N(0) < N*)', color = 'red', linewidth = 2, linestyle = '--')
plt.plot(Q, label = 'R(t) (N(0) > N*)', color = 'black', linewidth = 2, linestyle = '--')
plt.hlines(y = 10, xmin = 0, xmax = T, color = 'blue', linewidth = 2, linestyle = '--', label = 'N*')
plt.legend(loc = 'best', fontsize = 17)
plt.xlabel('Time [A.U.]', fontsize = 20)
plt.ylabel('Population', fontsize = 20)
plt.title('Linear Intake, Different Starting Conditions', fontsize = 25)
plt.xticks(ticks = [0, 20000, 40000, 60000, 80000, 100000], labels = [0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(bottom = 0, top = 18)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.grid(True)
# plt.savefig('different_starting_conditions.png', dpi = 300, bbox_inches = 'tight')
plt.close()