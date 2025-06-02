import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import eigh

N = 20  # Number of masses / agents

ID = np.identity(N)
L_ID = np.diag(np.ones(N-1), k=-1)
U_ID = np.diag(np.ones(N-1), k=1)

P_LIM = 1.1    # limit product level
P_INF = 1.0    # steady state product level
O = 0.001       # Opposition coefficient
GAMMA_F = 0.0001/0.01  # diffusion coefficient
GAMMA_P = 1.0/0.01  # diffusion coefficient
D = 0.001        # destruction effects

C_MOD = -1.0   # Consumption Modulation coefficient

###########################################################################

def external_force(t, amp, freq):
    return amp * np.cos(2 * np.pi * freq * t)


def build_matricies(m,k,c):
    # build the mass matrix
    M = np.diag(m)
    # build the damping matrix
    c_l = np.append(c[1:], 0)
    c_u = c.copy()
    c_u[0] = 0
    C = ID*(c+c_l) - (L_ID * c_l) - (U_ID * c_u)
    # build the stiffness matrix
    k_l = np.append(k[1:], 0)
    k_u = k.copy()
    k_u[0] = 0
    K = ID*(k+k_l) - (L_ID * k_l) - (U_ID * k_u)

    return M, K, C


def multi_msd_system(t,y, m, k, c, amp, freq):

    '''
    Multi Mass-Spring-Damper System
        y = [x1, x2, x3, x4, x5, x6, v1, v2, v3, v4, v5, v6]
        m = mass vector                [m1, m2, m3, m4, m5, m6]
        k = spring constant vector     [k1, k2, k3, k4, k5, k6]
        c = damping constant vector    [c1, c2, c3, c4, c5, c6]
        Fext = external force vector   [F1, F2, F3, F4, F5, F6]
    '''

    x = np.array(y[:N])
    v = np.array(y[N:])

    Fext = external_force(t, amp, freq)

    dxdt = v
    dvdt = np.zeros(N)

    M, K,C = build_matricies(m,k,c)


    # calculate the acceleration
    dvdt = np.linalg.inv(M).dot(Fext - C.dot(v) - K.dot(x))

    return np.concatenate([dxdt, dvdt])


def tentacle_system(t,y, amp, freq):

    '''
        y = [x[N], v[N], F1[N], F2[N], F3[N], P1[N], P2[N], P3[N]]
    '''

    print(f'time: {t:.2f}')

    x = np.array(y[:N])
    v = np.array(y[N:2*N])
    F1 = np.array(y[2*N:3*N])
    F2 = np.array(y[3*N:4*N])
    F3 = np.array(y[4*N:5*N])
    P1 = np.array(y[5*N:6*N])
    P2 = np.array(y[6*N:7*N])
    P3 = np.array(y[7*N:8*N])

    k = F1**C_MOD
    c = F2**C_MOD
    m = F3**C_MOD

    y_smd = np.concatenate([x, v])
    y_dot_smd = multi_msd_system(t, y_smd, m, k, c, amp, freq)

    dxdt = y_dot_smd[:N]
    dvdt = y_dot_smd[N:]

    # cellular plasticity =======================================================

    rel_x = np.diff(x, prepend=0)  # Subtract previous displacement (prepend 0 for the first)
    rel_v = np.diff(v, prepend=0)  # Subtract previous velocity
    rel_dvdt = np.diff(dvdt, prepend=0)  # Subtract previous acceleration


    F = np.concatenate([F1, F2, F3])
    P = np.concatenate([P1, P2, P3])
    C = np.concatenate([rel_x, rel_v, rel_dvdt])* F ** C_MOD

    
    F = np.clip(F, 0.001, None)
    P = np.clip(P, 0.001, None)

    
    opp1 = O*(F2+F3)
    opp2 = O*(F1+F3)
    opp3 = O*(F1+F2)
    
    Opp =np.concatenate([opp1, opp2, opp3])

    Diff_F = np.roll(F, 1) + np.roll(F, -1) - 2 * F # middle conditions
    Diff_F[0] = F[1] - F[0] # first condition
    Diff_F[-1] = F[-2] - F[-1] # last condition

    Diff_P = np.roll(P, 1) + np.roll(P, -1) - 2 * P # middle conditions
    Diff_P[0] = P[1] - P[0] # first condition
    Diff_P[-1] = P[-2] - P[-1] # last condition


    dFdt = (1 - P / P_INF - Opp) * F - D*np.abs(C*P)  + GAMMA_F * Diff_F
    dPdt = (1 - P / P_LIM) * F - np.abs(P * C) + GAMMA_P * Diff_P

    # Apply the condition to dFdt where F == 0.001 and dFdt < 0.0
    mask = (F <= 0.001) & (dFdt < 0.0) 
    dFdt[mask] = 0.0

    mask = (P <= 0.001) & (dPdt < 0.0) 
    dPdt[mask] = 0.0

    return np.concatenate([dxdt, dvdt, dFdt, dPdt])
    


# external forces
freq = np.zeros(N)
amp = np.zeros(N)
amp[-1] = 10.0
freq[-1] = 1.0

# initial conditions
x0 = np.zeros(N)
v0 = np.zeros(N)
f0 = np.ones(3*N)+ np.random.rand(3*N)*0.01
p0 = np.ones(3*N)+ np.random.rand(3*N)*0.01
y0 = np.concatenate([x0, v0, f0, p0])

# time simulation
tend = 120.0
t_span = (0, tend)
t_eval = np.linspace(*t_span, int(100*tend))

solution = solve_ivp(tentacle_system, t_span, y0, t_eval=t_eval, args=(amp, freq))

# Extract Results====================================================================
x = solution.y[:N]
v = solution.y[N:2*N]
k = solution.y[2*N:3*N]**C_MOD
c = solution.y[3*N:4*N]**C_MOD
m = solution.y[4*N:5*N]**C_MOD

time = solution.t

# calculate energy
PE = 0.5 * np.sum(k * x**2, axis=0)
KE = 0.5 * np.sum(m * v**2, axis=0)
E = PE + KE


# Plotting==========================================================================

# plt.figure()
# for i in range(N):
#     plt.plot(time, x[i], label=f'x_{i+1}')  
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Displacement')
# plt.title('Displacement of Masses')
# plt.show(block=False)


# Create a 2-row subplot: top row for time-series plots, bottom row for final values
fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})

# Top row: Time-series plots
for i in range(N):
    axes[0, 0].plot(time, k[i], '-', label=f'k_{i+1}')
axes[0, 0].set_title('Stiffness (k) Over Time')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Adapted Parameters')
axes[0, 0].legend()

for i in range(N):
    axes[0, 1].plot(time, c[i], '--', label=f'c_{i+1}')
axes[0, 1].set_title('Damping (c) Over Time')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].legend()

for i in range(N):
    axes[0, 2].plot(time, m[i], '-.', label=f'm_{i+1}')
axes[0, 2].set_title('Mass (m) Over Time')
axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].legend()

# Bottom row: Mean of the last half of the data
final_k = [np.mean(k[i][len(k[i]) // 2:]) for i in range(N)]  # Mean of last half of k
final_c = [np.mean(c[i][len(c[i]) // 2:]) for i in range(N)]  # Mean of last half of c
final_m = [np.mean(m[i][len(m[i]) // 2:]) for i in range(N)]  # Mean of last half of m

axes[1, 0].bar(range(1, N+1), final_k, color='blue')
axes[1, 0].set_title('Final Stiffness (k)')
axes[1, 0].set_xlabel('Index')
axes[1, 0].set_ylabel('Final Value')

axes[1, 1].bar(range(1, N+1), final_c, color='orange')
axes[1, 1].set_title('Final Damping (c)')
axes[1, 1].set_xlabel('Index')
axes[1, 1].set_ylabel('Final Value')

axes[1, 2].bar(range(1, N+1), final_m, color='green')
axes[1, 2].set_title('Final Mass (m)')
axes[1, 2].set_xlabel('Index')
axes[1, 2].set_ylabel('Final Value')

# Adjust layout for better spacing
plt.tight_layout()
plt.show(block= True)

# plt.figure()
# plt.plot(time, PE, label='PE')
# plt.plot(time, KE, label='KE')
# plt.plot(time, E, label='E')
# plt.xlabel('Time (s)')
# plt.ylabel('Energy')
# plt.legend()
# plt.title('Energy Analysis')
# plt.show(block=False)

# natural frequency and damping
K_fin, M_fin, C_fin = build_matricies(final_m, final_k, final_c)
# Solve eigenvalue problem for natural frequencies and mode shapes
eigenvalues, mode_shapes = eigh(K_fin, M_fin)
natural_frequencies = np.sqrt(eigenvalues)

# Transform damping matrix to modal coordinates
C_m = mode_shapes.T @ C_fin @ mode_shapes

# Compute damping ratios
damping_ratios = np.diag(C_m) / (2 * natural_frequencies)

# Output results
print("Natural Frequencies (rad/s):", natural_frequencies)
print("Damping Ratios:", damping_ratios)
print("Mode Shapes (columns):\n", mode_shapes)


# # Create a 2-row subplot: top row for time-series plots, bottom row for final values
# fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
# # Top row: Time-series plots
# for i in range(N):
#     axes[0, 0].plot(time, omega_n[i], '-', label=f'ω_{i+1}')
# axes[0, 0].set_title('Natural Frequency (ω_n) Over Time')
# axes[0, 0].set_xlabel('Time (s)')
# axes[0, 0].set_ylabel('Adapted Parameters')
# axes[0, 0].legend()

# for i in range(N):
#     axes[0, 1].plot(time, zeta[i], '--', label=f'ζ_{i+1}')
# axes[0, 1].set_title('Damping Ratio (ζ) Over Time')
# axes[0, 1].set_xlabel('Time (s)')
# axes[0, 1].set_ylabel('Adapted Parameters')
# axes[0, 1].legend()

# # Bottom row: Mean of the last half of the data
# final_omega_n = [np.mean(omega_n[i][len(omega_n[i]) // 2:]) for i in range(N)]  # Mean of last half of omega_n
# final_zeta = [np.mean(zeta[i][len(zeta[i]) // 2:]) for i in range(N)]  # Mean of last half of zeta

# axes[1, 0].bar(range(1, N+1), final_omega_n, color='blue')
# axes[1, 0].set_title('Final Natural Frequency (ω_n)')
# axes[1, 0].set_xlabel('Index')
# axes[1, 0].set_ylabel('Final Value')

# axes[1, 1].bar(range(1, N+1), final_zeta, color='orange')
# axes[1, 1].set_title('Final Damping Ratio (ζ)')
# axes[1, 1].set_xlabel('Index')
# axes[1, 1].set_ylabel('Final Value')

# # Adjust layout for better spacing
# plt.tight_layout()
# plt.show(block=False)



############ perform fixed msd system simulation ###############################

# m_mean = np.median(final_m)*np.ones(N)
# k_mean = np.median(final_k)*np.ones(N)
# c_mean = np.median(final_c)*np.ones(N)



# print(f"Mean Mass: {m_mean[0]}")
# print(f"Mean Stiffness: {k_mean[0]}")
# print(f"Mean Damping: {c_mean[0]}")

# fixed_solution = solve_ivp(multi_msd_system, t_span, np.concatenate([x0, v0]), t_eval=t_eval, args=(m_mean, k_mean, c_mean, amp, freq))

# # Extract Results====================================================================
# x_fixed = fixed_solution.y[:N]
# v_fixed = fixed_solution.y[N:2*N]
# time_fixed = fixed_solution.t

# # calculate energy
# PE_fixed = 0.5 * np.sum(k_mean[0] * x_fixed**2, axis=0)
# KE_fixed = 0.5 * np.sum(m_mean[0] * v_fixed**2, axis=0)
# E_fixed = PE_fixed + KE_fixed

# # plot results =====================================================================


# plt.figure()
# for i in range(N):
#     plt.plot(time_fixed, x_fixed[i], '--', label=f'x_{i+1} fixed')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Displacement')
# plt.title('Fixed Displacement of Masses')
# plt.show(block=False)


# plt.figure()
# plt.plot(time, PE_fixed, label='PE_fixed')
# plt.plot(time, KE_fixed, label='KE_fixed')
# plt.plot(time, E_fixed, label='E_fixed')
# plt.xlabel('Time (s)')
# plt.ylabel('Energy')
# plt.legend()
# plt.title('Fixed Energy Analysis')
# plt.show(block=True)








