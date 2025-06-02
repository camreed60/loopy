import numpy as np
import time
import csv
import matplotlib
print(matplotlib.get_backend())
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from Loopy import Loopy
import copy


# robot parameteters
N = 36
mids = range(N)
zeros = [0] * len(mids)

#robot = Loopy(device_path1='/dev/ttyUSB0', device_path2='/dev/ttyUSB1', baudrate=4000000)

'''
P_INF = 1.0
P_LIM = 1.1 
O = 0.01
D = 0.1
DF = 0.001 
DP = 10 

'''

'''
P_INF = 1.0
P_LIM = 1.1 
O = 0.01
D = 0.1
DF = 0.01 
DP = 100 
C_SCALE = np.array([1e+1, 1e+3, 1e-0]) 

'''

psi = 100000


# cellular plasticity parameters
P_INF = 1.0
P_LIM = 1.5 
O = 0.01
D = 0.1
psi = 100000
DF = 0.001 
DP = DF*psi

M = 3

C_SCALE = np.array([1e+1, 1e+3, 1e-0]) 

# fitzhugh nagumo parameters
A = 0.0
B = 1.0
TAU= 0.04
R = 0.0001
I = 0.1
DA = 1.0
DI = 50

# Diffusive only morphogen parameters
DD = 100

# Wave speed parameters
LAMBDA = 0.0

# simulation parameters
T_END = 120


# parameter check, print out then have to hit enter to continue
# G  = 1.0, K = 1/Pinf
# R = 1.0, I = 1/PLIM
# assume the consumption rate is 1.0
C_star = 1.0

P_star = (1.0 - D*1.0) / (1/P_INF - D/P_LIM)
Z = (1.0 - D*1.0) / (1.0/P_INF - 1.0/P_LIM)
F_star = Z*C_star

J11 = 1.0 - P_star/P_INF
J12 = -F_star/P_INF - D*C_star
J21 = 1.0 - P_star/P_LIM
J22 = -F_star/P_LIM - C_star

# print to two decimal points
print(f'P_star: {P_star:.2f} F_star: {F_star:.2f} Z: {Z:.2f}')
print(f'stability check {(J11 + J22):.2f} < 0')
print(f'Turing pattern check {((DP/DF)*J11 + J22):.2f} > 0')

#hit enter to continue
input()


def fitzhugh_nagumo(act, inh):
    act_dot = (act - act**3 / 3 - inh + R * I)
    inh_dot = ((act + A - B * inh) / TAU)
    return act_dot, inh_dot

def cellular_plasticity(fact, prod, con):

    fact = np.clip(fact, 0.001, None)
    prod = np.clip(prod, 0.001, None)

    opp = O * (np.sum(fact, axis=1, keepdims=True) - fact)
    fact_dot = (1 - prod / P_INF - opp) * fact - D*np.abs(con * prod)
    prod_dot = (1 - prod / P_LIM) * fact - np.abs(con * prod)

    # Apply the condition to dFdt where F == 0.001 and dFdt < 0.0
    mask = (fact <= 0.001) & (fact_dot < 0.0)
    fact_dot[mask] = 0.0
    mask = (prod <= 0.001) & (prod_dot < 0.0)
    prod_dot[mask] = 0.0

    return fact_dot, prod_dot

def diffusion(Morph):
    Morphxx = np.roll(Morph, 1, axis=0) + np.roll(Morph, -1, axis=0) - 2 * Morph  # Middle conditions
    # Morphxx[0] = Morph[1] + Morph[-1] - 2*Morph[0]  # First condition
    # Morphxx[-1] = Morph[-2] + Morph[0] - 2*Morph[-1]  # Last condition
    return Morphxx

def wave(Morph):
    Morph_dot = -LAMBDA * (np.roll(Morph, -1) - np.roll(Morph, 1)) / 2
    return Morph_dot


def loopy_system(t, y):
    '''
        y = [F[N*M], P[N*M], A[N], I[N], D[N]]
    '''
    global PREV_ACTS

    # unwrap state vector
    Facts = np.array(y[:N*M]).reshape(N, M)  # [num cells, num factories]
    Prods = np.array(y[N*M:2*N*M]).reshape(N, M)
    Acts = np.array(y[2*N*M:2*N*M+N])
    Inhs = np.array(y[2*N*M+N:2*N*M+2*N])
    Difs = np.array(y[2*N*M+2*N:])

    print(f'time: {t:.2f} max F: {np.max(Facts):.2f} min F: {np.min(Facts):.2f} max P: {np.max(Prods):.2f} total factory capacity: {np.sum(Facts):.2f}')


    # read sensor data
    pos = Difs #Acts - Inhs #Difs #np.ones(N)
    vel = Acts - PREV_ACTS
    acc = np.ones(N) * 0

    

    PREV_ACTS = copy.copy(Acts)

    Cons = np.ones((N,M))
    Cons[:,0] = pos * C_SCALE[0]
    Cons[:,1] = vel * C_SCALE[1]
    Cons[:,2] = acc * C_SCALE[2]


    # Cellular plasticity
    F_dot, P_dot = cellular_plasticity(Facts, Prods, Cons)

    # Fitzhugh-Nagumo
    A_dot, I_dot = fitzhugh_nagumo(Acts, Inhs)

    # Diffusion
    Fxx = DF*diffusion(Facts)
    Pxx = DP*diffusion(Prods)
    Axx = DA*diffusion(Acts)
    Ixx = DI*diffusion(Inhs)
    Dxx = DD*diffusion(Difs)

    # Wave
    Adotw = wave(Acts)
    Idotw = wave(Inhs)

    # sum effects
    F_dot += Fxx
    P_dot += Pxx
    A_dot += Axx + Adotw
    I_dot += Ixx + Idotw
    D_dot = Dxx


    return np.concatenate([F_dot.ravel(), P_dot.ravel(), A_dot, I_dot, D_dot])

# initial conditions
F0 = np.ones((N,M)) #+ np.random.rand(N,M) * 0.1
P0 = np.ones((N,M))
A0 = np.zeros(N)
I0 = np.zeros(N)
D0 = np.ones(N) * 2*np.pi/N

F0[N//2,0] += 0.1
# F0[3*N//4,1] += 0.1
A0[N//2] += 2*np.pi

PREV_ACTS = copy.copy(A0)

# set initial conditions
y0 = np.concatenate([F0.ravel(), P0.ravel(), A0, I0, D0])

# simulation parameters

sol = solve_ivp(loopy_system, (0, T_END), y0, method='RK45', rtol=1e-3, atol=1e-6)



# Function to animate results

def animate_results_with_F_star(sol, N, M):
    # Unpack solution
    Facts = sol.y[:M * N].reshape(N, M, -1)  # Shape [N, M, time]
    Prods = sol.y[M * N:2 * N * M].reshape(N, M, -1)  # Shape [N, M, time]
    Acts = sol.y[2 * N * M:2 * N * M + N].T
    Inhs = sol.y[2 * N * M + N:2 * N * M + 2 * N].T
    Difs = sol.y[2 * N * M + 2 * N:].T

    # Define Z for F* calculation
    Z = (1.0 - D) / (1.0 / P_INF - 1.0 / P_LIM)

    # Prepare the figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize plots for factories, products, and F*
    lines_factories = [
        axs[0].plot([], [], label=f"Factory {i+1}")[0] for i in range(M)
    ]
    lines_products = [
        axs[0].plot([], [], label=f"Product {i+1}", linestyle='--')[0] for i in range(M)
    ]
    line_f_star, = axs[0].plot([], [], label="F*", color="black", linestyle="-.")

    axs[0].set_xlim(0, N - 1)
    axs[0].set_ylim(0, 30.0)
    axs[0].set_title("Factories, Products, and F*")
    axs[0].set_xlabel("Cell Number")
    axs[0].set_ylabel("Value")
    axs[0].legend()

    # Initialize plots for morphogens
    line_activator, = axs[1].plot([], [], label="Activator")
    line_inhibitor, = axs[1].plot([], [], label="Inhibitor")
    line_diffusive, = axs[1].plot([], [], label="Diffusive")

    axs[1].set_xlim(0, N - 1)
    axs[1].set_ylim(-1.0, 1.0)
    axs[1].set_title("Morphogen Values")
    axs[1].set_xlabel("Cell Number")
    axs[1].set_ylabel("Value")
    axs[1].legend()

    # Update function for animation
    def update(frame):
        print(f"Frame: {frame}/{len(sol.t)}")

        # Update factories and products
        for i in range(M):
            lines_factories[i].set_data(np.arange(N), Facts[:, i, frame])
            lines_products[i].set_data(np.arange(N), Prods[:, i, frame])

        # Calculate and update F*
        Cons = np.abs(Difs[frame]) * C_SCALE[0]  # Simplified consumption calculation
        F_star = Z * Cons
        line_f_star.set_data(np.arange(N), F_star)

        # Update morphogens
        line_activator.set_data(np.arange(N), Acts[frame])
        line_inhibitor.set_data(np.arange(N), Inhs[frame])
        line_diffusive.set_data(np.arange(N), Difs[frame])

        return (*lines_factories, *lines_products, line_f_star, line_activator, line_inhibitor, line_diffusive)

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(0,len(sol.t),10), blit=True, interval=1)

    # Display the animation
    plt.tight_layout()
    plt.show()

# Call the updated animation function
animate_results_with_F_star(sol, N, M)
