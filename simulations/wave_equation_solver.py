"""
WVU Interactive Robotics Laboratory

ODE Solver for the 2nd order wave equation with periodic endpoints

Camndon Reed
06/02/2025
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Right hand side for the 1d wave equation u_tt = c^2 u_xx with periodic boundary conditions, discretized on N points
def wave_rhs(t, U, c, dx, N):
    """
    Inputs
    - t: time
    - U: vector of length 2N
    - c: wave speed
    - dx: spatial grid spacing
    - N: Number of grid points in x
    Returns
    - dUdT: time derivative [u_t, v_t] of length 2N
    """
    # Split U into u and v
    u = U[:N]
    v = U[N:]

    # Compute u_xx using periodic second-difference
    # Use np.roll for periodic indexing
    u_right = np.roll(u, -1)
    u_left = np.roll(u, +1)
    u_xx = (u_right - 2*u + u_left) / (dx*dx)

    # Build the time derivatives
    du_dt = v
    dv_dt = (c**2) * u_xx

    # Stack into vector of length 2N
    return np.concatenate((du_dt, dv_dt))

# Solve u_tt = c^2 u_xx, x exists within the interval [0, L] with periodic boundary conditions
def solve_wave_periodic(L=1.0, N=200, c=1.0, T=2.0):
    """
    Inputs
    - L: the end of the interval
    - N: number of grid points
    - c: wave speed
    - T: final time
    
    Returns
    - x: array of grid points
    - sol: object returned by solve_ivp
    """
    # Spatial grid
    x = np.linspace(0, L, N, endpoint=False) # N points, dx spacing
    dx = L / N

    # Initial condition: pick any u(x,0) and v(x,0)=u_t(x,0)
    # EX sine bump to satisfy periodicity u(x,0) = sin(2pi*x/L), v(x,0)=0
    u0 = np.sin(2*np.pi*x/L)
    v0 = np.zeros_like(u0)

    # Pack into U0
    U0 = np.concatenate((u0, v0))

    t_span = (0.0, T)
    t_eval = np.linspace(0, T, 500) # Optional, output times

    # Integration
    sol = solve_ivp(fun=lambda t, 
                    U:wave_rhs(t, U, c, dx, N), 
                    t_span=t_span,
                    y0 = U0, 
                    t_eval=t_eval, 
                    method='RK45', ## RK45 or RK23
                    atol=1e-6, 
                    rtol=1e-6)
    
    return x, sol

# Run and plot
if __name__ == "__main__":
    # Parameters
    L = 1.0 # domain length
    N = 300 # Number of grid points
    c = 1.0 # wave speed
    T = 2.0 # final time

    x, sol = solve_wave_periodic(L, N, c, T)

    # Extract u(x,t) for all time steps
    # sol.y.shape = (2N, Nt), so u_all = sol.y[:N, :]
    u_all = sol.y[:N, :]

    # Set up animation
    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot(x, u_all[:, 0], color="tab:blue", lw=2)
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    title_text = ax.set_title(f"Wave at t = {sol.t[0]:.2f}")

    def update(frame_index):
        """
        Update function for FuncAnimation:
        - frame_index runs from 0 to Nt-1
        """
        line.set_ydata(u_all[:, frame_index])
        title_text.set_text(f"Wave at t = {sol.t[frame_index]:.2f}")
        return (line, title_text)

    ani = FuncAnimation(
        fig,
        update,
        frames=u_all.shape[1],
        interval=20,     # milliseconds between frames
        blit=False
    )

    # Display the animation interactively
    plt.show()