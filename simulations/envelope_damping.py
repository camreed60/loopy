"""
WVU Interactive Robotics Laboratory

ODE Solver for the 2nd order wave equation on [0,L] with
“absorbing sponge layers” at each end so that outgoing waves
die out instead of reflecting.

Camndon Reed
06/02/2025
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def build_damping_profile(N, L, width_ratio=0.1, gamma_max=10.0):
    """
    Build a damping profile gamma[i] for x in [0, L], discretized with N points.
    The “sponge” zones live in [0, w] and [L-w, L], where w = width_ratio * L.
    gamma ramps from 0 (at x = w and x = L-w) up to gamma_max at the very edges.

    Returns:
      - gamma: length-N array with damping coefficients in [0, gamma_max].
    """
    x = np.linspace(0, L, N, endpoint=False)
    w = width_ratio * L
    gamma = np.zeros(N)

    # Left sponge: 0 <= x <= w
    left_zone = x <= w
    gamma[left_zone] = gamma_max * ((w - x[left_zone]) / w) ** 2

    # Right sponge: L - w <= x < L
    right_zone = x >= (L - w)
    gamma[right_zone] = gamma_max * ((x[right_zone] - (L - w)) / w) ** 2

    return gamma


def wave_rhs_absorbing(t, U, c, dx, N, gamma):
    """
    RHS for 1D wave equation u_tt = c^2 u_xx with sponge damping gamma[i].

    Inputs:
      - t: time (unused in RHS, but required by solve_ivp)
      - U: array of length 2N = [u_0..u_{N-1}, v_0..v_{N-1}]
      - c: wave speed
      - dx: spatial grid spacing
      - N: number of grid points
      - gamma: length-N array of damping coefficients

    Returns:
      - dUdt: array length 2N where
        dUdt[:N]   = du_i/dt = v_i,
        dUdt[N:]   = dv_i/dt = c^2 * u_xx_i - gamma_i * v_i.
    """
    u = U[:N]
    v = U[N:]

    # Pre‐allocate second‐derivative array
    u_xx = np.zeros_like(u)

    # Interior points: second difference with clamp at endpoints
    # i = 1..N-2 use u[i-1], u[i], u[i+1]
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[0:-2]) / (dx * dx)

    # Endpoints: we clamp u[-1]=0 and u[N]=0 when computing second difference
    # so that waves “see” zero outside the domain. This is not a true Dirichlet
    # clamp (it will slightly “reflect” if there were no damping), but since
    # gamma is large at the edges, the wave is already decaying well before i=0/N-1.
    u_xx[0]    = (u[1]  - 2 * u[0] + 0)      / (dx * dx)
    u_xx[-1]   = (0    - 2 * u[-1] + u[-2])  / (dx * dx)

    du_dt = v
    dv_dt = (c**2) * u_xx - gamma * v

    return np.concatenate((du_dt, dv_dt))


def solve_wave_absorbing(L=1.0, N=300, c=1.0, T=2.0,
                        sigma_ratio=0.02,
                        A_r=1.0, A_l=1.0,
                        width_ratio=0.1, gamma_max=10.0):
    """
    Solve u_tt = c^2 u_xx on [0, L], with absorbing (sponge) layers
    at each end so that outgoing pulses die out instead of reflecting.

    Inputs:
      - L: domain length
      - N: number of spatial grid points
      - c: wave speed
      - T: final time
      - sigma_ratio: width of initial Gaussian = sigma_ratio * L
      - A_r, A_l: amplitudes for right/left traveling parts (D'Alembert)
      - width_ratio: fraction of L used for each sponge layer
      - gamma_max: maximum damping coefficient at the very boundary

    Returns:
      - x: array of grid points (length N)
      - sol: result from solve_ivp (sol.y has shape (2N, Nt))
    """
    # 1) build spatial grid
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N

    # 2) build the damping profile gamma[i]
    gamma = build_damping_profile(N, L, width_ratio=width_ratio, gamma_max=gamma_max)

    # 3) set up D'Alembert‐based initial condition around x0 = L/2
    x0 = L / 2.0
    sigma = sigma_ratio * L
    G = np.exp(- ((x - x0)**2) / (2 * sigma**2))
    Gprime = -((x - x0)/(sigma**2)) * G

    #    u0 = (A_r + A_l) G
    u0 = (A_r + A_l) * G
    #    v0 = c*(A_l - A_r) Gprime
    v0 = c * (A_l - A_r) * Gprime

    U0 = np.concatenate((u0, v0))

    # 4) time sampling
    t_eval = np.linspace(0, T, 500)

    # 5) integrate with solve_ivp
    sol = solve_ivp(
        fun=lambda t, U: wave_rhs_absorbing(t, U, c, dx, N, gamma),
        t_span=(0.0, T),
        y0=U0,
        t_eval=t_eval,
        method='RK45',
        atol=1e-6,
        rtol=1e-6
    )

    return x, sol


# --------------------------------------------------
# Main block to run and animate
# --------------------------------------------------

if __name__ == "__main__":
    # Parameters
    L = 1.0            # domain length
    N = 300            # number of grid points
    c = 1.0            # wave speed
    T = 2.0            # final time
    sigma_ratio = 0.05 # initial Gaussian width = 0.05 * L
    A_r = 1.0          # amplitude of right‐moving pulse
    A_l = -1.0         # amplitude of left‐moving pulse
    width_ratio = 0.1  # sponge width = 10% of domain on each side
    gamma_max = 10.0   # max damping at x=0 and x=L

    # Solve with absorbing boundaries
    x, sol = solve_wave_absorbing(
        L=L, N=N, c=c, T=T,
        sigma_ratio=sigma_ratio,
        A_r=A_r, A_l=A_l,
        width_ratio=width_ratio,
        gamma_max=gamma_max
    )

    # Extract u(x,t) for all timesteps: sol.y[:N,:] has shape (N, Nt)
    u_all = sol.y[:N, :]

    # Set up animation
    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot(x, u_all[:, 0], color="tab:blue", lw=2)
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2 * max(abs(A_r), abs(A_l)), 1.2 * max(abs(A_r), abs(A_l)))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    title_text = ax.set_title(f"Wave at t = {sol.t[0]:.2f}")

    def update(frame_index):
        line.set_ydata(u_all[:, frame_index])
        title_text.set_text(f"Wave at t = {sol.t[frame_index]:.2f}")
        return (line, title_text)

    ani = FuncAnimation(
        fig,
        update,
        frames=u_all.shape[1],
        interval=20,  # milliseconds between frames
        blit=False    # full redraw so title updates cleanly
    )

    plt.show()
