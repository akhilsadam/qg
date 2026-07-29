import torch
import torch.nn.functional as F
from qg.solver.opt.basis import to_physical, to_spectral

# Generates forcing based on specified wavenumber and time effects
def unscaled_cosine(state, grid, derivative, 
                    A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0,
                    **kwargs):
    # grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None] # meshgrid for x, y
    
    w = A * (torch.cos(B * X + C * state.t)) \
        + D * (torch.cos(E * Y + F * state.t)) [None,:,:]
    
    wh = to_spectral(w)
    return wh

def local_cosines(state, grid, derivative, 
                    A=[2.0], kx=[2.0], ky=[2.0], cx=[0.5], cy=[0.5], w=0.0, sigma=[0.1], **kwargs):
    # grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)
    X, Y = x[None,:],y[:,None] # meshgrid for x, y
    
    omega = 0.0
    for Ai, kxi, kyi, cxi, cyi, sigmai in zip(A, kx, ky, cx, cy, sigma):
        # Gaussian envelope centered at (cx * Lx, cy * Ly) with width sigma
        if sigmai > 0:
            gsn = torch.exp(-((X - cxi * grid.Lx)**2 + (Y - cyi * grid.Ly)**2) / (2 * (sigmai * grid.Lx)**2)) / (2 * torch.pi * (sigmai * grid.Lx)**2) # Gaussian envelope
        else:
            gsn = 1.0
        omega = omega + gsn * Ai * (torch.cos(kxi * X + w * state.t) \
            + torch.cos(kyi * Y + w * state.t)) [None,:,:]
    
    wh = to_spectral(omega)
    return wh

# ---------------------------------------------------------------------------
# Separable trig forcing (cosine / sine)
#
#   F(x, y, t) = A * trig(B * x + C * t) + D * trig(E * y + F * t)
#
#     A: x-term amplitude   B: x wavenumber   C: x temporal frequency
#     D: y-term amplitude   E: y wavenumber   F: y temporal frequency
## ---------------------------------------------------------------------------
def _separable_trig(state, grid, trig, A, B, C, D, E, F):
    x = torch.linspace(0, grid.Lx, grid.Nx, device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=grid.device)
    X = x[None, None, :]  # (1, 1, Nx)
    Y = y[None, :, None]  # (1, Ny, 1)

    w = A * trig(B * X + C * state.t) \
        + D * trig(E * Y + F * state.t)  # (1, Ny, Nx)

    return to_spectral(w)

def cosine(state, grid, derivative, # add state as first argument if time-dependent
           A=0.0, B=0.0, C=0.0, D=1.0, E=1.0, F=0.0,
           **kwargs):
    return _separable_trig(state, grid, torch.cos, A, B, C, D, E, F)

def sine(state, grid, derivative, # add state as first argument if time-dependent
         A=0.0, B=0.0, C=0.0, D=1.0, E=1.0, F=0.0,
         **kwargs):
    return _separable_trig(state, grid, torch.sin, A, B, C, D, E, F)

# ---------------------------------------------------------------------------
# Piecewise forcing
#
#     g(x) = jet_lat * Ly + slope * (x - Lx/2)          (tilted jet axis)
#
#                -amp * sin(pi * y / g(x))                  , y <  g(x)
#     F(x, y) =
#                 amp * sin(pi * (y - g(x)) / (Ly - g(x)))  , y >= g(x)
#
#     amp = tau0 * 2*pi / (amp_span * Ly)
## ---------------------------------------------------------------------------
def piecewise(state, grid, derivative, # add state as first argument if time-dependent
              tau0=1.0, slope=0.2, jet_lat=0.5, amp_span=0.9,
              **kwargs):
    x = torch.linspace(0, grid.Lx, grid.Nx, device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=grid.device)
    X, Y = x[None, :], y[:, None]

    # g(x): jet axis latitude, tilted linearly across x about the domain center
    gx = jet_lat * grid.Ly + slope * (X - grid.Lx / 2)

    eps = 1e-12
    gx = torch.clamp(gx, eps, grid.Ly - eps)

    amp = tau0 * (2 * torch.pi) / (amp_span * grid.Ly)

    w = torch.where(
        Y < gx,
        -amp * torch.sin(torch.pi * Y / gx),
        amp * torch.sin(torch.pi * (Y - gx) / (grid.Ly - gx)),
    )

    return to_spectral(w[None, :, :])

def split_cosine(state, grid, derivative, # add state as first argument if time-dependent
                 A=1.0, y_split=None,
                 **kwargs):
    """
    Piecewise cosine wind forcing (mentor's spec):
        F(y) = -A*cos(y)   for 0 < y < y_split
                A*cos(y)   for y_split < y < Ly
    Constant in x and time. y_split defaults to Ly/2 (= pi on a 2*pi domain).
    Intended for doubly-periodic runs (penalty = 0, bc = periodic).
    """
    Nx = grid.Nx
    Ny = grid.Ny
    if y_split is None:
        y_split = grid.Ly / 2

    x = torch.linspace(0, grid.Lx, Nx, device=grid.device)[None, None, :]  # (1,1,Nx)
    y = torch.linspace(0, grid.Ly, Ny, device=grid.device)[None, :, None]  # (1,Ny,1)

    w = torch.where(y < y_split, -A * torch.cos(y), A * torch.cos(y)) + 0.0 * x  # (1,Ny,Nx)

    return to_spectral(w)
####################################################################################################

valid_fc = lambda _fc: hasattr(_fc, 'function') and _fc.function in fc_library

fc_library = {
    'unscaled_cosine': unscaled_cosine,
    'local_cosines': local_cosines,
    'cosine': cosine,
    'sine': sine,
    'piecewise': piecewise,
    'split_cosine': split_cosine,
}

def solve_forcing(_fc):
    if _fc is None or not isinstance(_fc, dict) or 'function' not in _fc or _fc['function'] not in fc_library:
        return _fc
    return lambda *args: fc_library[_fc['function']](*args, **_fc)