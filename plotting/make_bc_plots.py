#!/usr/bin/env python
# =============================================================================
# Plotting for the 15 BC/forcing runs from the cluster qg solver.
#
# Produces, PER forcing category (dgyre / fgyre / pw):
#   * 4 field plots x 5 BC cases  (instantaneous psi, instantaneous vorticity,
#     mean psi, steady-mean psi) with SYMMETRIC colorbars (white at 0) whose
#     vmax is SHARED across the 5 cases within that category+field.
#   * 3 overlay plots (5 cases, colored, legend): energy spectrum, energy vs
#     time, v-profile at x=pi.  Spectrum & energy-vs-time are cut to their
#     downward-sloping part and given a best-fit slope (saved to CSV).
#
# Total: 3 categories x (4x5 field + 3 overlay) = 60 field + 9 overlay = 69 png.
# =============================================================================

import os, sys, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                      # headless cluster: no display
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- their solver utilities -------------------------------------------------
UTILS_DIR = "/home/mjmvega/Codes/QG/Work/"
sys.path.append(UTILS_DIR)
from Grid.grid import Grid
from Operators.operators import SpectralDerivatives
from Operators.spectral_conversion import to_physical, to_spectral, get_puv
from Plotting.plots import spectrum
from Initial_forcing.ics import int_sq

# =============================================================================
# CONFIG  -- the only things you should need to touch
# =============================================================================
RUNS_DIR   = os.path.expanduser("~/QG/qg/runs/qg")           # where <scenario>/qg_data.npy live
PLOTS_ROOT = "/gdata/Results/mjmvega/QG/bc_study_plots"       # organized output root

Nx, Ny = 256, 256
Lx, Ly = 2*np.pi, 2*np.pi

VORTICITY_CHANNEL = 0        # channel 0 of the .npy is physical vorticity (from qg.py)
STEADY_START_FRAC = 0.5      # "steady state" = the last 50% of frames (raise if not settled)
DROP_FIRST_FRAME  = True     # frame 0 is the rest IC (all zeros) -> skip it

UTILS_XY   = False           # False: feed utils [Ny,Nx] (row=y). Set True if the SANITY
                             #        CHECK shows psi / v transposed (utils use [x,y]).
CROP_STAT  = 0.05            # ignore outer 5% when SETTING the vorticity color scale
CMAP       = "RdBu_r"        # diverging, ~white at 0
N_CONTOURS = 15

SPEC_FIT_END_FRAC = 0.85     # spectrum fit: from spectral peak up to 85% of k-range
CATEGORIES_TO_RUN = ["dgyre", "fgyre", "pw"]   # trim to ["dgyre"] for a quick first test

# category -> pretty title
CATEGORY_TITLE = {
    "dgyre": r"Double gyre  ($-\cos(y/2)$)",
    "fgyre": r"Four gyre  ($\sin 2y$)",
    "pw":    r"Tilted jet  (piecewise)",
}
# BC case label -> scenario-name prefix (matches your config filenames)
CASES = {
    "Case 1: no-slip all":              "bc1_noslip_all",
    "Case 2: sponge all":               "bc2_sponge_all",
    "Case 3: no-slip L/R + sponge T/B": "bc3_noslipLR_spongeTB",
    "Case 4: no-slip L + sponge T/R/B": "bc4_noslipL_spongeTRB",
    "Case 5: free-slip all":            "bc5_freeslip_all",
}
CASE_COLORS = dict(zip(CASES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))

# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grid   = Grid(Lx, Ly, Nx, Ny)
spec   = SpectralDerivatives(grid)
L      = Lx * Ly
MID    = Nx // 2                      # x = pi column index

def _ut(field2d):
    """np [Ny,Nx] -> torch on device, transposed to [x,y] iff UTILS_XY."""
    a = field2d.T if UTILS_XY else field2d
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)

def _un(field2d_t):
    """torch -> np [Ny,Nx] (undo the UTILS_XY transpose)."""
    a = field2d_t.detach().cpu().numpy()
    return a.T if UTILS_XY else a

def q_to(qh_field2d, which):
    """From a 2D vorticity field, return physical 'psi' or 'v' in [Ny,Nx]."""
    qh = to_spectral(_ut(qh_field2d))
    ph, uh, vh = get_puv(qh, spec)
    return _un(to_physical(ph if which == "psi" else vh))

def clean(name):
    return name.replace(" ", "_").replace(":", "").replace("/", "").replace("+", "")

# =============================================================================
# per-run processing: load .npy, derive field snapshots + 1D metrics, free array
# =============================================================================
def process_run(path):
    arr = np.load(path)                                   # [B, T, C, Ny, Nx]
    q_all = arr[0, :, VORTICITY_CHANNEL, :, :].astype(np.float32)   # [T, Ny, Nx]
    del arr
    T = q_all.shape[0]
    f0 = 1 if (DROP_FIRST_FRAME and T > 1) else 0
    s_start = max(f0, int(STEADY_START_FRAC * T))

    q_inst        = q_all[-1]                             # instantaneous steady (last frame)
    q_mean_all    = q_all[f0:].mean(axis=0)               # mean over all (real) frames
    q_mean_steady = q_all[s_start:].mean(axis=0)          # mean over steady window

    result = {
        "q_inst":          q_inst,
        "psi_inst":        q_to(q_inst,        "psi"),
        "psi_mean_all":    q_to(q_mean_all,    "psi"),
        "psi_mean_steady": q_to(q_mean_steady, "psi"),
        "v_profile":       q_to(q_mean_steady, "v")[:, MID],   # v(y) at x=pi, steady-mean
    }

    # energy vs time (all frames) + steady-averaged spectrum (nonlinear -> per frame)
    E = np.empty(T, dtype=np.float64)
    ek_sum, kvals, n_steady = None, None, 0
    for t in range(T):
        qh = to_spectral(_ut(q_all[t]))
        ph, uh, vh = get_puv(qh, spec)
        E[t] = 0.5 * (float(int_sq(uh, grid)) + float(int_sq(vh, grid))) / L
        if t >= s_start:
            e = torch.abs(uh) ** 2 + torch.abs(vh) ** 2
            z = torch.abs(qh) ** 2
            k, (ek, zk) = spectrum([e, z], spec)
            ek = ek.detach().cpu().numpy()
            ek_sum = ek if ek_sum is None else ek_sum + ek
            kvals = k.detach().cpu().numpy()
            n_steady += 1
    result["E"] = E
    result["frames"] = np.arange(T)
    result["k"] = kvals
    result["ek"] = ek_sum / max(n_steady, 1)
    result["steady_start"] = s_start
    return result

# =============================================================================
# 2D field plotting (symmetric, white-at-0, shared vmax)
# =============================================================================
def plot_field(field, title, vmax, save_path, label, contours=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    norm = Normalize(vmin=-vmax, vmax=vmax)                # symmetric -> 0 is the white center
    im = ax.imshow(field, cmap=CMAP, norm=norm, origin="lower", interpolation="nearest",
                   extent=[0, Lx, 0, Ly])
    if contours:
        cs = ax.contour(np.linspace(0, Lx, field.shape[1]),
                        np.linspace(0, Ly, field.shape[0]),
                        field, levels=np.linspace(-vmax, vmax, N_CONTOURS),
                        colors="k", linewidths=0.5)
        ax.clabel(cs, fontsize=6)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=np.linspace(-vmax, vmax, 5))
    cbar.set_label(label)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def crop_for_stat(f):
    mx, my = int(CROP_STAT * f.shape[1]), int(CROP_STAT * f.shape[0])
    return f[my:f.shape[0]-my, mx:f.shape[1]-mx] if (mx or my) else f

# =============================================================================
# best-fit slope on the downward part (log-log for spectrum, lin for energy)
# =============================================================================
def fit_downslope(x, y, loglog, end_frac=SPEC_FIT_END_FRAC):
    good = (x > 0) & np.isfinite(y) & (y > (0 if loglog else -np.inf))
    x, y = x[good], y[good]
    if len(x) < 4:
        return None
    start = int(np.argmax(y))                      # start at the peak
    end = max(start + 3, int(len(x) * end_frac))
    xf, yf = x[start:end], y[start:end]
    X = np.log10(xf) if loglog else xf
    Y = np.log10(yf) if loglog else yf
    slope, icpt = np.polyfit(X, Y, 1)
    yp = slope * X + icpt
    r2 = 1 - np.sum((Y - yp) ** 2) / np.sum((Y - np.mean(Y)) ** 2)
    fit_y = 10 ** yp if loglog else yp
    return dict(xf=xf, fit_y=fit_y, slope=slope, r2=r2, k0=xf[0], k1=xf[-1])

# =============================================================================
# MAIN
# =============================================================================
for cat in CATEGORIES_TO_RUN:
    print(f"\n===== category: {cat} =====")
    results = {}
    for label, prefix in CASES.items():
        path = os.path.join(RUNS_DIR, f"{prefix}__{cat}", "qg_data.npy")
        if not os.path.exists(path):
            warnings.warn(f"MISSING (skipped): {path}")
            continue
        print(f"  loading {label} ...")
        results[label] = process_run(path)
    if not results:
        print(f"  no runs found for {cat}; skipping.")
        continue

    # ---- folders for this category ----
    base = os.path.join(PLOTS_ROOT, cat)
    fdir = {k: os.path.join(base, v) for k, v in {
        "psi_inst":   "streamfunction_instantaneous",
        "vort_inst":  "vorticity_instantaneous",
        "psi_avg":    "streamfunction_average",
        "psi_steady": "streamfunction_steady_average",
        "overlays":   "overlays",
    }.items()}
    for d in fdir.values():
        os.makedirs(d, exist_ok=True)

    # ---- shared vmax across the 5 cases (per field type) ----
    vmax_psi_inst   = max(np.max(np.abs(r["psi_inst"]))        for r in results.values())
    vmax_psi_avg    = max(np.max(np.abs(r["psi_mean_all"]))    for r in results.values())
    vmax_psi_steady = max(np.max(np.abs(r["psi_mean_steady"])) for r in results.values())
    vmax_vort = np.percentile(
        np.concatenate([np.abs(crop_for_stat(r["q_inst"])).ravel() for r in results.values()]),
        99.5)

    # ---- 4 field plots x 5 cases ----
    for label, r in results.items():
        c = clean(label)
        plot_field(r["psi_inst"], f"{label}\ninstantaneous $\\psi$", vmax_psi_inst,
                   os.path.join(fdir["psi_inst"], f"{c}.png"), r"$\psi$")
        plot_field(r["q_inst"], f"{label}\ninstantaneous $\\omega$", vmax_vort,
                   os.path.join(fdir["vort_inst"], f"{c}.png"), r"$\omega$", contours=False)
        plot_field(r["psi_mean_all"], f"{label}\nmean $\\psi$", vmax_psi_avg,
                   os.path.join(fdir["psi_avg"], f"{c}.png"), r"$\overline{\psi}$")
        plot_field(r["psi_mean_steady"], f"{label}\nsteady-mean $\\psi$", vmax_psi_steady,
                   os.path.join(fdir["psi_steady"], f"{c}.png"), r"$\overline{\psi}_{steady}$")
    print(f"  field plots done (vmax psi_inst={vmax_psi_inst:.3g}, vort={vmax_vort:.3g})")

    title = CATEGORY_TITLE.get(cat, cat)

    # ---- OVERLAY 1: v-profile at x=pi ----
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.linspace(0, Ly, Ny)
    for label, r in results.items():
        ax.plot(y, r["v_profile"], color=CASE_COLORS[label], lw=2, label=label)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"Steady $v(x=\\pi,\\,y)$ — {title}")
    ax.set_xlabel("y"); ax.set_ylabel(r"$v(x=\pi, y)$")
    ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fdir["overlays"], "v_profile_xpi.png"),
                dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # ---- OVERLAY 2: energy vs time (cut to post-peak decay + best-fit slope) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    e_lines = ["case,slope_dE/dframe,r2,frame0,frame1"]
    for label, r in results.items():
        col = CASE_COLORS[label]
        ax.plot(r["frames"], r["E"], color=col, lw=1.2, alpha=0.45)
        f = fit_downslope(r["frames"].astype(float), r["E"], loglog=False)
        if f:
            ax.plot(f["xf"], r["E"][int(np.argmax(r["E"])):
                                     int(np.argmax(r["E"])) + len(f["xf"])],
                    color=col, lw=2.4, label=f"{label} (slope={f['slope']:.2e})")
            ax.plot(f["xf"], f["fit_y"], color=col, ls="--", lw=1.4)
            e_lines.append(f"{label},{f['slope']:.4e},{f['r2']:.3f},{f['k0']:.0f},{f['k1']:.0f}")
        else:
            ax.plot([], [], color=col, lw=2.4, label=label)
    ax.set_title(f"Kinetic energy vs time — {title}")
    ax.set_xlabel("frame"); ax.set_ylabel("KE")
    ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fdir["overlays"], "energy_over_time.png"),
                dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    with open(os.path.join(fdir["overlays"], "energy_decay_slopes.csv"), "w") as fh:
        fh.write("\n".join(e_lines))

    # ---- OVERLAY 3: energy spectrum (cut to inertial slope + best-fit) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    s_lines = ["case,alpha,r2,k_min,k_max"]
    for label, r in results.items():
        col = CASE_COLORS[label]
        f = fit_downslope(r["k"], r["ek"], loglog=True)
        if f:
            ax.loglog(f["xf"], r["ek"][int(np.argmax(r["ek"])):
                                       int(np.argmax(r["ek"])) + len(f["xf"])],
                      color=col, lw=2.2, label=f"{label} ($\\alpha$={f['slope']:.2f})")
            ax.loglog(f["xf"], f["fit_y"], color=col, ls="--", lw=1.4)
            s_lines.append(f"{label},{f['slope']:.4f},{f['r2']:.3f},{f['k0']:.2f},{f['k1']:.2f}")
        else:
            ax.loglog(r["k"], r["ek"], color=col, lw=2.2, label=label)
    k_ref = np.linspace(6, 40, 50)
    ax.loglog(k_ref, 10 ** (-1.5) * k_ref ** -3, "k--", lw=1.6, label=r"$k^{-3}$")
    ax.set_title(f"Steady energy spectrum — {title}")
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.grid(True, which="both", ls="--", alpha=0.35); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fdir["overlays"], "energy_spectrum.png"),
                dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    with open(os.path.join(fdir["overlays"], "spectrum_slopes.csv"), "w") as fh:
        fh.write("\n".join(s_lines))

    print(f"  overlays + CSVs written to {os.path.join(base, 'overlays')}")

print("\nDONE. Plots under:", PLOTS_ROOT)
