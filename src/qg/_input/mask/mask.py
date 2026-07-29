import torch
import numpy as np
from PIL import Image
import os

# TODO add subpixel rendering / anti-aliasing to non-SDF methods

def add_margin(pil_img, width, height, top, left, color):
    result = Image.new(pil_img.mode, (width, height), color)
    result.paste(pil_img, (left, top))
    return result

def sdf_raster(d, r, tolerance):
    # mask = torch.zeros_like(d)
    # mask[d < r] = 1  # Inside the circle
    # mask[torch.abs(d - r) < tolerance] = 0.5  # Boundary (within tolerance)

    mask = torch.clamp((r - d) / tolerance, -0.5, 0.5) + 0.5

    return mask

def _rect_sdf(x, y, x0, x1, y0, y1):
    # Signed distance to the axis-aligned rectangle [x0, x1] x [y0, y1].
    # Negative inside, positive outside. x is (1, Nx), y is (Ny, 1) -> (Ny, Nx).
    # Union of rectangles is min() of their SDFs; feed the result to sdf_raster
    # with r=0 so the zero level set is the region boundary.
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0)
    hy = 0.5 * (y1 - y0)
    dx = torch.abs(x - cx) - hx
    dy = torch.abs(y - cy) - hy
    outside = torch.sqrt(torch.clamp(dx, min=0.0) ** 2 + torch.clamp(dy, min=0.0) ** 2)
    inside = torch.clamp(torch.maximum(dx, dy), max=0.0)
    return outside + inside

def circular(grid, derivative, # add state as first argument if time-dependent
             r, tolerance=1, invert=False,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, Nx, device = grid.device)
    y = torch.linspace(0, Ly, Ny, device = grid.device)

    # Find the center of the domain
    x_center = Lx / 2
    y_center = Ly / 2

    # Compute the distance of each point from the center
    distance = torch.sqrt((x[None,:] - x_center)**2 + (y[:,None] - y_center)**2) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = sdf_raster(distance, r, tolerance * max(Lx/Nx, Ly/Ny))

    if invert:
        mask = 1 - mask

    return mask[None,:,:]  # Add batch dimension

def box(grid, derivative, # add state as first argument if time-dependent
             r, tolerance=1, invert=False,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, Nx, device = grid.device)
    y = torch.linspace(0, Ly, Ny, device = grid.device)

    # Find the center of the domain
    x_center = Lx / 2
    y_center = Ly / 2

    # Compute the distance of each point from the center
    distance = torch.max(torch.abs(x[None,:] - x_center), torch.abs(y[:,None] - y_center)) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = sdf_raster(distance, r, tolerance * max(Lx/Nx, Ly/Ny))

    if invert:
        mask = 1 - mask

    return mask[None,:,:]  # Add batch dimension

# ===========================================================================
# Wall / boundary masks for the BC cases.
#
# `walls` returns a strip of fractional `width` on the chosen `sides` (any of
# 'l','r','t','b'). Outer edges run past the domain so the wall is filled up to
# the boundary; only inner or trimmed edges get the ~1-cell SDF taper. Gap flags
# trim a strip's *end* by (width+gap), detaching it from the perpendicular wall
# (opening that corner):
#     h_gap_l / h_gap_r  trim the left / right end of the top & bottom strips
#     v_gap_t / v_gap_b  trim the top / bottom end of the left & right strips
#
# The five image cases map to these masks (each side is either a Brinkman wall
# or a sponge region; assign the penalty per mask in the config / operator):
#     case 1  no-slip all sides        : border
#     case 2  sponge all sides         : border
#     case 3  no-slip L/R, sponge T/B  : walls_lr        + walls_tb_gap
#     case 4  no-slip L, sponge T/R/B  : wall_l          + walls_trb_lgap
#     case 5  free-slip all sides      : walls_lr (chi_x) + walls_tb (chi_y)
# ===========================================================================
def _wall_strips(Lx, Ly, sides, width, gap,
                 h_gap_l=False, h_gap_r=False, v_gap_t=False, v_gap_b=False):
    wx, wy = width * Lx, width * Ly
    gx, gy = gap * Lx, gap * Ly
    hx0 = (wx + gx) if h_gap_l else -wx           # top/bottom left end
    hx1 = (Lx - wx - gx) if h_gap_r else (Lx + wx)  # top/bottom right end
    vy0 = (wy + gy) if v_gap_b else -wy           # left/right bottom end
    vy1 = (Ly - wy - gy) if v_gap_t else (Ly + wy)  # left/right top end
    strips = []
    if 'l' in sides: strips.append((-wx,     wx,      vy0,      vy1))
    if 'r' in sides: strips.append((Lx - wx, Lx + wx, vy0,      vy1))
    if 'b' in sides: strips.append((hx0,     hx1,     -wy,      wy))
    if 't' in sides: strips.append((hx0,     hx1,     Ly - wy,  Ly + wy))
    return strips

def walls(grid, derivative, # add state as first argument if time-dependent
          sides='lrtb', width=0.025, gap=0.025,
          h_gap_l=False, h_gap_r=False, v_gap_t=False, v_gap_b=False,
          tolerance=1, invert=False,
          **kwargs):
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    x = torch.linspace(0, Lx, Nx, device=grid.device)[None, :]
    y = torch.linspace(0, Ly, Ny, device=grid.device)[:, None]

    strips = _wall_strips(Lx, Ly, sides, width, gap,
                          h_gap_l, h_gap_r, v_gap_t, v_gap_b)

    if len(strips) == 0:
        return torch.zeros((1, Ny, Nx), device=grid.device)

    sdf = None
    for s in strips:
        d = _rect_sdf(x, y, *s)
        sdf = d if sdf is None else torch.minimum(sdf, d)

    mask = sdf_raster(sdf, 0, tolerance * max(Lx / Nx, Ly / Ny))

    if invert:
        mask = 1 - mask

    return mask[None, :, :]  # Add batch dimension

# --- named presets (thin wrappers over walls, kept >3 params on purpose so
#     obstacle.solve_mask treats them as static (grid, derivative) masks) ---
def border(grid, derivative, width=0.025, tolerance=1, invert=False, **kwargs):
    return walls(grid, derivative, sides='lrtb',
                 width=width, tolerance=tolerance, invert=invert)

def walls_lr(grid, derivative, width=0.025, tolerance=1, invert=False, **kwargs):
    return walls(grid, derivative, sides='lr',
                 width=width, tolerance=tolerance, invert=invert)

def walls_tb(grid, derivative, width=0.025, tolerance=1, invert=False, **kwargs):
    return walls(grid, derivative, sides='tb',
                 width=width, tolerance=tolerance, invert=invert)

def walls_tb_gap(grid, derivative, width=0.025, gap=0.025, tolerance=1, invert=False, **kwargs):
    return walls(grid, derivative, sides='tb', gap=gap, h_gap_l=True, h_gap_r=True,
                 width=width, tolerance=tolerance, invert=invert)

def wall_l(grid, derivative, width=0.025, tolerance=1, invert=False, **kwargs):
    return walls(grid, derivative, sides='l',
                 width=width, tolerance=tolerance, invert=invert)

def walls_trb_lgap(grid, derivative, width=0.025, gap=0.025, tolerance=1, invert=False, **kwargs):
    return walls(grid, derivative, sides='trb', gap=gap, h_gap_l=True,
                 width=width, tolerance=tolerance, invert=invert)

def fpc(grid, derivative, # add state as first argument if time-dependent
             tolerance=1,
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, Nx, device = grid.device)
    y = torch.linspace(0, Ly, Ny, device = grid.device)

    # Find the center of the domain
    x_center = Lx / 8
    y_center = Ly / 2
    r = Lx / 16  # Radius of the circle

    # Compute the distance of each point from the center
    distance = torch.sqrt((x[None,:] - x_center)**2 + (y[:,None] - y_center)**2) # yx

    # Create the mask: inside the circle (distance < r) is 1
    mask = sdf_raster(distance, r, tolerance * max(Lx/Nx, Ly/Ny))

    return mask[None,:,:]  # Add batch dimension

def cape(grid, derivative, height=1/4, sigma=1/16, tolerance=1, pad=0.24, **kwargs):
    # Use grid object for domain size and number of grid points
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Nx/Ny, Nx, device = grid.device)[None, :]
    y = torch.flip(torch.linspace(0, 1, Ny, device = grid.device), (0,))[:, None]

    # Find the center of the domain
    x_center = 3 / 16
    y_center = pad + 1 / 8

    # vertical cape profile
    cape_profile = (y_center + height * torch.exp(-((x-x_center)/sigma)**2) - y)
    cape = (cape_profile > 0) \
        * (y > y_center - 1/8)

    mask = sdf_raster(-1 * cape, 0, tolerance * max(Lx/Nx, Ly/Ny))

    return mask[None,:,:]  # Add batch dimension

def im(grid, derivative, # add state as first argument if time-dependent
             mask='osk.png', th=0.5, blur=0.0, pad=0, pad_mode='lrtd',
             **kwargs):
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    if pad > 0:
        # pad image out with 0-padding
        pads = [0,0,0,0]
        if 'l' in pad_mode: # left-right swap
            pads[1] = pad
        if 'r' in pad_mode:
            pads[0] = pad
        if 't' in pad_mode:
            pads[2] = pad
        if 'd' in pad_mode:
            pads[3] = pad

    if isinstance(mask, str):
        # get current file path
        path = os.path.dirname(os.path.abspath(__file__))

        # check if mask is not an absolute path
        if not os.path.isabs(mask):
            mask = os.path.join(path, 'mask', mask)
        img = Image.open(mask)

        # img = img.resize((Nx, Ny))
        # pad image if pad > 0
        if pad > 0:
            img = img.resize((Nx - pads[0] - pads[1], Ny - pads[2] - pads[3]))
            img = add_margin(img, Nx, Ny, pads[0], pads[1], 0)
        else:
            img = img.resize((Nx, Ny))

        img = img.convert('L')
        img = np.array(img).astype(np.float32)
        img /= np.max(img)
        img = torch.tensor(img)

    else:
        raise ValueError(f"Mask should be a string path to an image file, got {mask} instead. Not implemented yet.")



    mask = torch.where(img > th, 1.0, 0.0).to(grid.device, dtype=grid.ftype)

    if blur > 0.0:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask.cpu().numpy(), sigma=blur)
        mask = torch.tensor(mask).to(grid.device, dtype=grid.ftype)

    return mask[None,:,:]  # Add batch dimension


def nc(grid, derivative, # add state as first argument if time-dependent
             mask='riot_070725', clip=-200, pad=0.08, pad_mode='lrtd',
             **kwargs):

    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny


    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(mask):
        mask_path = os.path.join(path, 'mask', f'{mask}.nc')
        npy_path = os.path.join(path, 'mask', f'{mask}.npy')
        png_path = os.path.join(path, 'mask', f'{mask}.png')

    if not os.path.exists(png_path):
        if os.path.exists(npy_path):
            data = np.load(npy_path)
        elif os.path.exists(mask_path):
            from netCDF4 import Dataset
            with Dataset(mask_path, 'r') as nc_file:
                data = nc_file.variables[nc_library[mask]]
                data = np.array(data)
                print(data.shape)
                print(np.min(data), np.max(data))
                np.save(npy_path, data)

        mask = torch.from_numpy(data > clip)
        Image.fromarray(mask.numpy()).save(png_path)

    return im(grid, derivative, mask=png_path, th=0.5, blur=0.0, pad=int(pad * Nx), pad_mode=pad_mode) # use im function to return mask


####################################################################################################

valid_mask = lambda _mask: hasattr(_mask, 'function') and _mask.function in mask_library

mask_library = {
    'fpc': fpc,
    'cape': cape,
    'circular': circular,
    'box' : box,
    'walls': walls,
    'border': border,
    'walls_lr': walls_lr,
    'walls_tb': walls_tb,
    'walls_tb_gap': walls_tb_gap,
    'wall_l': wall_l,
    'walls_trb_lgap': walls_trb_lgap,
    'image': im,
    'netCDF': nc,
}

nc_library = {
    'riot_070725': 'raw_bath', # /gdata/projects/dri_riot/Grids/2025/Jul07/socal0450m/grids_riot_sa0450m.nc; point conception, channel islands near Santa Barbara, CA
}


def solve_mask(_mask):
    if _mask is None or not isinstance(_mask, dict) or 'function' not in _mask or _mask['function'] not in mask_library:
        return _mask
    return lambda *args: mask_library[_mask['function']](*args, **_mask)