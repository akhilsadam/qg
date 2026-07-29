# ============================================================================
# ADD to bc.py : mask-driven sponge on chosen walls (cases 2, 3, 4).
# Put inside `class BC:` (reuses Sponge). Register in bc_library:
#     'sponge-walls': BC.sponge_walls,
# Builds the same wall geometry as masks.walls, inline (no new imports).
# ============================================================================

    @staticmethod
    def sponge_walls(state, grid, derivative,
                     sides='lrtb', width=0.025, gap=0.025,
                     h_gap_l=False, h_gap_r=False, v_gap_t=False, v_gap_b=False,
                     sponge=4.0, tolerance=1, **kwargs):
        Lx, Ly, Nx, Ny = grid.Lx, grid.Ly, grid.Nx, grid.Ny
        x = torch.linspace(0, Lx, Nx, device=grid.device)[None, :]
        y = torch.linspace(0, Ly, Ny, device=grid.device)[:, None]

        wx, wy = width * Lx, width * Ly
        gx, gy = gap * Lx, gap * Ly
        hx0 = (wx + gx) if h_gap_l else -wx
        hx1 = (Lx - wx - gx) if h_gap_r else (Lx + wx)
        vy0 = (wy + gy) if v_gap_b else -wy
        vy1 = (Ly - wy - gy) if v_gap_t else (Ly + wy)

        rects = []
        if 'l' in sides: rects.append((-wx,     wx,      vy0,      vy1))
        if 'r' in sides: rects.append((Lx - wx, Lx + wx, vy0,      vy1))
        if 'b' in sides: rects.append((hx0,     hx1,     -wy,      wy))
        if 't' in sides: rects.append((hx0,     hx1,     Ly - wy,  Ly + wy))

        def _rect_sdf(x0, x1, y0, y1):
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            hxx, hyy = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
            dx = torch.abs(x - cx) - hxx
            dy = torch.abs(y - cy) - hyy
            outside = torch.sqrt(torch.clamp(dx, min=0.) ** 2 + torch.clamp(dy, min=0.) ** 2)
            inside = torch.clamp(torch.maximum(dx, dy), max=0.)
            return outside + inside

        sdf = None
        for r in rects:
            d = _rect_sdf(*r)
            sdf = d if sdf is None else torch.minimum(sdf, d)

        tol = tolerance * max(Lx / Nx, Ly / Ny)
        mask = (torch.clamp((0 - sdf) / tol, -0.5, 0.5) + 0.5)[None, :, :].to(grid.ftype)

        return Sponge.vorticity_sponge(state, derivative, sponge, mask)
