# 15 runs = 3 forcings x 5 boundary conditions  (QG double/four-gyre + jet)

## Code changes (once)
1. obstacle.py  -> add brinkman_free_slip_penalty              (Case 5)
2. operator/__init__.py -> 3-way penalty selector + import     (Case 5)
3. bc.py -> add BC.sponge_walls (see bc_sponge_walls_ADD_TO_bc.py),
            register 'sponge-walls': BC.sponge_walls           (Cases 2,3,4)
4. base qg.pde -> add `free_slip: false` so Hydra allows the key(Case 5)
5. masks.py -> already rewritten (border / walls_* presets)

## Run
Copy *.yaml into conf/scenario/, then:  bash run_all.sh
or:  python src/qg/train.py scenario=bc5_freeslip_all__dgyre
(fix the entrypoint to however you launch it)

## Files: bc{1..5}_..__{forcing}
forcing:
  dgyre = -cos(y/2)  (D=-1, E=1/2)  <- your working double-gyre forcing
  fgyre =  sin(2y)   (D=1,  E=2)     four gyre
  jet   =  piecewise                 tilted jet

## Params (reconciled with your working other-solver config)
N=256, Lx=Ly=2pi, dt=1e-5, T=100 (=1e7 steps), save every 50000,
mu=0.1, nu=5e-3, B(beta)=5, nv=1, penalty=1.25, start from rest (energy=0).

## READ BEFORE RUNNING
- COMPUTE: T=100 at dt=1e-5 is 1e7 steps PER run, x15. Do a T: 1 smoke test
  on one config first (~1e5 steps), confirm it looks right, then scale up.
- FORCING #1 is -cos(y/2), NOT literal cos(y). cos(y) over a 2pi box has two
  sign changes (not a clean double gyre); cos(y/2) has one. If you truly want
  cos(y), set E: 1.0 and D: 1.0 in the dgyre files.
- SPONGE WIDTH: I used 2.5% (your mask spec). Your WORKING sponge used 10%
  (x_left=0.1). If the 2.5% sponge is too reflective, set width: 0.1 in the
  sponge-walls bc (and matching gap). Brinkman wall width stays 2.5%.
- STABILITY: keep penalty > 1 (eta = penalty*dt). If bc4 blows up (Brinkman
  next to sponge), raise its penalty and/or sponge value first.
- Case 5 alternative with ZERO new code: set friction: 0.0 (no free_slip key)
  to use the existing thin-shell friction-slip instead of the new free_slip.
