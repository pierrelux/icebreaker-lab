# Icebreaker Lab

A fast, top-down simulation of a ship breaking sea ice, in two versions:

- **A browser simulator** ([live page](https://pierrelucbacon.com/icebreaker-lab/)): a 76 m icebreaker in a continuous ice sheet, large floes or broken pack, with a live trace of the ice force on the hull. Click the ice to set a course, or steer with A/D and W/S.
- **A JAX environment** (`icebreaker_jax.py`): the continuous-sheet physics with fixed array shapes, so `reset` and `step` jit-compile and vmap over thousands of environments.

## Two scales

| | Hull scale | Route scale |
|---|---|---|
| Code | `icebreaker-core.js`, `icebreaker_jax.py` | `stlnav/` |
| Domain | 600 m × 240 m, 0.5 m grid | Estuary and Gulf of St. Lawrence, 2 km grid |
| Step | 0.2 s (physics at 360 Hz) | 60 s |
| Ice | A sheet that the hull crushes and bends | Concentration and thickness fields from the Canadian ice-ocean forecast (CIOPS-East) or ice charts |
| Ship | 3 degrees of freedom, forces on the waterline | A point with a heading; speed from a resistance formula |
| Other ships | none | Generated traffic on real routes |

The route-scale environment is described in [docs/st-lawrence-plan.md](docs/st-lawrence-plan.md) (section "What is built"), with what is sourced and what is a placeholder. Its speed model is not validated: transit times it gives are model output. To check and run it:

```bash
uv run --no-project --python 3.12 --with jax --with numpy python checks/stlnav_checks.py data/fields_20261117.npz
```

```bash
uv run --no-project --python 3.12 --with jax --with xarray --with 'netCDF4<1.7.5' --with scipy --with matplotlib python -m stlnav.demo --cache data/fields_20261117.npz --ice data/ice_hist/sic_2026-02-15.nc
```

`data/` is not in the repository: the fields come from the archive that `archiver/stl_archive.py` builds on the Mila cluster.

## How the ice breaks

The hull waterline is sampled every 0.5 m and tested against the ice, which is stored as a signed-distance field on a 0.5 m grid. Where they overlap, the ice edge is crushed at a fixed pressure on the sloping hull face, which pushes the edge down as well as back. When the downward force on a contact zone exceeds the bearing capacity of the ice wedge,

$$P_f = C_f \left(\frac{\theta}{\pi}\right)^2 \sigma_f h^2,$$

a cusp of radius $R = C_l\, \ell\, (1 + C_v v_n)$ breaks off. Here $h$ is the ice thickness, $\sigma_f$ its flexural strength, $\theta$ the opening angle of the ice wedge, $v_n$ the normal speed at the contact, and $\ell$ the characteristic length of a floating plate. This is the model of Su, Riska and Moan (2010, *Cold Regions Science and Technology* 60:177–188). In the browser version, a finite floe also splits when the in-plane force exceeds $h K_{Ic} \sqrt{L}$, and broken pieces are rigid bodies.

## Status

| | Browser (`icebreaker-core.js`) | JAX (`icebreaker_jax.py`) |
|---|---|---|
| Continuous ice sheet | yes | yes |
| Separate floes, floe splitting | yes | no |
| Broken pieces | rigid polygons | a grid of broken ice; its share under the hull scales the submersion resistance |

**Against Lindqvist's formula** (an empirical fit to Baltic icebreaker trials), the browser version's ice resistance in a continuous sheet is 6 to 12% lower at 0.6 to 1.2 m of ice and 34% lower at 0.3 m (`node checks/resistance_vs_lindqvist.js`).

**JAX against the browser version**, uniform ice, 8 runs per thickness (`checks/compare_js.js`, `checks/compare_jax.py`): speed agrees within 2% and resistance within 1% at 0.6 to 1.2 m; at 0.3 m the JAX resistance is 22% higher.

**JAX throughput** on one Quadro RTX 8000 (`bench_icebreaker_jax.py`, `bench_icebreaker_mila.sh`), in environment steps per second, one step being 0.2 s of simulated time:

| Batch | Physics at 360 Hz | Physics at 180 Hz |
|---|---|---|
| 64 | 7,471 | 14,309 |
| 256 | 11,386 | 22,133 |
| 1024 | 14,256 | 27,912 |
| 4096 | 9,708 | 19,217 |

At 180 Hz (`Config(dt=1/180, inner=3)`), speed stays within 1.5% and resistance within 6% of the 360 Hz results. These throughput figures were measured before $C_l$ and the ice density were corrected to Su et al.'s values (below).

## Constants

The ice properties are those of Su et al. (2010), Table 1: density 880 kg/m³, Young's modulus 5.4 GPa, Poisson ratio 0.33, crushing strength 2.3 MPa, flexural strength 0.55 MPa, ice–hull friction 0.15. The paper chooses $C_f = 3.1$ by matching Lindqvist's resistance (Figs. 9 and 10). It gives no number for $C_l$ or $C_v$, only their signs; $C_l = 0.30$ is the value paired with $C_f = 3.1$ in Su's PhD thesis (NTNU 2011, Fig. 2.22), fitted to the Tor Viking II h–v curve. $C_v = -0.14$ is attributed to Wang (2001) by secondary sources; we have not checked it against Wang's thesis.

## Caveats

- $C_v$ is unverified (see above). The splitting coefficient is an estimate.
- Nothing has been compared with full-scale or model-basin measurements.
- The model is planar: no pitch or heave, and ice pushed under the hull is a resistance term, not simulated in depth.

## JAX usage

```python
import jax
from icebreaker_jax import Icebreaker, Config, EnvParams

env = Icebreaker(Config())
params = EnvParams.default()                     # thickness, strength, thickness variation, ridges
obs, state = env.reset(jax.random.key(0), params)
obs, state, reward, done, info = env.step(state, jax.numpy.array([1.0, 0.0]), params)   # [throttle, rudder]
```

Each environment holds about 3 MB of state with the default 600 m × 240 m domain.
