# Icebreaker Lab

A fast, top-down simulation of a ship breaking sea ice, in two versions:

- **A browser simulator** ([live page](https://pierrelucbacon.com/icebreaker-lab/)): a 76 m icebreaker in a continuous ice sheet, large floes or broken pack, with a live trace of the ice force on the hull. Click the ice to set a course, or steer with A/D and W/S.
- **A JAX environment** (`icebreaker_jax.py`): the continuous-sheet physics with fixed array shapes, so `reset` and `step` jit-compile and vmap over thousands of environments.

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

**Against Lindqvist's formula** (an empirical fit to Baltic icebreaker trials), the browser version's ice resistance in a continuous sheet is 14 to 29% lower at 0.3 to 1.2 m of ice (`node checks/resistance_vs_lindqvist.js`).

**JAX against the browser version**, uniform ice, 8 runs per thickness (`checks/compare_js.js`, `checks/compare_jax.py`): speed agrees within 2% and resistance within 3% at 0.6 to 1.2 m; at 0.3 m the JAX resistance is 9% higher.

**JAX throughput** on one Quadro RTX 8000 (`bench_icebreaker_jax.py`, `bench_icebreaker_mila.sh`), in environment steps per second, one step being 0.2 s of simulated time:

| Batch | Physics at 360 Hz | Physics at 180 Hz |
|---|---|---|
| 64 | 7,471 | 14,309 |
| 256 | 11,386 | 22,133 |
| 1024 | 14,256 | 27,912 |
| 4096 | 9,708 | 19,217 |

At 180 Hz (`Config(dt=1/180, inner=3)`), speed stays within 1.5% and resistance within 6% of the 360 Hz results.

## Caveats

- The empirical constants $C_f = 3.1$, $C_l = 0.32$ and $C_v = -0.14$ are recalled from Su et al. and not yet checked against the paper. The splitting coefficient is an estimate.
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
