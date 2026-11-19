# Icebreaker Lab

A fast, top-down simulation of a ship breaking sea ice, in two versions:

- **A browser page** ([live page](https://pierrelucbacon.com/icebreaker-lab/)): a 76 m icebreaker on a voyage through the ice and traffic of the St. Lawrence. Zoom out and you see the whole Gulf; zoom in on the ship and you see its hull break the ice, with a live trace of the ice force. Click the water to send the ship somewhere, or steer with A/D and W/S when you are close.
- **A JAX environment** (`icebreaker_jax.py`): the continuous-sheet physics with fixed array shapes, so `reset` and `step` jit-compile and vmap over thousands of environments.

## Two scales

| | Hull scale | Route scale |
|---|---|---|
| Code | `icebreaker-core.js`, `icebreaker_jax.py` | `stlnav/` |
| Domain | 600 m × 240 m, 0.5 m grid | Estuary and Gulf of St. Lawrence, 2 km grid |
| Step | 0.2 s (physics at 360 Hz) | 60 s |
| Ice | A sheet that the hull crushes and bends | Concentration and thickness fields from the Canadian ice-ocean forecast (CIOPS-East) or ice charts |
| Ship | 3 degrees of freedom, forces on the waterline | A point with a heading; speed from a force balance fitted to the hull-scale runs (icebreaker) or from Lindqvist's formula (other ships) |
| Other ships | none | Generated traffic on real routes |
| Action | [throttle, rudder]; throttle 1 is full ahead, negative is astern | [throttle, turn]; the same throttle, with negative values counted as 0 |

The route-scale environment is described in [docs/st-lawrence-plan.md](docs/st-lawrence-plan.md) (section "What is built"), with what is sourced and what is a placeholder. Neither scale has been compared with measurements, so the transit times it gives are model output. To check and run it:

```bash
uv run --no-project --python 3.12 --with jax --with numpy python checks/stlnav_checks.py data/fields_20261117.npz
```

```bash
uv run --no-project --python 3.12 --with jax --with xarray --with 'netCDF4<1.7.5' --with scipy --with matplotlib python -m stlnav.demo --cache data/fields_20261117.npz --ice data/ice_hist/sic_2026-02-15.nc
```

`data/` is not in the repository: the fields come from the archive that `archiver/stl_archive.py` builds on the Mila cluster.

### How the two scales are linked

![The two scales](docs/two-scales.png)

*a. Speed the icebreaker holds in level ice (an unbroken sheet of one thickness) at three throttles. Points: hull-scale runs, mean of 3, with bars of one standard deviation that are mostly smaller than the markers; open markers at zero are runs where the ship stopped. Solid lines: the route-scale model, fitted to those runs. Dashed lines: Lindqvist's formula alone, which the route-scale model used before. b. The same voyage at route scale with and without the fit; the ice is a chart of concentration with one assumed thickness. c. That voyage on the map. d. The hull-scale simulator run in the ice of the marked place.*

`stlnav/hullscale.py` links the two simulators in both directions.

**From hull scale to route scale.** The hull-scale ship is run in level ice at 7 thicknesses and 8 throttles, 3 runs each (panel a). More runs find the throttle below which it stops, and its speed in the channel left by another icebreaker. These runs are reduced to a table, `stlnav/tables/icebreaker.json`, that the route-scale speed model reads:

| Ice thickness (m) | 0.2 | 0.3 | 0.45 | 0.6 | 0.8 | 1.0 | 1.2 |
|---|---|---|---|---|---|---|---|
| Ice resistance at 4 m/s, as a multiple of Lindqvist's formula | 0.75 | 0.79 | 0.86 | 0.89 | 0.86 | 0.90 | 0.91 |
| Lowest throttle at which the ship keeps moving | 0.04 | 0.08 | 0.15 | 0.22 | 0.34 | 0.52 | 0.82 |
| The same according to Lindqvist's formula | 0.04 | 0.06 | 0.10 | 0.14 | 0.21 | 0.29 | 0.38 |
| Resistance in another icebreaker's channel, as a share of unbroken ice | 0.79 | 0.73 | 0.65 | 0.63 | 0.57 | 0.56 | 0.52 |

Three things come out of the hull-scale runs that the formula did not give:

- At 4 m/s the moving ship meets 75 to 91% of the resistance in the formula, so it is faster. The multiple also varies with speed (in thick ice it reaches 1.0 near the speed at which the ship stalls), and the table stores it for every thickness and speed.
- In thick ice the ship stops at a much higher throttle than the formula says: 0.82 against 0.38 in 1.2 m of ice. The mean resistance does not explain this. A ship that moves slowly is brought to rest by a single load peak and then cannot break the sheet from rest.
- In the channel of another icebreaker the resistance is 52 to 79% of that in unbroken ice. The route-scale model used a placeholder of 40%, so convoys are slower than before.

How well the route-scale model then agrees with the hull-scale simulator (`checks/two_scales.py`, 10 checks, all passing):

| Case | Fitted model | Lindqvist's formula alone |
|---|---|---|
| The 46 thickness and throttle pairs it was fitted to: mean and largest difference in speed | 0.2%, 2.3% | 7.0%, 17.7% |
| New runs between those pairs, ice of 0.25 and 0.5 m | 0.9%, 1.7% | 4.7%, 6.2% |
| New runs between those pairs, ice of 0.7, 0.9 and 1.1 m | 5.6%, 9.0% | 8.5%, 15.5% |
| Of 168 runs, 27 stop: runs where route scale disagrees | 3 | 24 |
| Speed in another icebreaker's channel, new runs | 1.2%, 2.6% | |
| Flexural strengths of 0.40 and 0.70 MPa (the fit is at 0.55) | 0.7%, 2.2% | |

In ice of 0.7 m and more, the differences of 5 to 9% at new points are not noise: in that check the two hull-scale runs of each case differ by 2.1% at most. The hull-scale resistance varies with thickness in ways that a table of 7 thicknesses does not follow. A finer table needs more runs, about 3 minutes of laptop time per added thickness.

**From route scale to hull scale.** `zoom` reads the ice at a place and time of a route-scale world and runs the hull-scale simulator in level ice of that thickness (panel d):

```python
from stlnav import hullscale
z = hullscale.zoom(env, world, lon, lat, t, thr=1.0)     # env, world: a stlnav environment whose class 0 is the icebreaker
z["hull_speed"], z["route_speed"], z["usable"]            # the two speeds for the same case, and whether the hull-scale run represents the local ice
```

**What the link does not cover.**

- Concentration. The hull-scale JAX simulator has no floes, so it is a continuous sheet or nothing. The route-scale model still scales resistance by concentration squared, which is a placeholder, and `zoom` marks places under 90% concentration as not usable.
- The other ship classes. The hull-scale simulator has one hull, so the merchant ships keep Lindqvist's formula. They use the icebreaker's channel values, which is optimistic for ships wider than the channel.
- Thickness that varies within a 2 km cell, ridges, and wind at hull scale.
- How fast the speed responds. The hull-scale ship's mass and force balance give a response time of about 13 s (computed, not measured); the route-scale model uses a time constant of 120 s.

To remake the table (about 25 minutes on a laptop), check the link (about 6 minutes) and redraw the figure:

```bash
uv run --no-project --python 3.12 --with jax --with numpy python -m stlnav.hullscale
```

```bash
uv run --no-project --python 3.12 --with jax --with numpy python checks/two_scales.py data/fields_20261117.npz
```

```bash
uv run --no-project --python 3.12 --with jax --with xarray --with 'netCDF4<1.7.5' --with scipy --with matplotlib python -m stlnav.demo_scales
```

The table records the hull-scale constants it was measured with. If one of them changes, importing `stlnav` warns that the table is out of date and the first check fails.

### In the browser

![Four zooms of the same voyage](docs/st-lawrence-view.jpg)

*The same voyage at four zooms. Top left: the Gulf, at 41 minutes per second. Top right: 3 m per pixel, where the ice-breaking simulator has taken over and the ice around the ship is made from the chart. Bottom left: the hull in level ice, in real time. Bottom right: further along, where the chart shows 80% ice and the ship pushes through floes.*

The page is one view of one ship on one clock. The camera sits on the map and zooms from the whole Gulf (1 km per pixel) down to the hull (a few centimetres per pixel). Which simulator moves the ship follows the zoom:

| | Far out | Close to the ship (under 10 m per pixel) |
|---|---|---|
| What moves the ship | the route-scale model, `stlnav-core.js` | the ice-breaking simulator, `icebreaker-core.js` |
| Clock | 30 times real time up to 2 hours per second, faster the further out | 1 to 4 times real time |
| Ice | the chart | level ice or floes made from the chart at the ship's place |

A slider and two buttons, "Zoom to the hull" and "Whole voyage", move between the two ends; the page says how much closer the view has to be for the ice to break.

The clock has to follow the zoom. Breaking ice is computed 360 times per second of the ship's time, which a browser manages at about the speed of a clock, and a voyage lasts 11 hours. Far out, the ship therefore moves with the model that the section above fits to the ice-breaking computation, so its speed does not jump when the two swap: in one run, the model had the ship at 7.48 m/s in 90% ice of 0.30 m, and the ice-breaking simulator held 7.46 to 7.49 m/s after taking over.

**Endless ice around the ship.** The ice-breaking simulator used to hold one fixed sheet 1,100 m long. For the page it has a new scenario, `field`: the water is cut into squares of 256 m, each made when the ship comes near and dropped when it is far away. A square holds one sheet of level ice where the chart shows 90% or more, floes that cover the chart's share of the water below that, or nothing. The ship starts inside the ice with a channel astern. `checks/field.js` (4 checks, all passing) compares it with the fixed sheet: 7.55 against 7.53 m/s in 0.3 m of level ice, 5.94 against 5.92 m/s in 0.6 m.

What is real and what is invented close to the ship: the share of ice and its (assumed) thickness come from the chart, cell by cell. The shapes of the floes are invented, because the chart only gives a share per kilometre. The coast is a smooth outline from 1 km data. Other ships are drawn but do not break ice.

**The route-scale model in JavaScript.** `stlnav-core.js` is a port of `stlnav/ships.py`, `traffic.py` and `env.py`. It copies no number by hand: the fields, routes, ship classes, hull-scale table and settings all come from `stl-data.json` (810 kB, 180 kB compressed), which `stlnav/export_web.py` writes. The fields are reduced for the page: currents every 6 km and wind every 16 km, in 9 frames 3 hours apart, and the day repeats.

The map is not drawn from the model's 2 km cells, which look like squares on a screen. `export_web.py` also writes `stl-map.png` (18 kB), the same chart on its own grid of about 1 km, and the page draws the ice from it with a smoothing of the edges between ice classes. The land is not drawn from cells at all: the export traces the edge of the chart's land as 113 closed lines (4,700 points, in `stl-data.json`), and the page fills them as a shape, which has a sharp edge at any zoom. That coast is still only as detailed as the chart, about 1 km. All of this is only drawn: the model works on 2 km cells, so within a cell of the coast the ship can be stopped by land that the picture shows as water.

`checks/stlnav_js.py` runs the Python model on the page's data and compares the port with it (8 checks, all passing):

| Case | Largest difference between the two |
|---|---|
| Speed a ship can hold, 1,200 random cases | 0.000001 m/s |
| Own ship alone, 10 h under the same actions | 4 m in position |
| Own ship among 25 ships, 6 h | 4 m in position; 1 m along a ship's route |
| The three collision rules | 2 m in closest approach |
| Convoy in 0.35 m of ice | none in the speeds, to 0.01 m/s |

Random departures are switched off in those comparisons, because the two random generators differ. The generator is checked on the mean number of ships at sea: 26.4 simulated against 25.4 expected.

The page steers with a small pilot of its own; the Python package has none, because there the policy is what is to be learned. The pilot follows the route 1.5 km to its right and turns to starboard when another ship will pass within 1.2 km. Over 60 test voyages it arrived 59 times and collided once; without that rule it arrived 36 times.

To rebuild the data file after a change to the routes, the classes or the table, run the checks of the page's two simulators, and look at the page locally (then open http://localhost:8000):

```bash
uv run --no-project --python 3.12 --with jax --with xarray --with 'netCDF4<1.7.5' --with scipy --with matplotlib python -m stlnav.export_web
```

```bash
uv run --no-project --python 3.12 --with jax --with numpy python checks/stlnav_js.py
```

```bash
node checks/field.js
```

```bash
python3 -m http.server 8000
```

The ice chart is from Fisheries and Oceans Canada (Daily Ocean Variables from Satellite, through the St. Lawrence Global Observatory), under the Open Government Licence - Canada. The currents and wind are from Environment and Climate Change Canada's MSC Datamart.

## How the ice breaks

The hull waterline is sampled every 0.5 m and tested against the ice, which is stored as a signed-distance field on a 0.5 m grid. Where they overlap, the ice edge is crushed at a fixed pressure on the sloping hull face, which pushes the edge down as well as back. When the downward force on a contact zone exceeds the bearing capacity of the ice wedge,

$$P_f = C_f \left(\frac{\theta}{\pi}\right)^2 \sigma_f h^2,$$

a cusp of radius $R = C_l\, \ell\, (1 + C_v v_n)$ breaks off. Here $h$ is the ice thickness, $\sigma_f$ its flexural strength, $\theta$ the opening angle of the ice wedge, $v_n$ the normal speed at the contact, and $\ell$ the characteristic length of a floating plate. This is the model of Su, Riska and Moan (2010, *Cold Regions Science and Technology* 60:177–188). In the browser version, a finite floe also splits when the in-plane force exceeds $h K_{Ic} \sqrt{L}$, and broken pieces are rigid bodies.

## Status

| | Browser (`icebreaker-core.js`) | JAX (`icebreaker_jax.py`) |
|---|---|---|
| Continuous ice sheet | yes | yes |
| Endless ice around the ship (`field`) | yes | no |
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
