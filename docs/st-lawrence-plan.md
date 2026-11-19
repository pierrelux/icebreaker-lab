# Icebreaker Lab, St. Lawrence version: data and plan

Draft of 2026-11-18, updated the same evening (sections marked *update*), for the OGSL collaboration (Mitacs Accélération, "Planification de trajectoires de navires dans les glaces du Saint-Laurent").

## What the simulator is for in the project

The Mitacs proposal needs two things that a simulator can supply.

1. **Task 1.2, the ship performance model.** The route planner needs the speed a given ship can reach, as a function of ice (concentration, thickness, floe size, ridges, pressure) and of wind and waves. The simulator can generate these speed curves for each ship class and each ice regime. The planner then uses a cheap fitted surrogate rather than the simulator itself.
2. **Task 3.2, local navigation in broken ice.** This means comparing a lattice planner (AUTO-IceNav), MPPI and RL policies on ice fields. Those fields are drawn from real St. Lawrence ice situations rather than random ones.

So the realistic version needs three things:
- **Real ice and environment states as input:** concentration, stage of development, floe sizes, drift, currents, wind and waves.
- **Physics for broken ice:** physics for the ice actually found in the Gulf and estuary, which is mostly broken, mobile and compressed by tides, rather than level ice.
- **Validation against observed ship speeds.**

## Data that exists

The table lists what can be imported. **V** means a subagent confirmed it on the official page or live service on 2026-11-18. **P** means partly confirmed, and **C** means cited only. I have not checked these entries myself.

### Ice

| Source | What | Resolution / archive | Access | Status |
|---|---|---|---|---|
| CIS daily Gulf chart (WIS28) | Egg code: concentrations, stage of development, floe size | Polygons, daily Dec–May; PDF/GIF 2011– | https://ice-glaces.ec.gc.ca/prods/WIS28/ | V |
| CIS SIGRID-3 shapefiles | Same egg code, machine-readable | Rolling ~2 weeks only | https://ice-glaces.ec.gc.ca/prods/sigrids/ | V |
| CIS archive, weekly East Coast regional charts | Egg code | E00 1968–2020, ZIP 2020– | https://iceweb1.cis.ec.gc.ca/Archive/ | V |
| CIS image analyses | Concentration and stage from SAR | ~100 m, daily | in /prods/sigrids (SGRDI*) | V |
| DFO/OGSL daily ice concentration (CIS charts gridded) | Concentration | ~1 km, 1997–2026 | ERDDAP `mpoDailySatelliteDataSic` on erddap.ogsl.ca | V |
| CIOPS-East (NEMO-CICE) | Ice concentration, volume (thickness), drift, internal pressure, strength, divergence, shear; currents, sea level | ~2 km, hourly, 48 h forecast | dd.weather.gc.ca/today/model_ciops/east/2km/, GeoMet | V |
| RIOPS | Same ice fields | ~5 km grid, 84 h forecast | dd.weather.gc.ca/today/model_riops/ | V |
| RCM SAR | Backscatter | Public accounts get 100 m over sea; finer products need a vetted account (open to Canadian professors, after security screening). Imaged the Gulf on 22–23 days of February 2026. The AWS `rcm-ceos-ard` set is land only. | EODMS | V (whether the Gulf scenes download on a public account: not checked) |
| Sentinel-1 SAR | Backscatter | IW GRDH, 10 m pixels, 1.7 GB a scene, 2014–. Winter 2025–26: central Gulf and Cabot Strait every 2 days (longest gap 5–7), estuary at Rimouski every 7 days (longest gap 12) | Copernicus Data Space, ASF | V (measured from the catalogue) |
| AMSR2 ASI (Bremen) | Concentration | 3–6 km daily, 2012– | seaice.uni-bremen.de | C |
| CIS climate normals 1991–2020 | Median concentration, ice type, freeze-up and break-up dates | Maps | canada.ca (ice climate normals, East Coast) | V |
| Galbraith et al. (DFO, annual) | Gulf ice volume and thickness indices | Annual | publications.gc.ca | C |

### Navigation environment

| Source | What | Resolution / archive | Access | Status |
|---|---|---|---|---|
| CHS IWLS API | Water level observations, predictions, forecasts, SPINE forecasts, some currents | Stations, 1 min–hourly, decades | api-iwls.dfo-mpo.gc.ca | V |
| ECCC OHPS / SLFE | Water level and depth-averaged current, Carillon to Saint-Joseph-de-la-Rive | ~100 m, 48 h | Datamart model_ohps/slfe, ERDDAP | V |
| DFO STLE200 | Surface currents, Trois-Rivières to Les Escoumins | ~200 m hourly, 48 h | ERDDAP `mpoStle200Nautilo` | V |
| HRDPS / RDPS | Wind and other weather | 2.5 / 10 km hourly | Datamart, GeoMet, ERDDAP `ecccHrdpsNautilo` | P |
| RDWPS, Gulf domain | Significant wave height, peak period and direction; ice-damped | ~2.5 km hourly, 48 h | ERDDAP `ecccRdwpsNautilo`, Datamart | V |
| CHS NONNA bathymetry | Depth | 10 m / 100 m | open.canada.ca | P |
| CCG channel conditions | Maintained depths by sector, shoals | Daily | e-navigation.canada.ca | P |
| NOTMAR 27C | Under-keel clearance rules, Québec–Montréal | Tables | e-navigation.canada.ca | P |
| CHS S-57 ENCs | Channel limits, aids, dredged areas | Vector | Paid licence (Direct User Licence) | P |
| DFO vessel density maps, Northwest Atlantic | Yearly by class (cargo, tanker, passenger, fishing, other) and monthly for all vessels, from AIS | GeoTIFF, one dataset a year 2013–2025; cell size not checked | open.canada.ca | V |
| Global Fishing Watch | Vessel presence from AIS (2012–) and Sentinel-1 detections (2017–) | Gridded, non-commercial licence, free token | 4Wings API | V (Gulf winter content not checked) |
| CCG recommended ice routes | Waypoints of the route the ice centre recommends, daily or as needed in winter (about 30 in 2026, 7 in 2025) | Waypoint files | e-navigation.canada.ca | V |
| Marine park and right-whale zones | Voluntary 10 kn zones, avoidance areas; mandatory Gulf right-whale zones | Text; geometries not found | Parks Canada, TC SSB 02/2026 | P |
| Ice routing and control zones | Recommended routes, zones U–Z, ice operations | Text and maps | CCG *Ice Navigation in Canadian Waters*; e-navigation portal | V |

### Three facts that shape the plan

1. **None of the forecast feeds keeps an archive.** The MSC Datamart keeps about 30 days, and the Nautilo ERDDAP copies keep 48 h. Older model output costs $118/h by request from ECCC. A multi-winter hindcast therefore means archiving it ourselves, starting before this ice season (December), or getting archives from ECCC, DFO-IML or OGSL.
2. **Ice thickness is poorly observed.** No satellite thickness product has been validated for thin, mobile Gulf ice. In practice thickness comes from the stage of development on the CIS charts:
   - grey: 10–15 cm
   - grey-white: 15–30 cm
   - thin first-year: 30–70 cm
   - medium first-year: 70–120 cm

   The other source is the CIOPS-E ice volume field. Pressure exists only in the models.
3. *(Superseded by the update below: AIS is no longer on the critical path.)* **No published study fits AIS ship speed to ice conditions in the Gulf of St. Lawrence.** This was done for the Baltic by Löptien & Axell (2014, *The Cryosphere*). Doing it for the Gulf with CIS charts would be the validation for this project and a publishable result in itself.

## What the current simulator lacks

| Gulf reality | Current simulator | Upgrade |
|---|---|---|
| Mostly broken and pack ice, 10/10 in places, with floe sizes from the egg code | Level ice. Floes and pack only in the browser version, with a hand-made floe field | Generate floe fields from egg-code polygons (concentration, thickness from stage, floe-size class). Port rigid floes to JAX. |
| Ice drifts with wind, tidal currents and currents | Static ice | Free drift: ice velocity ≈ 2–3% of the 10 m wind, turned 20–30°, plus the surface current from CIOPS-E or STLE200. Reduce the drift at high concentration. |
| Compressive ice, which besets ships | None | Pressure from CIOPS-E. Extra resistance from Lu, Kujala et al. (2021, 2023), plus a probability of getting beset. |
| Ridges and rafting | None | Ridge density and equivalent thickness, as in Kotovirta et al. (2009) and Kuuliala et al. (2017). |
| Brash in tracks | Brash grid scaling the Lindqvist submersion term | Brash-channel resistance from the Finnish-Swedish ice class rules (Riska; Juva & Riska 2002). |
| Spring ice weaker than winter ice | Fixed σf = 0.55 MPa | σf from brine volume (Timco & O'Brien 1994), with parameters from the NRC first-year ice report (doi:10.4224/40003139). |
| Many ship types: plaisanciers, merchant ships, cruise ships, CCG icebreakers | One 76 m icebreaker | Ship classes built from particulars (Equasis or the Nautilo user profile). Holtrop-Mennen calm-water resistance, thrust from installed power. |
| Open-water speed loss from wind and waves | None | Kwon's speed-loss method, or STAWAVE-2 / ISO 15016 added resistance. |
| Narrow channel, under-keel clearance, speed zones | Open rectangle | Navigable mask from NONNA + channel depths + NOTMAR 27C. Zones as polygon constraints. |
| Validated only against Lindqvist (known to be off by ~20% in thick ice) | — | AIS + CIS validation (below), plus published full-scale trials (Tor Viking II, CCGS Henry Larsen). |

## Plan

The phases follow the Mitacs timeline. Phase 0 should start now regardless of the proposal.

### Phase 0: start archiving (this week, before the ice season)

- A daily job on the Mila cluster, or a cheap always-on machine (not a laptop or a Claude session). It saves:
  - CIOPS-East ice and current fields over the Gulf and estuary
  - HRDPS 10 m wind
  - RDWPS waves
  - the CIS SIGRID-3 daily and image-analysis shapefiles
  - IWLS SPINE forecasts and observations
- Cropped to the domain and stored as NetCDF/Zarr: tens of GB per winter.
- Ask OGSL and ECCC whether an archive of these already exists. If it does, this job only fills the gap.

### Phase 1: a common environment layer (weeks 1–8, task 1.1)

- A Python package that turns every source into one queryable field env(x, y, t) on a common grid along the navigable routes:
  - ice concentration, stage, floe size, thickness, drift and pressure
  - current, wind, waves and water level
  - depth and UKC
  - zones
- Backed by xarray and Zarr, with a JAX-friendly export: fixed-shape arrays for a time window.
- Historical ice: CIS weekly regional archive (1968–), the DFO 1 km daily concentration (1997–2026), and the Phase 0 archive.

### Phase 2: realistic ice in the simulator (weeks 5–14, task 1.2)

1. **Scenario generator.** Turns an egg-code polygon or a CIOPS-E grid cell into a simulated ice field, with seeded randomness. Concentration and floe-size class set the floe field, and stage sets the thickness distribution.
2. **Rigid floes in JAX**, a fixed-size floe array. This adds the pack ice and broken ice the Gulf actually has. The Su bending model stays for breaking floes and level ice.
3. **Free drift** from wind, current and tide, with floe–floe contact as in the browser version.
4. **Formula-based resistance terms** where full simulation is too costly or not needed:
   - brash channel (Finnish-Swedish ice class rules)
   - ridge equivalent thickness
   - compressive-ice term

   These need calibrating against the floe simulator and published model tests (KRISO pack-ice tests; the UCL floe-field equation).
5. **Ship classes:** particulars, calm-water resistance (Holtrop-Mennen), thrust from installed power, and wind and wave speed loss (Kwon or STAWAVE-2).

Every coefficient taken from memory or a secondary source gets checked against its primary paper before use, as was done today for the Su constants.

### Phase 3: from simulator to route-planning cost (weeks 9–20)

- Run the simulator over a grid of ice regimes × ship classes on the GPU, to produce attainable speed and beset probability.
- Fit a small surrogate: speed = f(ship class, concentration, thickness, floe size, ridges, pressure, wind, waves). This is the "modèle de performance du navire" of task 1.2, in the hybrid form of Montewka et al. (2019).
- The planner (time-dependent A*, then MPPI) uses the surrogate. The full simulator is used only for local navigation (task 3.2).

### Phase 4: validation with AIS (weeks 9–34, task 1.3)

*Update: this phase now depends on getting AIS at all, and is a check of orders of magnitude, not a calibration. See the update section.*

- Match AIS tracks to the CIS chart and CIOPS-E state at each position and time, then fit observed speed against ice conditions, following Löptien & Axell (2014). This gives the first such study for the Gulf.
- Compare the surrogate and the simulator against those observed speeds, by ship class and ice regime.
- Secondary checks:
  - published full-scale trials (Tor Viking II through Su et al.; NRC trials of CCGS Henry Larsen)
  - Li et al. (2018) as the template for comparing transit models

### Phase 5: local navigation benchmark (weeks 36–45, task 3.2)

- Ice fields drawn from real CIS polygons and CIOPS-E drift.
- Compare AUTO-IceNav's lattice planner (open source, github.com/rdesc/AUTO-IceNav), MPPI and RL.
- Metrics: transit time, kinetic energy lost to collisions, and the rate of getting beset.

## Update of 2026-11-18, evening

### AIS is off the critical path

- **Access.** No free archive of AIS covers the Gulf. A VHF receiver hears 40 to 80 km, so one in Montréal cannot hear the Gulf. AISHub lists stations at Rimouski, Sorel-Tracy, Montréal and Longueuil only, and gives its feed to members who contribute a receiver. aisstream.io publishes no coverage map.
- **Use.** Even with the data, speed in ice is hard to fit from AIS: the captain chooses the speed, ships avoid heavy ice, and the ice under the ship is not known at ship scale.
- **Decision.** The simulator keeps the published resistance models and reports how sensitive the routes are to their coefficients. Traffic is generated (below). AIS, if OGSL or the Coast Guard can supply it, checks transit times and lane use.
- **Cheap options worth taking:** a receiver hosted at Rimouski (it would also qualify for AISHub), and the DFO monthly density maps to check the generated lanes and volumes.

### Satellite radar

- Sentinel-1 is free, permanent and frequent enough over the Gulf (every 2 days) for ice at 10 to 20 m, less so over the estuary (weekly). Nothing needs archiving.
- It does not replace AIS: a ship is a bright dot every few days, and is hard to tell from ice.
- Two uses to try: floe-scale ice fields for the local-navigation benchmark, and ship tracks left in the ice.

### What is built: `stlnav`, a route-scale environment

`stlnav/` is a second JAX environment, at the scale of a voyage (2 km grid, 60 s steps), next to the hull-scale `icebreaker_jax.py`.

| Part | What it does | Status |
|---|---|---|
| `fields.py` | Reads the archive into one grid: CIOPS-East ice and currents, HRDPS wind turned to east/north, RDWPS wave height. Can lay a past winter's chart concentration over it. | Runs on the archived day of 2026-11-17. CIOPS velocity axes assumed east/north, not checked. |
| `ships.py` | Speed a ship class can hold: thrust against open-water drag, Lindqvist ice resistance scaled by concentration, and wind drag. | Lindqvist matches `icebreaker_jax`. The concentration exponent (2) and the in-track factor (0.4) are placeholders. |
| `traffic.py` | Ships leave ports at given rates or on timetables, follow routes, slow in ice, queue and form convoys behind icebreakers, and apply three collision rules to the own ship. | Checked: volumes, the three encounters, escort. Traffic ships do not avoid each other across routes. |
| `routes.py` | Trunk lane, branches to Sept-Îles, Port-Cartier and Baie-Comeau, four winter ferries, six ship classes. | Sources below. |
| `env.py` | `reset` and `step` with fields, traffic and a point ship; observation of local fields and the 8 nearest ships. | 12,000 steps/s for 256 environments on a laptop CPU. Not yet run on a GPU. |
| `checks/stlnav_checks.py` | 12 checks, all passing. | |
| `demo.py`, `docs/stlnav-demo.png` | Les Escoumins to Sept-Îles on the Coast Guard ice route of 12 February 2026, with the ice chart of 15 February 2026 and an assumed 0.3 m thickness: 11.1 h for the icebreaker, 20.5 h for a Polar Class 7 tanker. | These times are model output, not validated against any observed transit. |

**Traffic sources** (read by a research subagent on 2026-11-18; I did not re-read them):

- Lanes: no separation scheme is published as coordinates. The trunk follows Transport Canada's dynamic right-whale zones (SSB 02/2026), which sit on the lanes south of Anticosti. The ice route is the Coast Guard's of 12 February 2026 (e-navigation.canada.ca).
- Volumes: Laurentian Pilotage Authority annual report 2024 (6,889 voyages Montréal–Les Escoumins; type mix); DFO 2010 (about 6,400 Cabot Strait transits a year, a 2001 figure); port reports for Sept-Îles (575 vessels, 2025), Port-Cartier (409, 2021) and Baie-Comeau (263, 2004), these three from secondary sources. No winter-month counts were found, so the annual means overstate winter traffic.
- Ferries: STQ winter timetable 2027 (Matane–Baie-Comeau 5 round trips a week, Matane–Godbout 3); Marine Atlantic (about 2 a day); CTMA (about 3 a week, unconfirmed).
- Ships: Mia Desgagnés and Blue Puttees from operator sheets; Federal Montreal class and Oceanex Connaigra from trade press. Bow angles are assumed for all of them.
- Not found anywhere: a published speed for a merchant ship in Gulf ice, escorted or not, and any numeric convoy spacing.

**What this version leaves out:** ice pressure and besetting, ridges, ice drift between frames, wave speed loss, depth and under-keel clearance, the river above Les Escoumins (too narrow for a 2 km grid), forecasts as the ship would have had them (the fields are near-analyses), and the link to the hull-scale simulator (the speed model is a formula, not yet a fit to it).

### Next steps

1. Run a winter day from our own archive once ice appears (late December), with CIOPS thickness in place of the assumed one.
2. Fit the speed model to the hull-scale simulator for level ice, and choose the concentration scaling from model tests.
3. Add ice pressure (already archived) and a besetting rule.
4. Archive the Coast Guard's daily ice routes with the rest, and use each day's route for that day's traffic.
5. Compare generated lanes and volumes with the DFO monthly density maps.

## Questions for OGSL

1. Historical AIS: which area, which years, raw positions or processed tracks, and under what licence? Nothing AIS appears in their public catalogue.
2. Do they or their partners (DFO-IML, ECCC) hold archives of CIOPS-East, OHPS, STLE200 and RDWPS? Their Nautilo ERDDAP copies keep 48 h only, and their latest time step was 2026-11-14 when checked, so the feed may be stalled.
3. Which ice layers will Nautilo show? The public ERDDAP mirror of CIOPS-East carries currents only.
4. Geometries for the marine-park voluntary speed zones and the beluga and blue-whale areas (Parks Canada or the working group).
5. Which ship classes the Nautilo user profile records, to define the simulator's classes.
6. Contacts at CCG (St. Lawrence ice operations centre) or the Laurentian Pilotage Authority, for observed transit times in ice.
7. Would OGSL or ISMER host an AIS receiver at Rimouski? At that point the estuary is about 50 km wide, so one antenna covers it.
8. Does the Coast Guard keep logs of escorts and of ice met by its icebreakers, and can they be shared? The ship and the ice are both known there, which AIS cannot give.
9. Winter traffic counts by month past Les Escoumins or through Cabot Strait.

## Main references

- Su, Riska & Moan (2010), *Cold Regions Science and Technology* 60:177–188; Su (2011), NTNU PhD thesis.
- Montewka et al. (2019), hybrid ship performance model for route planning, *Proc. IMechE Part O* 233(1).
- Kotovirta et al. (2009), route optimisation in ice, *Cold Regions Science and Technology* 55(1):52–62.
- Kuuliala et al. (2017), operability in ridged ice, *Cold Regions Science and Technology* 135.
- Löptien & Axell (2014), ice and AIS ship speeds in the Baltic, *The Cryosphere* 8:2409–2418.
- Li et al. (2018), benchmark of ship transit models in ice, *Cold Regions Science and Technology* 151:94–108.
- Juva & Riska (2002), power requirement in the Finnish-Swedish ice class rules, WNRB Report 53.
- Lu, Kujala & Goerlandt (2021), transit in compressive ice, *Ocean Engineering*.
- Paquin et al. (2024), CIOPS-East, *Ocean Dynamics*, doi:10.1007/s10236-024-01634-7.
- Timco & O'Brien (1994), flexural strength of sea ice, *Cold Regions Science and Technology*.
- de Schaetzen et al. (2024), AUTO-IceNav, arXiv:2411.17155.
- Holtrop & Mennen (1982); Kwon (2008); ISO 15016 / STAWAVE-2.
