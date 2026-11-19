// St. Lawrence route-scale model for the browser: a port of stlnav/ships.py, traffic.py and env.py.
//
// One ship (the "own ship") sails through fields of ice, current and wind on a 2 km grid, among generated
// traffic, in steps of 60 s. Its speed in ice comes from a force balance fitted to the hull-scale simulator
// (icebreaker-core.js is the browser version of that simulator).
//
// Every number comes from stl-data.json, written by stlnav/export_web.py: fields, routes, ship classes with
// the hull-scale table, and the settings of the traffic and of the environment. checks/stlnav_js.py compares
// this file with the Python version on the same data.
//
//   const world = StlNav.create(data, { thick: 0.3 });        // data = the parsed stl-data.json
//   const state = StlNav.reset(world, { lon, lat, th, goalLon, goalLat, cls: 0, seed: 1 });
//   const info = StlNav.step(world, state, [throttle, turn]); // throttle in [0, 1], turn in [-1, 1]
const StlNav = (() => {
  'use strict';
  const KY = 111.195, RAD = Math.PI / 180;
  const clip = (x, a, b) => (x < a ? a : x > b ? b : x);
  function bytes(s) {
    if (typeof atob === 'function') { const b = atob(s), a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }
    return new Uint8Array(Buffer.from(s, 'base64'));
  }
  function makeRng(seed) {     // mulberry32, and a normal deviate by Box-Muller
    let a = seed >>> 0;
    const u = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    u.normal = () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
    u.pick = w => { let s = 0; for (let i = 0; i < w.length; i++) s += w[i]; let x = u() * s; for (let i = 0; i < w.length; i++) { x -= w[i]; if (x < 0) return i; } return w.length - 1; };
    return u;
  }

  // ------------------------------------------------------------------ world and fields
  function create(d, opt) {
    const o = Object.assign({ thick: 0.3, sigF: 0.55e6, wrapTime: false }, opt || {}), g = d.grid, raw = bytes(d.ice.data), n = g.nx * g.ny;
    const conc = new Float32Array(n), land = new Uint8Array(n), ind = new Uint8Array(n);
    for (let k = 0; k < n; k++) { if (raw[k] === 255) land[k] = 1; else { conc[k] = raw[k] / d.ice.scale; ind[k] = conc[k] > 0.01 ? 1 : 0; } }
    const coarse = s => ({ step: s.step, scale: s.scale, nx: s.nx, ny: s.ny, u: new Int8Array(bytes(s.u).buffer), v: new Int8Array(bytes(s.v).buffer) });
    const r = d.routes;
    return { g, conc, land, ind, cur: coarse(d.cur), wind: coarse(d.wind), thick: o.thick, sigF: o.sigF, wrapTime: o.wrapTime, K: d.consts, hGrid: d.h_grid, vGrid: d.v_grid,
      classes: d.classes, routes: r, R: r.length.length, cfg: d.env, tc: d.traffic, places: d.places, voyages: d.voyages, about: d.about };
  }

  const F = { conc: 0, thick: 0, uc: 0, vc: 0, uw: 0, vw: 0 };
  function coarseAt(c, fx, fy, k0, k1, at, out, a, b) {   // bilinear in space on two frames, linear in time
    const x = clip(fx / c.step, 0, c.nx - 1), y = clip(fy / c.step, 0, c.ny - 1), i = Math.min(x | 0, c.nx - 2), j = Math.min(y | 0, c.ny - 2), ax = x - i, ay = y - j;
    const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay, q = j * c.nx + i, m = c.nx * c.ny;
    const f = (arr, k) => { const p = k * m + q; return arr[p] * w00 + arr[p + 1] * w10 + arr[p + c.nx] * w01 + arr[p + c.nx + 1] * w11; };
    out[a] = (f(c.u, k0) * (1 - at) + f(c.u, k1) * at) * c.scale; out[b] = (f(c.v, k0) * (1 - at) + f(c.v, k1) * at) * c.scale;
  }
  // Fields at a point and time (seconds after the first frame). Fills and returns `out` (default: a shared object).
  function sample(w, lon, lat, t, out) {
    out = out || F; const g = w.g;
    const fx = clip((lon - g.lon0) / g.dlon, 0, g.nx - 1), fy = clip((lat - g.lat0) / g.dlat, 0, g.ny - 1), i = Math.min(fx | 0, g.nx - 2), j = Math.min(fy | 0, g.ny - 2), ax = fx - i, ay = fy - j, q = j * g.nx + i;
    const bil = a => (a[q] * (1 - ax) + a[q + 1] * ax) * (1 - ay) + (a[q + g.nx] * (1 - ax) + a[q + g.nx + 1] * ax) * ay;
    out.conc = bil(w.conc); out.thick = w.thick * bil(w.ind);
    const span = (g.nt - 1) * g.dt, tt = w.wrapTime ? ((t % span) + span) % span : t, ft = clip(tt / g.dt, 0, g.nt - 1), k0 = Math.min(ft | 0, Math.max(g.nt - 2, 0)), k1 = Math.min(k0 + 1, g.nt - 1), at = ft - k0;
    coarseAt(w.cur, fx, fy, k0, k1, at, out, 'uc', 'vc'); coarseAt(w.wind, fx, fy, k0, k1, at, out, 'uw', 'vw');
    return out;
  }
  function landAt(w, lon, lat) {   // share of land around a point; 1 outside the grid
    const g = w.g; let fx = (lon - g.lon0) / g.dlon, fy = (lat - g.lat0) / g.dlat;
    if (fx < 0 || fy < 0 || fx > g.nx - 1 || fy > g.ny - 1) return 1;
    const i = Math.min(fx | 0, g.nx - 2), j = Math.min(fy | 0, g.ny - 2), ax = fx - i, ay = fy - j, q = j * g.nx + i, a = w.land;
    return (a[q] * (1 - ax) + a[q + 1] * ax) * (1 - ay) + (a[q + g.nx] * (1 - ax) + a[q + g.nx + 1] * ax) * ay;
  }

  // ------------------------------------------------------------------ ships (ships.py)
  function lindqvist(w, c, h, sigF, out) {   // level-ice resistance a + b v
    const K = w.K; h = Math.max(h, 1e-3);
    const phi = c.stem, al = c.entrance, psi = Math.atan(Math.tan(phi) / Math.sin(al));
    const rc = 0.5 * sigF * h * h * (Math.tan(phi) + K.MU * Math.cos(phi) / Math.cos(psi)) / (1 - K.MU * Math.sin(phi) / Math.cos(psi));
    const rb = 27 / 64 * sigF * c.B * Math.pow(h, 1.5) / Math.sqrt(K.E / (12 * (1 - K.NU * K.NU) * K.G * K.RHO_W)) * (Math.tan(psi) + K.MU * Math.cos(phi) / (Math.cos(psi) * Math.sin(al))) * (1 + 1 / Math.cos(psi));
    const k = K.MU * (0.7 * c.L - c.T / Math.tan(phi) - c.B / (4 * Math.tan(al)) + c.T * Math.cos(phi) * Math.cos(psi) * Math.sqrt(1 / Math.pow(Math.sin(phi), 2) + 1 / Math.pow(Math.tan(al), 2)));
    const rs = (K.RHO_W - K.RHO_I) * K.G * h * c.B * (c.T * (c.B + c.T) / (c.B + 2 * c.T) + k);
    out[0] = rc + rb + rs; out[1] = (rc + rb) * 1.4 / Math.sqrt(K.G * h) + rs * 9.4 / Math.sqrt(K.G * c.L);
  }
  const CI = [0, 0];
  function cell(grid, x, out) {   // interval of the grid that holds x, and the weight of its upper end
    let n = 0; while (n < grid.length && grid[n] <= x) n++;
    const i = clip(n - 1, 0, grid.length - 2); out[0] = i; out[1] = clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0, 1);
  }
  function atH(w, curve, h) { cell(w.hGrid, h, CI); return curve[CI[0]] * (1 - CI[1]) + curve[CI[0] + 1] * CI[1]; }
  const CS = [0, 0];
  function atHV(w, table, h, v, out) {   // value and slope in v of a table over thickness and speed
    cell(w.hGrid, h, CI); const i = CI[0], wh = CI[1]; cell(w.vGrid, v, CI); const j = CI[0], wv = CI[1], g = w.vGrid;
    const c0 = table[i][j] * (1 - wh) + table[i + 1][j] * wh, c1 = table[i][j + 1] * (1 - wh) + table[i + 1][j + 1] * wh;
    out[0] = c0 * (1 - wv) + c1 * wv; out[1] = v >= g[0] && v <= g[g.length - 1] ? (c1 - c0) / (g[j + 1] - g[j]) : 0;
  }
  function iceScale(w, c, h, conc, inTrack) {
    if (!(conc > 0.01 && h > 0.01)) return 0;
    const cc = clip(conc, 0, 1); return cc * cc * (1 - (inTrack ? 1 : 0) * (1 - atH(w, c.track, h)));
  }
  const AB = [0, 0];
  // Steady speed through the water (m/s) at throttle thr in [0, 1]; 0 if the ship is stopped by the ice.
  function attainableSpeed(w, c, thr, h, conc, sigF, fExt, inTrack) {
    lindqvist(w, c, h, sigF, AB); const a = AB[0], b = AB[1], scale = iceScale(w, c, h, conc, inTrack);
    const kx = 0.7 * c.TB / (c.uow * c.uow), lin = (0.3 * thr + 0.01) * c.TB / c.uow, push = thr * c.TB - (fExt || 0);
    const top = (-lin + Math.sqrt(lin * lin + 4 * kx * Math.max(push, 0))) / (2 * kx);
    let v = top;
    for (let it = 0; it < 8; it++) {
      atHV(w, c.c, h, v, CS);
      const f = push - lin * v - kx * v * v - scale * CS[0] * (a + b * v), df = -lin - 2 * kx * v - scale * (CS[1] * (a + b * v) + CS[0] * b);
      v = clip(v - f / Math.min(df, -1), 0, top);
    }
    atHV(w, c.c, h, 0, CS);
    return push > a * atH(w, c.stall, h) * scale && push > a * CS[0] * scale ? v : 0;
  }
  function iceResistance(w, c, h, v, conc, sigF, inTrack) { lindqvist(w, c, h, sigF, AB); atHV(w, c.c, h, v, CS); return iceScale(w, c, h, conc, inTrack) * CS[0] * (AB[0] + AB[1] * v); }
  function windForce(w, c, v, heading, uw, vw) { const a = v - (uw * Math.cos(heading) + vw * Math.sin(heading)); return 0.5 * w.K.RHO_A * 0.8 * c.area * a * Math.abs(a); }

  // ------------------------------------------------------------------ traffic (traffic.py)
  function interp(x, xp, fp) {
    const n = xp.length; if (x <= xp[0]) return fp[0]; if (x >= xp[n - 1]) return fp[n - 1];
    let k = 1; while (xp[k] < x) k++;
    return fp[k - 1] + (fp[k] - fp[k - 1]) * (x - xp[k - 1]) / (xp[k] - xp[k - 1]);
  }
  const PL = [0, 0, 0];
  function place(w, r, s, off, out) {   // longitude, latitude and heading (radians from east) at distance s along route r
    out = out || PL; const rt = w.routes, cum = rt.cum[r], lons = rt.lon[r], lats = rt.lat[r];
    const lo = interp(s, cum, lons), la = interp(s, cum, lats), s1 = Math.max(s - 0.3, 0), s2 = Math.min(s + 0.3, rt.length[r]);
    const kx = KY * Math.cos(la * RAD), hd = Math.atan2((interp(s2, cum, lats) - interp(s1, cum, lats)) * KY, (interp(s2, cum, lons) - interp(s1, cum, lons)) * kx);
    out[0] = lo + off * Math.sin(hd) / kx; out[1] = la - off * Math.cos(hd) / KY; out[2] = hd; return out;
  }
  function newTraffic(n) {
    return { n, active: new Uint8Array(n), route: new Int32Array(n), s: new Float64Array(n), off: new Float64Array(n), off0: new Float64Array(n), v: new Float64Array(n),
      vdes: new Float64Array(n), cls: new Int32Array(n), comply: new Uint8Array(n) };
  }
  function spawn(w, ts, rng, r, s0) {
    let slot = -1; for (let i = 0; i < ts.n; i++) if (!ts.active[i]) { slot = i; break; }
    if (slot < 0) return;
    const tc = w.tc, cls = rng.pick(w.routes.mix[r]), vdes = w.classes[cls].uow * (tc.speed_lo + (tc.speed_hi - tc.speed_lo) * rng()), off = w.routes.off[r] + tc.off_spread * rng.normal();
    ts.active[slot] = 1; ts.route[slot] = r; ts.s[slot] = s0; ts.off[slot] = off; ts.off0[slot] = off; ts.v[slot] = vdes; ts.vdes[slot] = vdes; ts.cls[slot] = cls; ts.comply[slot] = rng() < tc.p_comply ? 1 : 0;
  }
  function typical(w) {   // typical speed and transit time of each route
    const rt = w.routes, tc = w.tc, v = [], transit = [];
    for (let r = 0; r < w.R; r++) { let u = 0; for (let c = 0; c < w.classes.length; c++) u += rt.mix[r][c] * w.classes[c].uow; v.push(u * 0.5 * (tc.speed_lo + tc.speed_hi)); transit.push(rt.length[r] * 1e3 / v[r]); }
    return { v, transit };
  }
  // Traffic as it would be at time t if the generator had always been running (open-water speeds).
  function resetTraffic(w, n, t, rng) {
    const ts = newTraffic(n), rt = w.routes, tc = w.tc, ty = typical(w), expect = rt.rate.map((q, r) => q * ty.transit[r]), total = expect.reduce((a, b) => a + b, 0);
    for (let i = 0; i < n; i++) {
      const r = total > 0 ? rng.pick(expect) : 0, cls = rng.pick(rt.mix[r]), off = rt.off[r] + tc.off_spread * rng.normal();
      ts.route[i] = r; ts.cls[i] = cls; ts.vdes[i] = ts.v[i] = w.classes[cls].uow * (tc.speed_lo + (tc.speed_hi - tc.speed_lo) * rng()); ts.off[i] = ts.off0[i] = off;
      ts.s[i] = rng() * rt.length[r]; ts.comply[i] = rng() < tc.p_comply ? 1 : 0; ts.active[i] = total > 0 && rng() < Math.min(total / n, 1) ? 1 : 0;
    }
    for (let r = 0; r < w.R; r++) {   // a timetabled ship that left before t and has not arrived yet
      const per = Math.max(rt.period[r], 1), since = (((t - rt.phase[r]) % per) + per) % per;
      if (rt.period[r] > 0 && since < ty.transit[r]) spawn(w, ts, rng, r, since * ty.v[r] / 1e3);
    }
    return ts;
  }
  const FT = { conc: 0, thick: 0, uc: 0, vc: 0, uw: 0, vw: 0 };
  // Advance the traffic by dt seconds from time t. own = [lon, lat, east speed, north speed] of the own ship over the ground.
  function stepTraffic(w, ts, t, dt, own, sigF, rng, departures) {
    const n = ts.n, tc = w.tc, rt = w.routes, lon = new Float64Array(n), lat = new Float64Array(n), cs = new Float64Array(n), sn = new Float64Array(n);
    const conc = new Float64Array(n), thick = new Float64Array(n), uc = new Float64Array(n), vc = new Float64Array(n), target = new Float64Array(n), offGoal = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      place(w, ts.route[i], ts.s[i], ts.off[i], PL); lon[i] = PL[0]; lat[i] = PL[1]; cs[i] = Math.cos(PL[2]); sn[i] = Math.sin(PL[2]);
      sample(w, lon[i], lat[i], t, FT); conc[i] = FT.conc; thick[i] = FT.thick; uc[i] = FT.uc; vc[i] = FT.vc;
    }
    const g = new Float64Array(n).fill(Infinity), lead = new Int32Array(n), gb = new Float64Array(n).fill(Infinity), behind = new Int32Array(n);
    for (let i = 0; i < n; i++) {   // nearest ship ahead (g, lead) and behind (gb, behind) on the same route
      if (!ts.active[i]) continue;
      for (let j = 0; j < n; j++) {
        if (!ts.active[j] || ts.route[j] !== ts.route[i] || !(ts.s[j] > ts.s[i])) continue;
        const d = ts.s[j] - ts.s[i]; if (d < g[i]) { g[i] = d; lead[i] = j; } if (d < gb[j]) { gb[j] = d; behind[j] = i; }
      }
    }
    for (let i = 0; i < n; i++) {
      const inIce = conc[i] > tc.ice_conc, c = w.classes[ts.cls[i]];
      let tg = Math.min(ts.vdes[i], attainableSpeed(w, c, 1, thick[i], conc[i], sigF, 0, inIce && g[i] < tc.d_channel));
      if (inIce && g[i] < tc.d_follow) tg = Math.min(tg, Math.max(ts.v[lead[i]] + (g[i] - tc.d_keep) * 1e3 / 300, 0));
      if (inIce && ts.cls[i] === tc.escort_cls && gb[i] < tc.d_follow) tg = Math.min(tg, Math.max(ts.v[behind[i]] + (0.5 * tc.d_channel - gb[i]) * 1e3 / 300, 0));
      // collision rules towards the own ship
      const kx = KY * Math.cos(lat[i] * RAD), dx = (own[0] - lon[i]) * kx, dy = (own[1] - lat[i]) * KY, ve = ts.v[i] * cs[i] + uc[i], vn = ts.v[i] * sn[i] + vc[i];
      const rvx = (own[2] - ve) * 1e-3, rvy = (own[3] - vn) * 1e-3, rv2 = rvx * rvx + rvy * rvy + 1e-12, tcpa = -(dx * rvx + dy * rvy) / rv2, dcpa = Math.hypot(dx + rvx * tcpa, dy + rvy * tcpa);
      const risk = ts.active[i] && ts.comply[i] && tcpa > 0 && tcpa < tc.t_risk && dcpa < tc.d_risk;
      const brg = Math.atan2(dx * sn[i] - dy * cs[i], dx * cs[i] + dy * sn[i]) / RAD, ospd = Math.hypot(own[2], own[3]) + 1e-6, same = (own[2] * cs[i] + own[3] * sn[i]) / ospd;
      const obrg = Math.atan2(-(dx * own[3] - dy * own[2]) / ospd, -(dx * own[2] + dy * own[3]) / ospd) / RAD;
      const headOn = Math.abs(brg) < 90 && same < -0.94, overtaking = Math.abs(obrg) > 112.5 && ts.v[i] > ospd && same > 0, crossing = brg >= 10 && brg <= 112.5 && !headOn && !overtaking;
      if (risk && crossing) tg *= tc.slow;
      target[i] = tg; offGoal[i] = risk && (headOn || overtaking) ? ts.off0[i] + tc.sidestep : ts.off0[i];
    }
    const k = 1 - Math.exp(-dt / tc.tau);
    for (let i = 0; i < n; i++) {
      ts.v[i] += (target[i] - ts.v[i]) * k; ts.off[i] += clip(offGoal[i] - ts.off[i], -tc.off_rate * dt, tc.off_rate * dt);
      ts.s[i] += Math.max(ts.v[i] + uc[i] * cs[i] + vc[i] * sn[i], 0) * dt * 1e-3;
      if (!(ts.s[i] < rt.length[ts.route[i]])) ts.active[i] = 0;
    }
    if (departures === false) return;
    for (let r = 0; r < w.R; r++) {
      const per = Math.max(rt.period[r], 1), timetable = rt.period[r] > 0 && Math.floor((t + dt - rt.phase[r]) / per) > Math.floor((t - rt.phase[r]) / per);
      if (timetable || rng() < Math.min(rt.rate[r] * dt, 1)) spawn(w, ts, rng, r, 0);
    }
  }

  // ------------------------------------------------------------------ the own ship (env.py)
  function goal(s) { const dx = (s.goalLon - s.lon) * KY * Math.cos(s.lat * RAD), dy = (s.goalLat - s.lat) * KY; return Math.hypot(dx, dy); }
  // p: { lon, lat, th (optional: towards the goal), goalLon, goalLat, cls, t0, seed, traffic (false = none) }
  function reset(w, p) {
    const rng = makeRng(p.seed === undefined ? 1 : p.seed), t0 = p.t0 || 0, cls = p.cls || 0;
    const th = p.th !== undefined ? p.th : Math.atan2((p.goalLat - p.lat) * KY, (p.goalLon - p.lon) * KY * Math.cos(p.lat * RAD));
    const ts = p.traffic === false ? newTraffic(w.cfg.n_traffic) : resetTraffic(w, w.cfg.n_traffic, t0, rng);
    return { lon: p.lon, lat: p.lat, th, v: 0.8 * w.classes[cls].uow, t: t0, t0, cls, goalLon: p.goalLon, goalLat: p.goalLat, traffic: ts, rng, departures: p.traffic !== false };
  }
  // One step of cfg.dt seconds. action = [throttle in 0..1, turn in -1..1]. Changes `s` and returns what happened.
  function step(w, s, action) {
    const c = w.cfg, tc = w.tc, ts = s.traffic, n = ts.n, own = w.classes[s.cls], dt = c.dt, thr = clip(action[0], 0, 1), turn = clip(action[1], -1, 1);
    const here = sample(w, s.lon, s.lat, s.t, {}), cs = Math.cos(s.th), sn = Math.sin(s.th), kx = KY * Math.cos(s.lat * RAD);
    const rx0 = new Float64Array(n), ry0 = new Float64Array(n), act0 = Uint8Array.from(ts.active), route0 = Int32Array.from(ts.route), s0 = Float64Array.from(ts.s);
    let inTrack = false;
    for (let i = 0; i < n; i++) {
      place(w, ts.route[i], ts.s[i], ts.off[i], PL); rx0[i] = (PL[0] - s.lon) * kx; ry0[i] = (PL[1] - s.lat) * KY;
      const back = rx0[i] * Math.cos(PL[2]) + ry0[i] * Math.sin(PL[2]), side = Math.abs(-rx0[i] * Math.sin(PL[2]) + ry0[i] * Math.cos(PL[2]));
      if (ts.active[i] && back > 0 && back < tc.d_channel && side < tc.w_channel) inTrack = true;
    }
    const fWind = windForce(w, own, s.v, s.th, here.uw, here.vw), vGoal = attainableSpeed(w, own, thr, here.thick, here.conc, w.sigF, fWind, inTrack);
    const v = s.v + (vGoal - s.v) * (1 - Math.exp(-dt / c.tau_v)), th = s.th + turn * own.turn * clip(v / own.uow, 0.2, 1) * dt;
    const ve = v * cs + here.uc, vn = v * sn + here.vc, lon = s.lon + ve * dt * 1e-3 / kx, lat = s.lat + vn * dt * 1e-3 / KY, d0 = goal(s);
    stepTraffic(w, ts, s.t, dt, [s.lon, s.lat, ve, vn], w.sigF, s.rng, s.departures);
    let closest = Infinity, hit = false, near = 0;   // closest approach to each ship during the step (straight-line motion of both)
    for (let i = 0; i < n; i++) {
      if (!(act0[i] && ts.active[i] && route0[i] === ts.route[i] && ts.s[i] >= s0[i])) continue;
      place(w, ts.route[i], ts.s[i], ts.off[i], PL);
      const ex = (PL[0] - lon) * kx - rx0[i], ey = (PL[1] - lat) * KY - ry0[i], q = clip(-(rx0[i] * ex + ry0[i] * ey) / (ex * ex + ey * ey + 1e-12), 0, 1), d = Math.hypot(rx0[i] + q * ex, ry0[i] + q * ey);
      if (d < closest) closest = d; if (d * 1e3 < 0.5 * (own.L + w.classes[ts.cls[i]].L) + c.hit_margin) hit = true; if (d < c.near_km) near++;
    }
    s.lon = lon; s.lat = lat; s.th = th; s.v = v; s.t += dt;
    const d1 = goal(s), aground = landAt(w, lon, lat) > 0.5, arrived = d1 < c.goal_km;
    return { arrived, collision: hit, aground, nearMisses: near, closestKm: closest, inTrack, speed: v, goalKm: d1, progressKm: d0 - d1, conc: here.conc, thick: here.thick, ve, vn, vGoal };
  }

  // Advance the clock and the traffic while something else moves the own ship: in the page, the ice-breaking simulator when the view
  // is close. The caller has already set s.lon, s.lat, s.th and s.v; ve and vn are the ship's velocity over the ground, m/s.
  function follow(w, s, dt, ve, vn) {
    const c = w.cfg, ts = s.traffic, own = w.classes[s.cls], kx = KY * Math.cos(s.lat * RAD); s.t += dt; s.facc = (s.facc || 0) + dt;
    while (s.facc >= c.dt) { stepTraffic(w, ts, s.t, c.dt, [s.lon, s.lat, ve, vn], w.sigF, s.rng, s.departures); s.facc -= c.dt; }
    let closest = Infinity, hit = false;
    for (let i = 0; i < ts.n; i++) {
      if (!ts.active[i]) continue; place(w, ts.route[i], ts.s[i], ts.off[i], PL);
      const d = Math.hypot((PL[0] - s.lon) * kx, (PL[1] - s.lat) * KY); if (d < closest) closest = d;
      if (d * 1e3 < 0.5 * (own.L + w.classes[ts.cls[i]].L) + c.hit_margin) hit = true;
    }
    const d1 = goal(s);
    return { arrived: d1 < c.goal_km, collision: hit, aground: landAt(w, s.lon, s.lat) > 0.5, closestKm: closest, goalKm: d1 };
  }

  // ------------------------------------------------------------------ a pilot for the page (stlnav has no counterpart: there the policy is what is learned)
  const dist = (lon, lat, q) => Math.hypot((q.lon - lon) * KY * Math.cos(lat * RAD), (q.lat - lat) * KY);
  // The route of voyage `index` as the ship will sail it: 1.5 km to the right of the line, clear of the ships that keep to it.
  function voyage(w, index) {
    const raw = w.voyages[index].path, path = raw.map(q => ({ lat: q[0], lon: q[1] }));
    for (let i = 1; i < raw.length - 1; i++) {
      const a = raw[i - 1], b = raw[i + 1], c = Math.cos(raw[i][0] * RAD), hd = Math.atan2((b[0] - a[0]) * KY, (b[1] - a[1]) * KY * c);
      path[i] = { lat: raw[i][0] - 1.5 * Math.cos(hd) / KY, lon: raw[i][1] + 1.5 * Math.sin(hd) / (KY * c) };
    }
    return { path, k: 1, manual: null, avoiding: false };
  }
  // Action for one step: head for nav.manual if there is one, else for the next point of the route, and keep clear of other ships.
  // Keeping clear: when a ship will pass within `clear` km in the next 15 minutes, turn to starboard, as the traffic does when it
  // follows the rules; if land lies that way, keep to the route instead, at a quarter of the throttle. A ship coming up from astern is
  // left to pass: it moves to its starboard, so the own ship eases to port. Within 1 km of a ship the own ship also slows down.
  // opt.land(lon, lat) -> true on land replaces the model's own land, whose cells are 2 km wide (the page passes its finer coast).
  function pilot(w, s, nav, thr, opt) {
    const o = Object.assign({ avoid: true, clear: 1.2 }, opt || {}), own = w.classes[s.cls], dt = w.cfg.dt, last = nav.path.length - 1, ts = s.traffic;
    const dry = o.land || ((lon, lat) => landAt(w, lon, lat) > 0.2);
    nav.avoiding = false; nav.tgt = nav.manual || nav.path[nav.k];
    if (o.avoid) {
      const cs = Math.cos(s.th), sn = Math.sin(s.th), kx = KY * Math.cos(s.lat * RAD); let risk = false, astern = false, near = Infinity;
      for (let i = 0; i < ts.n; i++) {
        if (!ts.active[i]) continue; place(w, ts.route[i], ts.s[i], ts.off[i], PL);
        const dx = (PL[0] - s.lon) * kx, dy = (PL[1] - s.lat) * KY, d = Math.hypot(dx, dy); if (d > 10) continue;
        const ux = ts.v[i] * Math.cos(PL[2]), uy = ts.v[i] * Math.sin(PL[2]), rx = (ux - s.v * cs) * 1e-3, ry = (uy - s.v * sn) * 1e-3, t = -(dx * rx + dy * ry) / (rx * rx + ry * ry + 1e-12);
        if (!(d < 0.6 || (t > 0 && t < 900 && Math.hypot(dx + rx * t, dy + ry * t) < o.clear))) continue;
        const brg = Math.abs(Math.atan2(dx * sn - dy * cs, dx * cs + dy * sn)) / RAD;             // the other ship's bearing off the bow
        if (brg > 112.5 && ux * cs + uy * sn > s.v) astern = true; else risk = true;
        near = Math.min(near, d);
      }
      if (risk || astern) {
        nav.avoiding = true; const slow = near < 1 ? Math.min(thr, 0.25) : thr, a = s.th - 0.8, c = Math.cos(s.lat * RAD);
        if (!risk) return [thr, 0.5];
        let blocked = false; for (const d of [0.5, 1, 2, 4]) if (dry(s.lon + d * Math.cos(a) / (KY * c), s.lat + d * Math.sin(a) / KY)) blocked = true;
        if (!blocked) return [slow, -1];
        thr = Math.min(thr, 0.25);
      }
    }
    for (let pass = 0; pass < 4; pass++) {
      const tgt = nav.manual || nav.path[nav.k], d = dist(s.lon, s.lat, tgt); nav.tgt = tgt;
      if (nav.manual && d < (nav.manual.reach || 2)) {   // detour done: back to the route, at the nearest point still ahead
        nav.manual = null; let j = nav.k; for (let q = nav.k; q <= last; q++) if (dist(s.lon, s.lat, nav.path[q]) < dist(s.lon, s.lat, nav.path[j])) j = q;
        nav.k = j < last && dist(s.lon, s.lat, nav.path[j + 1]) < dist(nav.path[j].lon, nav.path[j].lat, nav.path[j + 1]) ? j + 1 : j; continue;
      }
      if (!nav.manual && d < (tgt.reach || 6) && nav.k < last) { nav.k++; continue; }   // a point of the route is passed at 6 km, or at its own `reach`
      const dx = (tgt.lon - s.lon) * KY * Math.cos(s.lat * RAD), dy = (tgt.lat - s.lat) * KY, err = (((Math.atan2(dy, dx) - s.th + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      return [thr, clip(err / (own.turn * clip(s.v / own.uow, 0.2, 1) * dt), -1, 1)];
    }
    return [thr, 0];
  }

  return { create, sample, landAt, lindqvist, attainableSpeed, iceResistance, windForce, place, newTraffic, resetTraffic, stepTraffic, reset, step, follow, voyage, pilot, makeRng, typical, KY };
})();
if (typeof module !== 'undefined') module.exports = StlNav;
