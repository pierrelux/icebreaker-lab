// The other half of checks/stlnav_js.py: repeats each of its cases with stlnav-core.js and compares.
// Not meant to be run by hand; the Python script writes the file of reference results and calls this.
const fs = require('fs'), path = require('path'), StlNav = require('../stlnav-core.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'stl-data.json'), 'utf8')), ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
let okAll = true;
const report = (name, ok, detail) => { okAll = okAll && ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`); };
const KY = StlNav.KY, km = (a, b) => Math.hypot((a[0] - b[0]) * KY * Math.cos(a[1] * Math.PI / 180), (a[1] - b[1]) * KY);
const act = k => [0.75 + 0.25 * Math.sin(k / 50), 0.25 * Math.sin(k / 23)];
const W = () => StlNav.create(data, { thick: ref.thick });
const loadTraffic = (w, t) => { const ts = StlNav.newTraffic(t.active.length); for (const k of Object.keys(t)) ts[k].set(t[k]); return ts; };

// 1. speeds
{
  const w = W(), q = ref.speeds; let worst = 0, stopDiff = 0;
  for (let c = 0; c < 3; c++) for (let i = 0; i < q.h.length; i++) {
    const v = StlNav.attainableSpeed(w, w.classes[c], q.thr[i], q.h[i], q.conc[i], 0.55e6, q.fext[i], q.track[i]), r = q.v[c][i];
    if ((v === 0) !== (r === 0)) stopDiff++; else worst = Math.max(worst, Math.abs(v - r));
  }
  report('speed a ship can hold', worst < 2e-3 && stopDiff <= 2, `${3 * q.h.length} random cases, three classes: largest difference ${worst.toExponential(1)} m/s; cases where only one of the two stops the ship: ${stopDiff}`);
}
// 2. fields
{
  const w = W(), q = ref.fields, worst = { conc: 0, thick: 0, uc: 0, vc: 0, uw: 0, vw: 0, land: 0 };
  for (let i = 0; i < q.lon.length; i++) {
    const f = StlNav.sample(w, q.lon[i], q.lat[i], q.t[i], {});
    for (const k of ['conc', 'thick', 'uc', 'vc', 'uw', 'vw']) worst[k] = Math.max(worst[k], Math.abs(f[k] - q[k][i]));
    worst.land = Math.max(worst.land, Math.abs(StlNav.landAt(w, q.lon[i], q.lat[i]) - q.land[i]));
  }
  const m = Math.max(...Object.values(worst));
  report('fields at a place and time', m < 2e-3, `${q.lon.length} random places and times: largest difference ${m.toExponential(1)} (concentration, thickness in m, current and wind in m/s, land share)`);
}
// 3. the own ship alone
{
  const w = W(), q = ref.alone, s = StlNav.reset(w, { lon: q.start[0], lat: q.start[1], th: q.start[2], goalLon: q.goal[0], goalLat: q.goal[1], traffic: false });
  let worst = 0, dv = 0, j = 0;
  for (let k = 0; k < 600; k++) { StlNav.step(w, s, act(k)); if (k % 60 === 59) { const r = q.states[j++]; worst = Math.max(worst, km([s.lon, s.lat], r)); dv = Math.max(dv, Math.abs(s.v - r[3])); } }
  report('own ship alone, 10 h under the same actions', worst < 0.2 && dv < 0.01, `largest difference in position ${(worst * 1e3).toFixed(0)} m, in speed ${dv.toExponential(1)} m/s (10 comparisons, one per hour)`);
}
// 4. the own ship among traffic
{
  const w = W(), q = ref.with_traffic, a = ref.alone, s = StlNav.reset(w, { lon: a.start[0], lat: a.start[1], th: a.start[2], goalLon: a.goal[0], goalLat: a.goal[1], traffic: false });
  s.traffic = loadTraffic(w, q.traffic);
  let own = 0, ds = 0, dvv = 0, flips = 0, near = 0, tracks = 0, dc = 0, j = 0, ships = 0;
  for (let k = 0; k < 360; k++) {
    const info = StlNav.step(w, s, act(k)); near += info.nearMisses; tracks += info.inTrack ? 1 : 0;
    const rc = q.closest[k]; if (rc >= 0 && isFinite(info.closestKm)) dc = Math.max(dc, Math.abs(info.closestKm - rc));
    if (k % 60 === 59) {
      const r = q.records[j++]; own = Math.max(own, km([s.lon, s.lat], r.own));
      for (let i = 0; i < s.traffic.n; i++) {
        if (!!s.traffic.active[i] !== !!r.traffic.active[i]) { flips++; continue; }
        if (!r.traffic.active[i]) continue;
        if (j === 1) ships++;
        ds = Math.max(ds, Math.abs(s.traffic.s[i] - r.traffic.s[i])); dvv = Math.max(dvv, Math.abs(s.traffic.v[i] - r.traffic.v[i]));
      }
    }
  }
  report('own ship among traffic, 6 h, no departures', own < 0.2 && ds < 0.2 && dvv < 0.05 && flips === 0 && near === q.near && tracks === q.in_track,
    `${ships} ships at sea after the first hour. Largest differences: own position ${(own * 1e3).toFixed(0)} m, a ship's place along its route ${(ds * 1e3).toFixed(0)} m, its speed ${dvv.toExponential(1)} m/s, `
    + `closest approach ${(dc * 1e3).toFixed(0)} m. Ships present in one and not the other: ${flips}. Near misses ${near} against ${q.near}; steps in another ship's track ${tracks} against ${q.in_track}`);
}
// 5. encounters and convoy on the test route
{
  const base = () => { const w = W(); w.routes = ref.test_route; w.R = 1; w.conc.fill(0); w.ind.fill(0); w.cur.u.fill(0); w.cur.v.fill(0); w.wind.u.fill(0); w.wind.v.fill(0); return w; };
  let worst = 0; const txt = [];
  for (const e of ref.encounters) {
    const w = base(), s = StlNav.reset(w, { lon: e.own[0], lat: e.own[1], th: e.own[2], goalLon: e.goal[0], goalLat: e.goal[1], traffic: false }), ts = s.traffic;
    s.v = e.own[3]; ts.active[0] = 1; ts.s.fill(20); ts.v.fill(6); ts.vdes.fill(6); ts.cls.fill(1); ts.comply.fill(e.comply);
    let best = Infinity; for (let k = 0; k < 60; k++) best = Math.min(best, StlNav.step(w, s, [e.thr, 0]).closestKm);
    worst = Math.max(worst, Math.abs(best - e.closest)); if (e.comply) txt.push(`${e.kind} ${best.toFixed(2)} km (${e.closest.toFixed(2)})`);
  }
  report('the three collision rules', worst < 0.02, `closest approach when the other ship gives way, JavaScript (Python): ${txt.join(', ')}; largest difference over the 6 cases ${(worst * 1e3).toFixed(0)} m`);
  let dv = 0; const tx = [];
  for (const c of ref.convoy) {
    const w = base(); w.conc.fill(1); w.ind.fill(1); w.thick = 0.35;
    const ts = StlNav.newTraffic(64); ts.active[0] = 1; ts.active[1] = c.leader; ts.s.fill(10); ts.s[1] = 12; ts.v.fill(1); ts.vdes.fill(7); ts.cls.fill(1); ts.cls[1] = 0; ts.comply.fill(1);
    for (let i = 0; i < 60; i++) StlNav.stepTraffic(w, ts, i * 60, 60, [0, 0, 0, 0], 0.55e6, null, false);
    dv = Math.max(dv, Math.abs(ts.v[0] - c.v[0])); tx.push(`${c.leader ? 'behind an icebreaker' : 'alone'} ${ts.v[0].toFixed(2)} m/s (${c.v[0].toFixed(2)})`);
  }
  report('convoy in 0.35 m of ice', dv < 0.01, `bulk carrier, JavaScript (Python): ${tx.join(', ')}`);
}
// 6. the traffic generator on its mean (its random numbers differ from Python's)
{
  const w = W(); w.conc.fill(0); w.ind.fill(0);
  const ty = StlNav.typical(w); let expected = 0; for (let r = 0; r < w.R; r++) expected += w.routes.rate[r] * ty.transit[r] + (w.routes.period[r] > 0 ? ty.transit[r] / w.routes.period[r] : 0);
  let sum = 0, cnt = 0, most = 0;
  for (let run = 0; run < 16; run++) {
    const rng = StlNav.makeRng(100 + run), ts = StlNav.resetTraffic(w, w.cfg.n_traffic, 0, rng);
    for (let k = 0; k < 1440; k++) { StlNav.stepTraffic(w, ts, k * 120, 120, [0, 0, 0, 0], 0.55e6, rng, true); let a = 0; for (let i = 0; i < ts.n; i++) a += ts.active[i]; sum += a; cnt++; most = Math.max(most, a); }
  }
  const got = sum / cnt;
  report('mean number of ships at sea in open water', Math.abs(got - expected) / expected < 0.1 && most < w.cfg.n_traffic, `simulated ${got.toFixed(1)}, rate x transit time ${expected.toFixed(1)}, most at once ${most} of ${w.cfg.n_traffic} slots (16 runs of 48 h)`);
}
// 7. the pilot of the page (JavaScript only): voyages at full throttle in the page's ice, with and without its rule for keeping clear
{
  const run = (vi, seed, avoid) => {
    const w = StlNav.create(data, { thick: 0.3, wrapTime: true }), nav = StlNav.voyage(w, vi), p = nav.path, e = p[p.length - 1];
    const s = StlNav.reset(w, { lon: p[0].lon, lat: p[0].lat, th: Math.atan2((p[1].lat - p[0].lat) * KY, (p[1].lon - p[0].lon) * KY * Math.cos(p[0].lat * Math.PI / 180)), goalLon: e.lon, goalLat: e.lat, seed });
    for (let k = 0; k < 60 * 72; k++) { const info = StlNav.step(w, s, StlNav.pilot(w, s, nav, 1, { avoid })); if (info.arrived) return (s.t - s.t0) / 3600; if (info.collision) return -1; if (info.aground) return -2; }
    return -3;
  };
  const N = 15, txt = []; let arrived = 0, aground = 0, bare = 0;
  for (let vi = 0; vi < data.voyages.length; vi++) {
    const h = []; for (let seed = 1; seed <= N; seed++) { const r = run(vi, seed, true); if (r > 0) h.push(r); if (r === -2) aground++; if (run(vi, seed, false) > 0) bare++; }
    arrived += h.length; txt.push(`${data.voyages[vi].name} ${h.length}/${N}` + (h.length ? ` in ${(h.reduce((x, y) => x + y, 0) / h.length).toFixed(1)} h` : ''));
  }
  const tot = N * data.voyages.length;
  report("the page's pilot reaches the destination", arrived / tot >= 0.9 && aground === 0, `${arrived} of ${tot} voyages arrive (${txt.join('; ')}); the rest end in a collision. Without the rule for keeping clear: ${bare} of ${tot}`);
}
console.log(okAll ? 'ALL PASS' : 'SOME CHECKS FAILED');
process.exit(okAll ? 0 : 1);
