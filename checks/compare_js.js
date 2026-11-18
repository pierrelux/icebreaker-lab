// JavaScript core on uniform ice with a straight edge: speed and ice resistance, 8 runs per thickness.
// Compare with checks/compare_jax.py.
//   node checks/compare_js.js
const IceSim = require('../icebreaker-core.js');
const stats = a => { const m = a.reduce((x, y) => x + y) / a.length; return [+m.toFixed(2), +Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)).toFixed(2)]; };
for (const h of [0.3, 0.6, 0.9, 1.2]) {
  const us = [], fs = [];
  for (let seed = 1; seed <= 8; seed++) {
    const sim = IceSim.create({ scenario: 'level', h, seed, cracks: false }), S = sim.sheets[0];
    S.hvar = 0; S.ridges = null;                                                   // uniform thickness
    for (let j = 0; j < S.ny; j++) for (let i = 0; i < S.nx; i++) { const x = i * S.cell, y = j * S.cell; S.sdf[j * S.nx + i] = Math.max(150 - x, x - 1097, 3 - y, y - 437); }
    const N = Math.round(70 / sim.dt); let su = 0, n = 0;
    for (let i = 0; i < N; i++) { IceSim.step(sim); if (Math.abs(sim.t - 30) < sim.dt / 2) { sim.accX = 0; sim.accN = 0; } if (sim.t > 30) { su += sim.ship.u; n++; } }
    us.push(su / n); fs.push(-sim.accX / sim.accN / 1e3);
  }
  console.log(JSON.stringify({ core: 'js', h, speed_mean_sd: stats(us), ice_kN_mean_sd: stats(fs).map(Math.round) }));
}
