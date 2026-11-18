// Steady-state ice resistance of the JavaScript core against Lindqvist's formula.
//   node checks/resistance_vs_lindqvist.js
const IceSim = require('../icebreaker-core.js');
for (const [scenario, h] of [['level', 0.3], ['level', 0.6], ['level', 0.9], ['level', 1.2], ['floes', 0.6], ['pack', 0.6]]) {
  const sim = IceSim.create({ scenario, h }), N = Math.round(70 / sim.dt); let su = 0, n = 0, lq = 0; const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    IceSim.step(sim);
    if (Math.abs(sim.t - 30) < sim.dt / 2) { sim.accX = 0; sim.accN = 0; }        // average over the last 40 s only
    if (sim.t > 30) { su += sim.ship.u; n++; lq += IceSim.lindqvist(sim, h, sim.ship.u).total; }
  }
  const wall = (Date.now() - t0) / 1000, ice = -sim.accX / sim.accN;
  console.log(JSON.stringify({ scenario, h, speed: +(su / n).toFixed(2), ice_kN: Math.round(ice / 1e3), lindqvist_kN: Math.round(lq / n / 1e3),
    ratio: scenario === 'level' ? +(ice / (lq / n)).toFixed(2) : null, cusps: sim.stats.cusps, splits: sim.stats.splits, x_realtime: Math.round(70 / wall) }));
}
