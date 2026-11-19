// The endless ice field of icebreaker-core.js (scenario 'field'), which the page uses around the ship.
//   node checks/field.js
// The ship starts inside the ice with a channel astern; ice is made ahead of it and dropped behind it.
const IceSim = require('../icebreaker-core.js');
let okAll = true;
const report = (name, ok, detail) => { okAll = okAll && ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`); };
function run(opt, seconds, u0, from) {
  const sim = IceSim.create(Object.assign({ scenario: 'field', seed: 3, u0 }, opt)); sim.ship.thr = 1;
  const N = Math.round(seconds / sim.dt), out = { first: 0, mean: 0, maxSheets: 0, maxPieces: 0, sim }; let su = 0, n = 0, s0 = 0, n0 = 0;
  for (let i = 0; i < N; i++) {
    IceSim.step(sim); if (i % 6 === 0) IceSim.follow(sim);
    if (sim.t < 15) { s0 += sim.ship.u; n0++; } if (sim.t > from) { su += sim.ship.u; n++; }
    out.maxSheets = Math.max(out.maxSheets, sim.sheets.length); out.maxPieces = Math.max(out.maxPieces, sim.pieces.length);
  }
  out.first = s0 / n0; out.mean = su / n; return out;
}
function classic(h) {   // the fixed sheet that the ship enters from open water
  const sim = IceSim.create({ scenario: 'level', h, seed: 3, cracks: false }); sim.sheets[0].ridges = null; let su = 0, n = 0;
  for (let i = 0, N = Math.round(90 / sim.dt); i < N; i++) { IceSim.step(sim); if (sim.t > 45) { su += sim.ship.u; n++; } }
  return su / n;
}
for (const [h, u0] of [[0.3, 7.3], [0.6, 5.9]]) {
  const f = run({ h, field: { at: () => ({ conc: 1, h }) } }, 120, u0, 45), c = classic(h), s = f.sim.ship;
  report(`level ice of ${h} m: same speed as in the fixed sheet`, Math.abs(f.mean - c) / c < 0.03 && Math.abs(f.first - u0) / u0 < 0.05 && isFinite(s.x),
    `endless field ${f.mean.toFixed(2)} m/s, fixed sheet ${c.toFixed(2)} m/s; first 15 s ${f.first.toFixed(2)} m/s from a start at ${u0}; ${Math.round(s.x)} m sailed, ` +
    `at most ${f.maxSheets} sheets and ${f.maxPieces} pieces alive`);
}
{
  const f = run({ h: 0.3, field: { at: () => ({ conc: 0.7, h: 0.3 }) } }, 90, 7.8, 30), g = run({ h: 0.3, field: { at: () => ({ conc: 0.3, h: 0.3 }) } }, 60, 8.2, 20), l = run({ h: 0.3, field: { at: () => ({ conc: 1, h: 0.3 }) } }, 60, 7.3, 30);
  report('floes: the less ice, the faster', l.mean < f.mean && f.mean < g.mean && f.maxSheets < 500,
    `level ${l.mean.toFixed(2)} m/s, 70 % of floes ${f.mean.toFixed(2)} m/s, 30 % ${g.mean.toFixed(2)} m/s; at most ${f.maxSheets} floes alive at 70 %`);
}
{
  const f = run({ h: 0.4, field: { at: x => ({ conc: x > 600 ? 0.6 : 1, h: 0.4 }) } }, 150, 6.8, 120), s = f.sim.ship;
  report('from level ice into floes without a stop', isFinite(s.u) && s.x > 900 && f.mean > f.first + 0.4, `${f.first.toFixed(2)} m/s in the level ice, ${f.mean.toFixed(2)} m/s in the floes beyond 600 m; ${Math.round(s.x)} m sailed`);
}
console.log(okAll ? 'ALL PASS' : 'SOME CHECKS FAILED');
process.exit(okAll ? 0 : 1);
