// Icebreaker Lab: physics core. No DOM; runs in a browser or in Node.
//
// Top-down 2D model with the out-of-plane bending handled analytically:
//   * every ice sheet is a rigid body carrying a signed-distance field (negative inside the ice);
//   * the hull waterline is sampled every ~0.5 m and tested against those fields;
//   * contact crushes the ice edge at pressure sigC on the inclined hull face;
//   * when the vertical part of that force exceeds the bearing capacity of the ice wedge,
//       P_f = C_f (theta/pi)^2 sigF h^2,
//     a cusp of radius R = C_l * l * (1 + C_v * v_n) breaks off (l = characteristic length);
//   * a finite floe splits when the in-plane force exceeds C_s h K_Ic sqrt(L);
//   * broken pieces are rigid convex polygons; pieces under the hull add Lindqvist's
//     submersion resistance.
//
// Scenarios 'level', 'floes' and 'pack' are fixed patches that the ship enters from open water. Scenario 'field' is endless: the ship
// starts inside the ice, and ice is made ahead of it and dropped behind it as it goes (see `follow`).
const IceSim = (() => {
  'use strict';
  const PI = Math.PI, TAU = 2 * PI, DEG = PI / 180;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const wrapPi = a => { a = (a + PI) % TAU; if (a < 0) a += TAU; return a - PI; };

  function makeRng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) >>> 0; let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hash2(ix, iy, s) {
    let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(s, 1274126177)) | 0;
    h = Math.imul(h ^ (h >>> 13), 1103515245); h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }
  function vnoise(x, y, s) {
    const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
    const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
    const a = hash2(ix, iy, s), b = hash2(ix + 1, iy, s), c = hash2(ix, iy + 1, s), d = hash2(ix + 1, iy + 1, s);
    return (a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy) * 2 - 1;
  }

  const DEFAULTS = {
    scenario: 'level', seed: 7, cell: 0.5, dt: 1 / 360, cracks: true,
    h: 0.6,            // ice thickness, m
    sigF: 0.55e6,      // flexural strength, Pa
    sigC: 2.3e6,       // effective crushing pressure, Pa
    mu: 0.15,          // ice-hull friction
    E: 5.4e9, nu: 0.33, rhoW: 1025, rhoI: 880, g: 9.81,
    KIc: 150e3,        // fracture toughness, Pa sqrt(m)
    Cf: 3.1, Cl: 0.30, Cv: -0.14, Csplit: 1.2,
    minSheetArea: 60, maxPieces: 9000,
    ship: {
      L: 76, B: 18, T: 6.5, Cb: 0.55, bowFrac: 0.36, sternFrac: 0.2, bowPow: 2, sternTaper: 0.45,
      // angle of the hull surface normal below horizontal (0 = vertical side)
      phiMid: 8 * DEG, phiBow: 46 * DEG, phiStem: 62 * DEG, phiStern: 30 * DEG,
      ds: 0.5, TB: 2.0e6, uow: 8.4,
    },
  };

  const COS24 = new Float64Array(24), SIN24 = new Float64Array(24);
  for (let k = 0; k < 24; k++) { COS24[k] = Math.cos(k * TAU / 24); SIN24[k] = Math.sin(k * TAU / 24); }
  const T2 = [0, 0], G2 = [0, 0];

  // ------------------------------------------------------------------ hull
  function makeHull(o) {
    const L = o.L, hb = o.B / 2, Lb = o.bowFrac * L, Ls = o.sternFrac * L, xb = L / 2, xs = -L / 2;
    const half = x => {
      if (x > xb - Lb) return hb * (1 - Math.pow((x - (xb - Lb)) / Lb, o.bowPow));
      if (x < xs + Ls) return hb * (1 - o.sternTaper * Math.pow((xs + Ls - x) / Ls, 2.5));
      return hb;
    };
    // dense outline: stem, +y side aft, transom, -y side forward (counter-clockwise)
    const dx = [], dy = [], M = 1500;
    for (let i = 0; i <= M; i++) { const x = xb - L * i / M; dx.push(x); dy.push(half(x)); }
    const yT = half(xs);
    for (let i = 1; i < 40; i++) { dx.push(xs); dy.push(yT * (1 - i / 20)); }
    for (let i = M; i >= 1; i--) { const x = xb - L * i / M; dx.push(x); dy.push(-half(x)); }
    const m = dx.length, cum = new Float64Array(m + 1);
    for (let i = 0; i < m; i++) { const j = (i + 1) % m; cum[i + 1] = cum[i] + Math.hypot(dx[j] - dx[i], dy[j] - dy[i]); }
    const total = cum[m], n = 2 * Math.round(total / (2 * o.ds)), ds = total / n;
    const F = () => new Float64Array(n);
    const H = { n, L, B: o.B, ds, px: F(), py: F(), nx: F(), ny: F(), phi: F(), sinp: F(), cosp: F(), tanp: F(),
      wx: F(), wy: F(), pen: F(), plx: F(), ply: F(), f: F(), own: new Int32Array(n), area: 0, R: 0 };
    let seg = 0;
    for (let k = 0; k < n; k++) {
      const t = k * ds; while (cum[seg + 1] < t) seg++;
      const j = (seg + 1) % m, f = (t - cum[seg]) / (cum[seg + 1] - cum[seg] || 1);
      H.px[k] = dx[seg] + (dx[j] - dx[seg]) * f; H.py[k] = dy[seg] + (dy[j] - dy[seg]) * f;
    }
    let area = 0; const key = [];
    for (let k = 0; k < n; k++) {
      const a = (k + n - 1) % n, b = (k + 1) % n;
      let tx = H.px[b] - H.px[a], ty = H.py[b] - H.py[a]; const l = Math.hypot(tx, ty); tx /= l; ty /= l;
      let nx = ty, ny = -tx; if (nx * H.px[k] + ny * H.py[k] < 0) { nx = -nx; ny = -ny; }
      H.nx[k] = nx; H.ny[k] = ny;
      const x = H.px[k], xiB = (x - (xb - Lb)) / Lb, xiS = (xs + Ls - x) / Ls;
      let phi = o.phiMid;
      if (xiB > 0) phi += (o.phiBow - o.phiMid) * sstep(0, 0.3, xiB) + (o.phiStem - o.phiBow) * sstep(0.8, 1, xiB);
      else if (xiS > 0) phi += (o.phiStern - o.phiMid) * sstep(0, 0.8, xiS);
      H.phi[k] = phi; H.sinp[k] = Math.sin(phi); H.cosp[k] = Math.cos(phi); H.tanp[k] = Math.tan(phi);
      if (xiB > 0.5 && k % 2 === 0) key.push(k);
      area += H.px[k] * H.py[b] - H.px[b] * H.py[k]; H.R = Math.max(H.R, Math.hypot(H.px[k], H.py[k]));
    }
    H.area = Math.abs(area) / 2; H.R += 1; H.key = new Int32Array(key);
    // signed distance to the waterline on a body-fixed grid (for the broken pieces)
    const hc = 0.4, gx0 = xs - 4, gy0 = -hb - 4, gnx = Math.ceil((L + 8) / hc) + 1, gny = Math.ceil((o.B + 8) / hc) + 1;
    const G = new Float32Array(gnx * gny);
    for (let j = 0; j < gny; j++) for (let i = 0; i < gnx; i++) {
      const qx = gx0 + i * hc, qy = gy0 + j * hc; let best = 1e18, inside = true;
      for (let k = 0; k < n; k += 2) {
        const b = (k + 2) % n, ax = H.px[k], ay = H.py[k], ex = H.px[b] - ax, ey = H.py[b] - ay, wx = qx - ax, wy = qy - ay;
        let t = (wx * ex + wy * ey) / (ex * ex + ey * ey); t = t < 0 ? 0 : t > 1 ? 1 : t;
        const ddx = wx - ex * t, ddy = wy - ey * t, d2 = ddx * ddx + ddy * ddy; if (d2 < best) best = d2;
        if (ex * wy - ey * wx < 0) inside = false;
      }
      G[j * gnx + i] = (inside ? -1 : 1) * Math.sqrt(best);
    }
    H.sd = (bx, by) => {
      const fx = (bx - gx0) / hc, fy = (by - gy0) / hc;
      if (fx < 0 || fy < 0 || fx >= gnx - 1 || fy >= gny - 1) return 9;
      const i = fx | 0, j = fy | 0, u = fx - i, v = fy - j, k = j * gnx + i;
      return (G[k] * (1 - u) + G[k + 1] * u) * (1 - v) + (G[k + gnx] * (1 - u) + G[k + gnx + 1] * u) * v;
    };
    H.grad = (bx, by, out) => {
      const gx = H.sd(bx + hc, by) - H.sd(bx - hc, by), gy = H.sd(bx, by + hc) - H.sd(bx, by - hc), l = Math.hypot(gx, gy) || 1;
      out[0] = gx / l; out[1] = gy / l;
    };
    return H;
  }

  function makeShip(o) {
    const hull = makeHull(o), m = o.Cb * o.L * o.B * o.T * 1025, Iz = m * (0.25 * o.L) * (0.25 * o.L);
    return {
      o, hull, L: o.L, B: o.B, T: o.T, m, mx: 1.05 * m, my: 1.7 * m, Izz: 1.6 * Iz,
      x: 0, y: 0, th: 0, u: 0, v: 0, r: 0, thr: 1, rud: 0, rudCmd: 0, TB: o.TB, uow: o.uow, delMax: 35 * DEG,
      kx: 0.7 * o.TB / (o.uow * o.uow), kx1: 0.01 * o.TB / o.uow,
      ky: 0.5 * 1025 * 0.9 * o.L * o.T, ky1: 2e4,
      kr: 0.5 * 1025 * 0.9 * o.T * Math.pow(o.L, 4) / 32 * 1.6, kr1: 2e7,
    };
  }

  // ------------------------------------------------------------------ ice sheet
  class Sheet {
    constructor(nx, ny, cell, h, seed) {
      this.nx = nx; this.ny = ny; this.cell = cell; this.inv = 1 / cell; this.h = h; this.seed = seed | 0;
      this.sdf = new Float32Array(nx * ny).fill(4);
      this.x = 0; this.y = 0; this.th = 0; this.cs = 1; this.sn = 0; this.cx = 0; this.cy = 0;   // pose of the centroid; centroid in grid coordinates
      this.vx = 0; this.vy = 0; this.w = 0; this.fx = 0; this.fy = 0; this.tq = 0; this.hfx = 0; this.hfy = 0; this.htq = 0;
      this.im = 0; this.ii = 0; this.mass = 0; this.area = 0; this.R = 0; this.fixed = false;
      this.hvar = 0.15; this.hox = 0; this.hoy = 0; this.ridges = null;
      this.bpts = null; this.topo = false; this.asleep = false; this.still = 0; this.dead = false;
      this.cracks = []; this.crackDirty = false; this.onDirty = null;
    }
    toLocal(wx, wy, out) { const ax = wx - this.x, ay = wy - this.y; out[0] = this.cx + this.cs * ax + this.sn * ay; out[1] = this.cy - this.sn * ax + this.cs * ay; }
    toWorld(lx, ly, out) { const ax = lx - this.cx, ay = ly - this.cy; out[0] = this.x + this.cs * ax - this.sn * ay; out[1] = this.y + this.sn * ax + this.cs * ay; }
    sample(lx, ly) {
      const gx = lx * this.inv, gy = ly * this.inv, nx = this.nx;
      if (!(gx >= 0 && gy >= 0 && gx <= nx - 1 && gy <= this.ny - 1)) return 1e3;
      let i = gx | 0, j = gy | 0; if (i > nx - 2) i = nx - 2; if (j > this.ny - 2) j = this.ny - 2;
      const fx = gx - i, fy = gy - j, k = j * nx + i, s = this.sdf;
      return (s[k] * (1 - fx) + s[k + 1] * fx) * (1 - fy) + (s[k + nx] * (1 - fx) + s[k + nx + 1] * fx) * fy;
    }
    sampleC(lx, ly) { const c = this.cell; return this.sample(clamp(lx, 0, (this.nx - 1) * c), clamp(ly, 0, (this.ny - 1) * c)); }
    gradW(lx, ly, out) {   // unit normal of the ice edge in world axes, pointing from ice to water
      const e = this.cell * 0.5, gx = this.sampleC(lx + e, ly) - this.sampleC(lx - e, ly), gy = this.sampleC(lx, ly + e) - this.sampleC(lx, ly - e), l = Math.hypot(gx, gy) || 1;
      out[0] = (this.cs * gx - this.sn * gy) / l; out[1] = (this.sn * gx + this.cs * gy) / l;
    }
    hmul(lx, ly) {         // local thickness as a multiple of the nominal thickness
      const x = lx + this.hox, y = ly + this.hoy; let m = 1 + this.hvar * vnoise(x / 47, y / 47, this.seed);
      const rg = this.ridges;
      if (rg) for (let i = 0; i < rg.length; i++) {
        const r = rg[i], ex = r.x2 - r.x1, ey = r.y2 - r.y1, t = clamp(((x - r.x1) * ex + (y - r.y1) * ey) / (ex * ex + ey * ey), 0, 1);
        const dx = x - r.x1 - ex * t, dy = y - r.y1 - ey * t, q = (dx * dx + dy * dy) / (r.w * r.w);
        if (q < 6) m += r.gain * Math.exp(-q) * (0.75 + 0.25 * vnoise(x / 5, y / 5, this.seed + 3));
      }
      return m < 0.35 ? 0.35 : m;
    }
    carve(lx, ly, R, ph1, ph2, amp) {   // remove a (slightly irregular) disc of ice
      const c = this.cell, Rm = R * (1 + amp) + c, nx = this.nx, s = this.sdf;
      const i0 = Math.max(0, Math.floor((lx - Rm) / c)), i1 = Math.min(nx - 1, Math.ceil((lx + Rm) / c));
      const j0 = Math.max(0, Math.floor((ly - Rm) / c)), j1 = Math.min(this.ny - 1, Math.ceil((ly + Rm) / c));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const dx = i * c - lx, dy = j * c - ly, d = Math.sqrt(dx * dx + dy * dy); let Re = R;
        if (amp > 0 && d > 1e-6) { const a = Math.atan2(dy, dx); Re = R * (1 + amp * (0.6 * Math.sin(3 * a + ph1) + 0.4 * Math.sin(5 * a + ph2))); }
        const v = Re - d, k = j * nx + i; if (v > s[k]) s[k] = v;
      }
      this.bpts = null; this.crackDirty = true;
      if (this.onDirty) this.onDirty(lx - Rm, ly - Rm, lx + Rm, ly + Rm);
    }
    carveCapsule(x1, y1, x2, y2, w) {
      const c = this.cell, nx = this.nx, s = this.sdf, m = w + c;
      const i0 = Math.max(0, Math.floor((Math.min(x1, x2) - m) / c)), i1 = Math.min(nx - 1, Math.ceil((Math.max(x1, x2) + m) / c));
      const j0 = Math.max(0, Math.floor((Math.min(y1, y2) - m) / c)), j1 = Math.min(this.ny - 1, Math.ceil((Math.max(y1, y2) + m) / c));
      const ex = x2 - x1, ey = y2 - y1, e2 = ex * ex + ey * ey || 1;
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const px = i * c - x1, py = j * c - y1, t = clamp((px * ex + py * ey) / e2, 0, 1), dx = px - ex * t, dy = py - ey * t;
        const v = w - Math.sqrt(dx * dx + dy * dy), k = j * nx + i; if (v > s[k]) s[k] = v;
      }
      this.bpts = null; this.crackDirty = true;
      if (this.onDirty) this.onDirty(Math.min(x1, x2) - m, Math.min(y1, y2) - m, Math.max(x1, x2) + m, Math.max(y1, y2) + m);
    }
    recompute(P) {   // area, mass, centroid and inertia from the grid; keeps the grid where it is in the world
      const nx = this.nx, ny = this.ny, s = this.sdf, c = this.cell; let A = 0, sx = 0, sy = 0;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (s[j * nx + i] < 0) { A++; sx += i; sy += j; }
      if (!A) { this.area = 0; return; }
      const ncx = sx / A * c, ncy = sy / A * c; let I = 0, r2 = 0;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (s[j * nx + i] < 0) { const dx = i * c - ncx, dy = j * c - ncy, q = dx * dx + dy * dy; I += q; if (q > r2) r2 = q; }
      this.toWorld(ncx, ncy, T2); const dxw = T2[0] - this.x, dyw = T2[1] - this.y;
      this.vx -= this.w * dyw; this.vy += this.w * dxw; this.x = T2[0]; this.y = T2[1]; this.cx = ncx; this.cy = ncy;
      this.area = A * c * c; this.mass = P.rhoI * this.h * this.area; this.R = Math.sqrt(r2) + c;
      const inertia = P.rhoI * this.h * c * c * (I + A * c * c / 6);
      if (!this.fixed) { this.im = 1 / this.mass; this.ii = 1 / inertia; }
    }
    boundary() {     // edge points about 1.5 m apart, in grid coordinates
      if (this.bpts) return this.bpts;
      const nx = this.nx, ny = this.ny, s = this.sdf, c = this.cell, B = 3, bx = Math.ceil(nx / B), used = new Uint8Array(bx * Math.ceil(ny / B)), out = [];
      for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
        const k = j * nx + i; if (s[k] >= 0) continue;
        if (s[k - 1] >= 0 || s[k + 1] >= 0 || s[k - nx] >= 0 || s[k + nx] >= 0) { const q = ((j / B) | 0) * bx + ((i / B) | 0); if (used[q]) continue; used[q] = 1; out.push(i * c, j * c); }
      }
      return (this.bpts = new Float32Array(out));
    }
  }

  const ellOf = (P, h) => Math.pow(P.E * h * h * h / (12 * (1 - P.nu * P.nu) * P.rhoW * P.g), 0.25);

  // Lindqvist (1989) level-ice resistance, used as a reference curve and for the submersion term.
  function lindqvist(sim, h, v) {
    const P = sim.P, o = sim.ship.o, mu = P.mu;
    const phi = PI / 2 - o.phiStem, alpha = Math.atan(o.B / (o.bowFrac * o.L)), psi = Math.atan(Math.tan(phi) / Math.sin(alpha));
    const Rc = 0.5 * P.sigF * h * h * (Math.tan(phi) + mu * Math.cos(phi) / Math.cos(psi)) / (1 - mu * Math.sin(phi) / Math.cos(psi));
    const Rb = 27 / 64 * P.sigF * o.B * Math.pow(h, 1.5) / Math.sqrt(P.E / (12 * (1 - P.nu * P.nu) * P.g * P.rhoW))
      * (Math.tan(psi) + mu * Math.cos(phi) / (Math.cos(psi) * Math.sin(alpha))) * (1 + 1 / Math.cos(psi));
    const k = mu * (0.7 * o.L - o.T / Math.tan(phi) - o.B / (4 * Math.tan(alpha)) + o.T * Math.cos(phi) * Math.cos(psi) * Math.sqrt(1 / Math.sin(phi) ** 2 + 1 / Math.tan(alpha) ** 2));
    const Rs = (P.rhoW - P.rhoI) * P.g * h * o.B * (o.T * (o.B + o.T) / (o.B + 2 * o.T) + k);
    return { Rc, Rb, Rs, total: (Rc + Rb) * (1 + 1.4 * v / Math.sqrt(P.g * h)) + Rs * (1 + 9.4 * v / Math.sqrt(P.g * o.L)) };
  }

  // ------------------------------------------------------------------ hull against ice sheets
  function hullIce(sim) {
    const P = sim.P, sh = sim.ship, H = sh.hull, n = H.n, sheets = sim.sheets;
    const cs = Math.cos(sh.th), sn = Math.sin(sh.th), wx = H.wx, wy = H.wy, pen = H.pen, own = H.own, plx = H.plx, ply = H.ply, fv = H.f;
    sim.FX = sim.FY = sim.MZ = 0;
    for (let i = 0; i < n; i++) { wx[i] = sh.x + cs * H.px[i] - sn * H.py[i]; wy[i] = sh.y + sn * H.px[i] + cs * H.py[i]; pen[i] = 0; own[i] = -1; fv[i] = 0; }
    let any = false;
    for (let k = 0; k < sheets.length; k++) {
      const S = sheets[k], ddx = S.x - sh.x, ddy = S.y - sh.y, rr = S.R + H.R;
      if (ddx * ddx + ddy * ddy > rr * rr) continue;
      const c = S.cs, s = S.sn, ox = S.x, oy = S.y, lcx = S.cx, lcy = S.cy;
      for (let i = 0; i < n; i++) {
        const ax = wx[i] - ox, ay = wy[i] - oy, lx = lcx + c * ax + s * ay, ly = lcy - s * ax + c * ay, d = -S.sample(lx, ly);
        if (d > pen[i]) { pen[i] = d; own[i] = k; plx[i] = lx; ply[i] = ly; any = true; }
      }
    }
    if (!any) return;
    const vxs = sh.u * cs - sh.v * sn, vys = sh.u * sn + sh.v * cs, r = sh.r, Z = sim.Z, v0 = 0.08;
    let start = 0; while (start < n && own[start] >= 0) start++; if (start === n) start = 0;
    let FX = 0, FY = 0, MZ = 0; Z.len = 0;
    for (let q = 0; q < n; q++) {
      const i = (start + q) % n, k = own[i];
      if (k < 0) { if (Z.len > 0) closeZone(sim, Z); continue; }
      if (Z.len > 0 && (Z.k !== k || Z.len >= Z.max)) closeZone(sim, Z);
      const S = sheets[k], lx = plx[i], ly = ply[i], hL = S.h * S.hmul(lx, ly);
      if (Z.len === 0) { Z.k = k; Z.max = 2.2 * P.Cl * ellOf(P, hL); Z.fv = Z.fh = Z.w = Z.x = Z.y = Z.nx = Z.ny = Z.vn = Z.h = 0; }
      const nxw = cs * H.nx[i] - sn * H.ny[i], nyw = sn * H.nx[i] + cs * H.ny[i];
      const rx = wx[i] - sh.x, ry = wy[i] - sh.y; let rvx = vxs - r * ry, rvy = vys + r * rx, sx = 0, sy = 0;
      if (S.im > 0) { sx = wx[i] - S.x; sy = wy[i] - S.y; rvx -= S.vx - S.w * sy; rvy -= S.vy + S.w * sx; }
      const vn = rvx * nxw + rvy * nyw, vt = -rvx * nyw + rvy * nxw, d = pen[i];
      const sinp = H.sinp[i], cosp = H.cosp[i], tanp = H.tanp[i];
      // contact band on the inclined hull face: the edge is crushed from the top down
      const geo = Math.min(hL, d / tanp) / cosp * H.ds;
      const dN = P.sigC * geo * clamp(vn / v0, 0, 1);              // crushing is plastic: no force while unloading
      const vd = vn > 0 ? vn * sinp : 0, sl = Math.hypot(vt, vd) + 1e-4, sd = vd / sl, st = vt / sl;
      const fn = dN * (cosp + P.mu * sinp * sd), ft = P.mu * dN * st, fz = dN * (sinp - P.mu * cosp * sd);
      const fx = -fn * nxw + ft * nyw, fy = -fn * nyw - ft * nxw;
      FX += fx; FY += fy; MZ += rx * fy - ry * fx;
      if (S.im > 0) { S.fx -= fx; S.fy -= fy; S.tq -= sx * fy - sy * fx; S.asleep = false; S.still = 0; }
      fv[i] = fn / H.ds;
      Z.len += H.ds; Z.fv += fz; Z.fh += fn; Z.w += geo; Z.x += geo * lx; Z.y += geo * ly; Z.nx += geo * nxw; Z.ny += geo * nyw; Z.vn += geo * vn; Z.h += geo * hL;
      // through-thickness crushing: ice deeper inside the hull than this no longer exists
      const dmax = hL * tanp + 0.03;
      if (d > dmax) { const rc = 1.2, nlx = S.cs * nxw + S.sn * nyw, nly = -S.sn * nxw + S.cs * nyw; S.carve(lx - nlx * (dmax + rc), ly - nly * (dmax + rc), rc, 0, 0, 0); S.topo = true; }
    }
    if (Z.len > 0) closeZone(sim, Z);
    sim.FX = FX; sim.FY = FY; sim.MZ = MZ;
  }

  function closeZone(sim, Z) {
    Z.len = 0;
    if (Z.w <= 0 || Z.fv <= 0) return;
    const P = sim.P, S = sim.sheets[Z.k], w = Z.w, lx = Z.x / w, ly = Z.y / w, hL = Z.h / w, vn = Z.vn / w;
    const R = P.Cl * ellOf(P, hL) * clamp(1 + P.Cv * vn, 0.4, 1.15);
    let c = 0; for (let a = 0; a < 24; a++) if (S.sample(lx + R * COS24[a], ly + R * SIN24[a]) < 0) c++;
    const th = clamp(c / 24 * TAU, 0.15 * PI, 1.25 * PI) / PI;          // opening angle of the ice wedge, in units of pi
    if (Z.fv > P.Cf * th * th * P.sigF * hL * hL) { sim.pending.push({ type: 0, S, lx, ly, R }); return; }
    if (S.im > 0 && Z.fh > P.Csplit * hL * P.KIc * 3.1) {
      const nl = Math.hypot(Z.nx, Z.ny) || 1, nxw = Z.nx / nl, nyw = Z.ny / nl, dxl = S.cs * nxw + S.sn * nyw, dyl = -S.sn * nxw + S.cs * nyw;
      let Lp = 0; for (let t = 1; t < 600; t++) { if (S.sample(lx + dxl * t, ly + dyl * t) > 0.3) break; Lp = t; }
      if (Lp >= 10 && Z.fh > P.Csplit * hL * P.KIc * Math.sqrt(Lp)) sim.pending.push({ type: 1, S, lx, ly, dxl, dyl });
    }
  }

  // ------------------------------------------------------------------ fracture
  function breakCusp(sim, S, lx, ly, R) {
    const rnd = sim.rand, NA = 24, da = TAU / NA, ph1 = rnd() * TAU, ph2 = rnd() * TAU, amp = 0.13, inn = sim.inn;
    const Re = a => R * (1 + amp * (0.6 * Math.sin(3 * a + ph1) + 0.4 * Math.sin(5 * a + ph2)));
    let cnt = 0;
    for (let k = 0; k < NA; k++) {
      const ca = COS24[k], sa = SIN24[k], re = Re(k * da); let c = 0;
      if (S.sample(lx + ca * re * 0.35, ly + sa * re * 0.35) < 0) c++;
      if (S.sample(lx + ca * re * 0.65, ly + sa * re * 0.65) < 0) c++;
      if (S.sample(lx + ca * re * 0.92, ly + sa * re * 0.92) < 0) c++;
      inn[k] = c >= 2 ? 1 : 0; cnt += inn[k];
    }
    if (cnt >= 2 && sim.pieces.length < sim.o.maxPieces + 600) {
      let k0 = 0; if (cnt < NA) while (!(inn[k0] && !inn[(k0 + NA - 1) % NA])) k0++;
      let visited = 0, k = k0;
      while (visited < NA) {
        if (!inn[k % NA]) { k++; visited++; continue; }
        let len = 0; while (visited + len < NA && inn[(k + len) % NA]) len++;
        if (len >= 2) wedges(sim, S, lx, ly, R, Re, (k - 0.5) * da, len * da);
        k += len; visited += len;
      }
    }
    S.carve(lx, ly, R, ph1, ph2, amp); S.topo = true;
    if (sim.o.cracks && cnt > 0) {                       // hairline radial cracks left in the parent sheet
      const nc = 1 + (rnd() < 0.6 ? 1 : 0);
      for (let c = 0; c < nc; c++) {
        let k = (rnd() * NA) | 0, tries = 0; while (!inn[k] && tries++ < NA) k = (k + 1) % NA;
        const a = k * da + (rnd() - 0.5) * da; let x = lx + Re(a) * Math.cos(a), y = ly + Re(a) * Math.sin(a), dir = a + (rnd() - 0.5) * 0.7;
        const segs = 2 + (rnd() < 0.4 ? 1 : 0), len = R * (0.45 + 1.7 * rnd() * rnd()) / segs;
        for (let s = 0; s < segs; s++) {
          const qx = x + Math.cos(dir) * len, qy = y + Math.sin(dir) * len; if (S.sample(qx, qy) > -0.2) break;
          S.cracks.push(x, y, qx, qy); x = qx; y = qy; dir += (rnd() - 0.5) * 0.9;
        }
      }
    }
    sim.stats.cusps++; S.toWorld(lx, ly, T2); sim.events.push(T2[0], T2[1], R, sim.t);
    if (sim.events.length > 480) sim.events.splice(0, 240);
  }

  function wedges(sim, S, lx, ly, R, Re, a0, span) {   // the broken cusp, cut into wedges by radial cracks
    const rnd = sim.rand; let m = clamp(Math.round(span * R / (1.7 + 1.3 * rnd())), Math.ceil(span / 2.4), 5); if (m < 1) m = 1;
    let b0 = a0;
    for (let q = 0; q < m; q++) {
      const b1 = q === m - 1 ? a0 + span : a0 + span * (q + 1 + 0.36 * (rnd() - 0.5)) / m, mid = 0.5 * (b0 + b1), st = Math.max(2, Math.ceil((b1 - b0) / 0.4));
      const xs = [lx + 0.1 * R * Math.cos(mid)], ys = [ly + 0.1 * R * Math.sin(mid)];
      for (let s = 0; s <= st; s++) { const a = b0 + (b1 - b0) * s / st, re = Re(a) * 0.97; xs.push(lx + re * Math.cos(a)); ys.push(ly + re * Math.sin(a)); }
      for (let i = 0; i < xs.length; i++) { S.toWorld(xs[i], ys[i], T2); xs[i] = T2[0]; ys[i] = T2[1]; }
      const p = addPiece(sim, xs, ys, S.h, 0.93);
      if (p) {
        const rx = p.x - S.x, ry = p.y - S.y; p.vx = S.vx - S.w * ry; p.vy = S.vy + S.w * rx;
        p.sub = true; p.tSub = sim.t; p.w = (rnd() - 0.5) * 0.6;   // the flared bow pushes it under
      }
      b0 = b1;
    }
  }

  function split(sim, S, ev) {   // a crack runs from the contact across the floe
    const rnd = sim.rand, a0 = Math.atan2(ev.dyl, ev.dxl) + (rnd() - 0.5) * 0.5, stepL = 2.2; let a = a0, x = ev.lx, y = ev.ly, out = false;
    const path = [x - Math.cos(a0) * 1.2, y - Math.sin(a0) * 1.2];
    for (let it = 0; it < 500; it++) {
      a += (rnd() - 0.5) * 0.5 + 0.15 * wrapPi(a0 - a); x += Math.cos(a) * stepL; y += Math.sin(a) * stepL; path.push(x, y);
      if (S.sample(x, y) > 0.3) { out = true; break; }
    }
    if (!out) { if (sim.o.cracks) for (let i = 0; i + 3 < path.length; i += 2) S.cracks.push(path[i], path[i + 1], path[i + 2], path[i + 3]); return; }
    for (let i = 0; i + 3 < path.length; i += 2) S.carveCapsule(path[i], path[i + 1], path[i + 2], path[i + 3], 0.42);
    sim.stats.splits++;
    separate(sim, S, { lx: ev.lx, ly: ev.ly, a0 });
  }

  function label(S) {            // connected pieces of ice on the grid
    const nx = S.nx, ny = S.ny, s = S.sdf, N = nx * ny, lab = new Int32Array(N), stack = new Int32Array(N), comps = [];
    for (let k0 = 0; k0 < N; k0++) {
      if (s[k0] >= 0 || lab[k0]) continue;
      const id = comps.length + 1; let sp = 0, cnt = 0, i0 = nx, i1 = 0, j0 = ny, j1 = 0; stack[sp++] = k0; lab[k0] = id;
      while (sp) {
        const k = stack[--sp], i = k % nx, j = (k / nx) | 0; cnt++;
        if (i < i0) i0 = i; if (i > i1) i1 = i; if (j < j0) j0 = j; if (j > j1) j1 = j;
        if (i > 0 && s[k - 1] < 0 && !lab[k - 1]) { lab[k - 1] = id; stack[sp++] = k - 1; }
        if (i < nx - 1 && s[k + 1] < 0 && !lab[k + 1]) { lab[k + 1] = id; stack[sp++] = k + 1; }
        if (j > 0 && s[k - nx] < 0 && !lab[k - nx]) { lab[k - nx] = id; stack[sp++] = k - nx; }
        if (j < ny - 1 && s[k + nx] < 0 && !lab[k + nx]) { lab[k + nx] = id; stack[sp++] = k + nx; }
      }
      comps.push({ id, cnt, i0, i1, j0, j1 });
    }
    return { lab, comps };
  }

  function extract(sim, S, lab, c) {
    const pad = 3, nx = c.i1 - c.i0 + 1 + 2 * pad, ny = c.j1 - c.j0 + 1 + 2 * pad, T = new Sheet(nx, ny, S.cell, S.h, S.seed);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const si = c.i0 - pad + i, sj = c.j0 - pad + j; let v = 2;
      if (si >= 0 && sj >= 0 && si < S.nx && sj < S.ny) { const k = sj * S.nx + si; v = S.sdf[k]; if (v < 0 && lab[k] !== c.id) v = 1.5; }
      T.sdf[j * nx + i] = v;
    }
    const ox = (c.i0 - pad) * S.cell, oy = (c.j0 - pad) * S.cell;
    T.th = S.th; T.cs = S.cs; T.sn = S.sn; T.hvar = S.hvar; T.hox = S.hox + ox; T.hoy = S.hoy + oy; T.ridges = S.ridges;
    S.toWorld(ox, oy, T2); T.x = T2[0]; T.y = T2[1];
    const rx = T.x - S.x, ry = T.y - S.y; T.vx = S.vx - S.w * ry; T.vy = S.vy + S.w * rx; T.w = S.w;
    T.recompute(sim.P);
    const cr = S.cracks;
    for (let i = 0; i < cr.length; i += 4) { const x1 = cr[i] - ox, y1 = cr[i + 1] - oy, x2 = cr[i + 2] - ox, y2 = cr[i + 3] - oy; if (T.sample((x1 + x2) / 2, (y1 + y2) / 2) < -0.1) T.cracks.push(x1, y1, x2, y2); }
    return T;
  }

  function convexHull(xs, ys) {
    const idx = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b] || ys[a] - ys[b]), h = [];
    const cross = (o, a, b) => (xs[a] - xs[o]) * (ys[b] - ys[o]) - (ys[a] - ys[o]) * (xs[b] - xs[o]);
    for (const i of idx) { while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], i) <= 0) h.pop(); h.push(i); }
    const lo = h.length + 1;
    for (let q = idx.length - 2; q >= 0; q--) { const i = idx[q]; while (h.length >= lo && cross(h[h.length - 2], h[h.length - 1], i) <= 0) h.pop(); h.push(i); }
    h.pop(); return h;
  }

  function pieceFromComp(sim, S, lab, c) {
    const xs = [], ys = [], nx = S.nx;
    for (let j = c.j0; j <= c.j1; j++) for (let i = c.i0; i <= c.i1; i++) {
      const k = j * nx + i; if (lab[k] !== c.id) continue;
      if (lab[k - 1] !== c.id || lab[k + 1] !== c.id || lab[k - nx] !== c.id || lab[k + nx] !== c.id) { S.toWorld(i * S.cell, j * S.cell, T2); xs.push(T2[0]); ys.push(T2[1]); }
    }
    if (xs.length < 3) return;
    let h = convexHull(xs, ys); if (h.length < 3) return;
    if (h.length > 10) { const g = []; for (let i = 0; i < 10; i++) g.push(h[Math.floor(i * h.length / 10)]); h = g; }
    const p = addPiece(sim, h.map(i => xs[i]), h.map(i => ys[i]), S.h, 1);
    if (p) { const rx = p.x - S.x, ry = p.y - S.y; p.vx = S.vx - S.w * ry; p.vy = S.vy + S.w * rx; p.w = S.w; }
  }

  function separate(sim, S, ev) {   // after a cut: one grid per connected body
    const { lab, comps } = label(S), c2 = S.cell * S.cell; S.topo = false;
    if (comps.length === 1 && comps[0].cnt * c2 >= sim.o.minSheetArea) { S.recompute(sim.P); S.bpts = null; return; }
    S.dead = true; sim.dirtySheets = true;
    for (const c of comps) {
      const area = c.cnt * c2; if (area < 0.8) continue;
      if (area < sim.o.minSheetArea) { pieceFromComp(sim, S, lab, c); continue; }
      const T = extract(sim, S, lab, c);
      if (ev) {    // the two halves spring apart a little
        S.toWorld(ev.lx, ev.ly, T2); const dx = S.cs * Math.cos(ev.a0) - S.sn * Math.sin(ev.a0), dy = S.sn * Math.cos(ev.a0) + S.cs * Math.sin(ev.a0);
        const side = dx * (T.y - T2[1]) - dy * (T.x - T2[0]) > 0 ? 1 : -1; T.vx += -dy * side * 0.1; T.vy += dx * side * 0.1;
      }
      sim.sheets.push(T);
    }
  }

  // ------------------------------------------------------------------ sheet against sheet
  function edgeContacts(P, A, B) {   // A's edge points that lie inside B push the two apart
    const SP = 1.5, dref = 0.3, pts = A.boundary(), m = pts.length, hh = Math.min(A.h, B.h), BR2 = B.R * B.R;
    for (let q = 0; q < m; q += 2) {
      const ax = pts[q] - A.cx, ay = pts[q + 1] - A.cy, rax = A.cs * ax - A.sn * ay, ray = A.sn * ax + A.cs * ay, wx = A.x + rax, wy = A.y + ray;
      const bx = wx - B.x, by = wy - B.y; if (bx * bx + by * by > BR2) continue;
      const lx = B.cx + B.cs * bx + B.sn * by, ly = B.cy - B.sn * bx + B.cs * by, d = -B.sample(lx, ly); if (d <= 0) continue;
      B.gradW(lx, ly, G2); const gx = G2[0], gy = G2[1];
      const rvx = (A.vx - A.w * ray) - (B.vx - B.w * by), rvy = (A.vy + A.w * rax) - (B.vy + B.w * bx), vn = rvx * gx + rvy * gy, vt = -rvx * gy + rvy * gx;
      let F = P.sigC * hh * SP * (Math.min(d / dref, 1) + 0.3 * Math.max(0, d - dref)) - 1.0e6 * hh * SP * vn; if (F < 0) F = 0;
      const Ft = -0.3 * F * clamp(vt / 0.05, -1, 1), fx = F * gx - Ft * gy, fy = F * gy + Ft * gx;
      A.hfx += fx; A.hfy += fy; A.htq += rax * fy - ray * fx; B.hfx -= fx; B.hfy -= fy; B.htq -= bx * fy - by * fx;
      if (A.im) A.asleep = false; if (B.im) B.asleep = false;
    }
  }
  function sheetContacts(sim) {      // only floes that are moving start a test; resting pairs cost nothing
    const S = sim.sheets, n = S.length, P = sim.P;
    if (sim.awake.length < n) sim.awake = new Uint8Array(2 * n);
    const aw = sim.awake;
    for (let a = 0; a < n; a++) { const A = S[a]; A.hfx = A.hfy = A.htq = 0; aw[a] = A.im > 0 && !A.asleep ? 1 : 0; }
    if (n < 2) return;
    for (let a = 0; a < n; a++) {
      if (!aw[a]) continue; const A = S[a];
      for (let b = 0; b < n; b++) {
        if (a === b) continue; const B = S[b], dx = B.x - A.x, dy = B.y - A.y, rr = A.R + B.R; if (dx * dx + dy * dy > rr * rr) continue;
        edgeContacts(P, A, B);
        if (!aw[b] && B.im > 0) edgeContacts(P, B, A);
      }
    }
  }

  // coarse grid over the floes, so that a piece only looks at the floes around it
  function buildSheetGrid(sim) {
    const S = sim.sheets, g = sim.sg; g.on = false; if (S.length <= 8) return;
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (let k = 0; k < S.length; k++) { const s = S[k]; if (s.x - s.R < x0) x0 = s.x - s.R; if (s.x + s.R > x1) x1 = s.x + s.R; if (s.y - s.R < y0) y0 = s.y - s.R; if (s.y + s.R > y1) y1 = s.y + s.R; }
    const cell = 48, gw = Math.floor((x1 - x0) / cell) + 1, gh = Math.floor((y1 - y0) / cell) + 1; if (gw * gh > 20000) return;
    while (g.lists.length < gw * gh) g.lists.push([]);
    for (let c = 0; c < gw * gh; c++) g.lists[c].length = 0;
    for (let k = 0; k < S.length; k++) {
      const s = S[k], i0 = Math.floor((s.x - s.R - x0) / cell), i1 = Math.floor((s.x + s.R - x0) / cell), j0 = Math.floor((s.y - s.R - y0) / cell), j1 = Math.floor((s.y + s.R - y0) / cell);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) g.lists[j * gw + i].push(k);
    }
    g.on = true; g.x0 = x0; g.y0 = y0; g.cell = cell; g.gw = gw; g.gh = gh;
  }
  function sheetsNear(sim, x, y) {   // indices of candidate floes at a point, or null for "all of them"
    const g = sim.sg; if (!g.on) return null;
    const i = Math.floor((x - g.x0) / g.cell), j = Math.floor((y - g.y0) / g.cell);
    return i < 0 || j < 0 || i >= g.gw || j >= g.gh ? EMPTY : g.lists[j * g.gw + i];
  }
  const EMPTY = [];

  function integrateSheets(sim, dt) {
    const S = sim.sheets;
    for (let k = 0; k < S.length; k++) {
      const s = S[k];
      if (s.im === 0 || s.asleep) { s.fx = s.fy = s.tq = 0; continue; }
      s.vx += (s.fx + s.hfx) * s.im * dt; s.vy += (s.fy + s.hfy) * s.im * dt; s.w += (s.tq + s.htq) * s.ii * dt; s.fx = s.fy = s.tq = 0;
      const sp = Math.hypot(s.vx, s.vy), aw = Math.abs(s.w);
      const f = 1 / (1 + (15 + 12 * sp) * s.area * s.im * dt); s.vx *= f; s.vy *= f;                     // water drag under the floe
      s.w /= 1 + (15 + 12 * aw * s.R * 0.5) * s.area * s.R * s.R * 0.25 * s.ii * dt;
      s.x += s.vx * dt; s.y += s.vy * dt;
      if (s.w !== 0) { s.th += s.w * dt; s.cs = Math.cos(s.th); s.sn = Math.sin(s.th); }
      if (sp < 0.02 && aw < 0.002) { s.still += dt; if (s.still > 0.6) { s.asleep = true; s.vx = s.vy = s.w = 0; } } else s.still = 0;
    }
  }

  // ------------------------------------------------------------------ ship
  function integrateShip(sim, dt) {
    const s = sim.ship, cs = Math.cos(s.th), sn = Math.sin(s.th), ap = sim.ap;
    if (ap.on) {
      if (ap.hasTarget) { const dx = ap.tx - s.x, dy = ap.ty - s.y; if (dx * dx + dy * dy > 900) ap.heading = Math.atan2(dy, dx); else ap.hasTarget = false; }
      s.rudCmd = clamp(2.2 * wrapPi(ap.heading - s.th) - 9 * s.r, -1, 1);
    }
    s.rud += clamp(s.rudCmd - s.rud, -1.5 * dt, 1.5 * dt);
    const Xi = sim.FX * cs + sim.FY * sn, Yi = -sim.FX * sn + sim.FY * cs;
    const T = s.thr >= 0 ? s.thr * s.TB * (1 - 0.3 * clamp(s.u / s.uow, 0, 1.5)) : s.thr * 0.6 * s.TB;
    const del = s.rud * s.delMax, au = Math.abs(s.u);                                         // azimuthing thrusters at the stern
    const Xt = T * Math.cos(del), Yt = -T * Math.sin(del), Nt = 0.45 * s.L * T * Math.sin(del);
    const Xh = -(s.kx * au + s.kx1) * s.u, Yh = -(s.ky * Math.abs(s.v) + s.ky1) * s.v, Nh = -(s.kr * Math.abs(s.r) + s.kr1) * s.r;
    sim.cover += (clamp(sim.subArea / (0.7 * s.hull.area), 0, 1) - sim.cover) * Math.min(1, dt / 0.5);
    const Xs = -sim.cover * sim.Rs0 * (1 + 9.4 * au / Math.sqrt(sim.P.g * s.L)) * Math.tanh(s.u / 0.25);
    const du = (Xi + Xt + Xh + Xs + s.my * s.v * s.r) / s.mx, dv = (Yi + Yt + Yh - s.mx * s.u * s.r) / s.my, dr = (sim.MZ + Nt + Nh) / s.Izz;
    s.u += du * dt; s.v += dv * dt; s.r += dr * dt;
    s.x += (s.u * cs - s.v * sn) * dt; s.y += (s.u * sn + s.v * cs) * dt; s.th += s.r * dt;
    sim.accX += Xi + Xs; sim.accN++;
  }

  // ------------------------------------------------------------------ broken pieces
  function addPiece(sim, xs, ys, h, shrink) {
    const n = xs.length; if (n < 3) return null;
    let A = 0, cx = 0, cy = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) { const c = xs[j] * ys[i] - xs[i] * ys[j]; A += c; cx += (xs[j] + xs[i]) * c; cy += (ys[j] + ys[i]) * c; }
    if (Math.abs(A) < 1.0) return null;                 // under half a square metre: slush, not simulated
    cx /= 3 * A; cy /= 3 * A;
    const rnd = sim.rand, p = { x: cx, y: cy, th: 0, vx: 0, vy: 0, w: 0, n, lx: new Float32Array(n), ly: new Float32Array(n), wx: new Float32Array(n), wy: new Float32Array(n),
      r: 0, area: 0, im: 0, ii: 0, h, sub: false, sleep: false, still: 0, stuck: false, dead: false, tint: 0, flip: rnd() < 0.45 ? 1 : 2, latK: 1.6 * rnd(), tSub: -9, touch: -9 };
    let r2 = 0, I = 0;
    for (let i = 0; i < n; i++) { const q = A < 0 ? n - 1 - i : i, x = (xs[q] - cx) * shrink, y = (ys[q] - cy) * shrink; p.lx[i] = x; p.ly[i] = y; p.wx[i] = cx + x; p.wy[i] = cy + y; if (x * x + y * y > r2) r2 = x * x + y * y; }
    for (let i = 0, j = n - 1; i < n; j = i++) { const c = p.lx[j] * p.ly[i] - p.lx[i] * p.ly[j]; I += c * (p.lx[j] * p.lx[j] + p.lx[j] * p.lx[i] + p.lx[i] * p.lx[i] + p.ly[j] * p.ly[j] + p.ly[j] * p.ly[i] + p.ly[i] * p.ly[i]); }
    p.area = Math.abs(A) / 2 * shrink * shrink; p.r = Math.sqrt(r2);
    p.im = 1 / (sim.P.rhoI * h * p.area); p.ii = 12 / (sim.P.rhoI * h * Math.abs(I));
    sim.pieces.push(p); return p;
  }
  function updateVerts(p) { const c = Math.cos(p.th), s = Math.sin(p.th); for (let i = 0; i < p.n; i++) { p.wx[i] = p.x + c * p.lx[i] - s * p.ly[i]; p.wy[i] = p.y + s * p.lx[i] + c * p.ly[i]; } }
  function inside(p, x, y) { for (let i = 0, j = p.n - 1; i < p.n; j = i++) if ((p.wx[i] - p.wx[j]) * (y - p.wy[j]) - (p.wy[i] - p.wy[j]) * (x - p.wx[j]) < 0) return false; return true; }

  const CT = { nx: 0, ny: 0, d: 0, x: 0, y: 0 };
  function sat(a, b) {             // separating-axis test for two convex pieces
    const dx = b.x - a.x, dy = b.y - a.y, rr = a.r + b.r; if (dx * dx + dy * dy >= rr * rr) return false;
    let best = 1e9, bnx = 0, bny = 0;
    for (let pass = 0; pass < 2; pass++) {
      const Q = pass ? b : a, X = Q.wx, Y = Q.wy;
      for (let i = 0, j = Q.n - 1; i < Q.n; j = i++) {
        const ex = X[i] - X[j], ey = Y[i] - Y[j], l = Math.sqrt(ex * ex + ey * ey); if (l < 1e-6) continue;
        const nx = ey / l, ny = -ex / l; let a0 = 1e9, a1 = -1e9, b0 = 1e9, b1 = -1e9;
        for (let k = 0; k < a.n; k++) { const t = a.wx[k] * nx + a.wy[k] * ny; if (t < a0) a0 = t; if (t > a1) a1 = t; }
        for (let k = 0; k < b.n; k++) { const t = b.wx[k] * nx + b.wy[k] * ny; if (t < b0) b0 = t; if (t > b1) b1 = t; }
        const ov = Math.min(a1, b1) - Math.max(a0, b0); if (ov <= 0) return false;
        if (ov < best) { best = ov; bnx = nx; bny = ny; }
      }
    }
    if (bnx * dx + bny * dy < 0) { bnx = -bnx; bny = -bny; }
    let ib = 0, tb = 1e9, ia = 0, ta = -1e9;
    for (let k = 0; k < b.n; k++) { const t = b.wx[k] * bnx + b.wy[k] * bny; if (t < tb) { tb = t; ib = k; } }
    for (let k = 0; k < a.n; k++) { const t = a.wx[k] * bnx + a.wy[k] * bny; if (t > ta) { ta = t; ia = k; } }
    if (inside(a, b.wx[ib], b.wy[ib])) { CT.x = b.wx[ib]; CT.y = b.wy[ib]; }
    else if (inside(b, a.wx[ia], a.wy[ia])) { CT.x = a.wx[ia]; CT.y = a.wy[ia]; }
    else { CT.x = 0.5 * (b.wx[ib] + a.wx[ia]); CT.y = 0.5 * (b.wy[ib] + a.wy[ia]); }
    CT.nx = bnx; CT.ny = bny; CT.d = best; return true;
  }
  function nudge(o, dx, dy) { o.x += dx; o.y += dy; if (o.wx) for (let k = 0; k < o.n; k++) { o.wx[k] += dx; o.wy[k] += dy; } }
  function resolve(a, b, nx, ny, depth, cx, cy, mu, beta) {   // normal from a to b
    const rax = cx - a.x, ray = cy - a.y, rbx = cx - b.x, rby = cy - b.y;
    const rvx = (b.vx - b.w * rby) - (a.vx - a.w * ray), rvy = (b.vy + b.w * rbx) - (a.vy + a.w * rax), vn = rvx * nx + rvy * ny;
    const ran = rax * ny - ray * nx, rbn = rbx * ny - rby * nx, k = a.im + b.im + a.ii * ran * ran + b.ii * rbn * rbn; if (k <= 0) return;
    if (vn < 0) {
      const j = -vn / k;
      a.vx -= j * nx * a.im; a.vy -= j * ny * a.im; a.w -= j * ran * a.ii; b.vx += j * nx * b.im; b.vy += j * ny * b.im; b.w += j * rbn * b.ii;
      const tx = -ny, ty = nx, vt = rvx * tx + rvy * ty, rat = rax * ty - ray * tx, rbt = rbx * ty - rby * tx, kt = a.im + b.im + a.ii * rat * rat + b.ii * rbt * rbt;
      const jt = clamp(-vt / kt, -mu * j, mu * j);
      a.vx -= jt * tx * a.im; a.vy -= jt * ty * a.im; a.w -= jt * rat * a.ii; b.vx += jt * tx * b.im; b.vy += jt * ty * b.im; b.w += jt * rbt * b.ii;
    }
    const c = clamp(depth - 0.03, 0, 0.6) * beta / (a.im + b.im);
    if (c > 0) { if (a.im) nudge(a, -c * a.im * nx, -c * a.im * ny); if (b.im) nudge(b, c * b.im * nx, c * b.im * ny); }
  }

  function pair(a, b, bAsleep) {
    if (!sat(a, b)) return;
    if (!bAsleep) { resolve(a, b, CT.nx, CT.ny, CT.d, CT.x, CT.y, 0.3, 0.5); return; }
    if (a.vx * CT.nx + a.vy * CT.ny > 0.2) { b.sleep = false; b.still = 0; resolve(a, b, CT.nx, CT.ny, CT.d, CT.x, CT.y, 0.3, 0.5); return; }
    const im = b.im, ii = b.ii; b.im = 0; b.ii = 0; resolve(a, b, CT.nx, CT.ny, CT.d, CT.x, CT.y, 0.3, 0.5); b.im = im; b.ii = ii;   // a resting piece acts as an obstacle
  }

  // all[0..na) are awake; the rest are sleeping pieces nearby, which only act as obstacles until hit hard enough
  function collide(sim, all, na, minx, miny, maxx, maxy) {
    const N = all.length; let rs = 0; for (let i = 0; i < N; i++) rs += all[i].r;
    const cell = clamp(2.6 * rs / N, 3, 30), half = 0.5 * cell, gw = Math.floor((maxx - minx) / cell) + 1, gh = Math.floor((maxy - miny) / cell) + 1, ng = gw * gh;
    if (sim.gcount.length < ng + 1) { sim.gcount = new Int32Array(2 * ng + 1); sim.gfill = new Int32Array(2 * ng + 1); }
    if (sim.gitem.length < N) { sim.gitem = new Int32Array(2 * N); sim.gcell = new Int32Array(2 * N); }
    const cnt = sim.gcount, fill = sim.gfill, item = sim.gitem, gc = sim.gcell, big = sim.big; cnt.fill(0, 0, ng + 1); fill.fill(0, 0, ng); big.length = 0;
    for (let i = 0; i < N; i++) { const p = all[i]; if (p.r > half) { gc[i] = -1; big.push(i); continue; } const c = Math.floor((p.y - miny) / cell) * gw + Math.floor((p.x - minx) / cell); gc[i] = c; cnt[c + 1]++; }
    for (let c = 0; c < ng; c++) cnt[c + 1] += cnt[c];
    for (let i = 0; i < N; i++) { const c = gc[i]; if (c >= 0) item[cnt[c] + fill[c]++] = i; }
    for (let i = 0; i < na; i++) {
      const c = gc[i]; if (c < 0) continue; const a = all[i], ci = c % gw, cj = (c / gw) | 0;
      for (let oj = -1; oj <= 1; oj++) { const j2 = cj + oj; if (j2 < 0 || j2 >= gh) continue;
        for (let oi = -1; oi <= 1; oi++) { const i2 = ci + oi; if (i2 < 0 || i2 >= gw) continue; const c2 = j2 * gw + i2;
          for (let q = cnt[c2]; q < cnt[c2 + 1]; q++) { const bi = item[q]; if (bi === i || (bi < na && bi < i)) continue; pair(a, all[bi], bi >= na); }
        } }
    }
    for (let q = 0; q < big.length; q++) {       // the few large pieces are tested against everything
      const bq = big[q], A = all[bq];
      for (let j = 0; j < N; j++) {
        if (j === bq || (gc[j] < 0 && j < bq) || (bq >= na && j >= na)) continue;
        if (bq < na) pair(A, all[j], j >= na); else pair(all[j], A, true);
      }
    }
  }

  function stepPieces(sim, dt) {
    const sh = sim.ship, H = sh.hull, pcs = sim.pieces, sheets = sim.sheets, act = sim.act, px = sim.proxy;
    const cs = Math.cos(sh.th), sn = Math.sin(sh.th);
    px.x = sh.x; px.y = sh.y; px.vx = sh.u * cs - sh.v * sn; px.vy = sh.u * sn + sh.v * cs; px.w = sh.r;
    const u0 = sh.u, washX = -sh.L / 2;
    let subArea = 0, minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9; act.length = 0; buildSheetGrid(sim);
    for (let q = 0; q < pcs.length; q++) {
      const p = pcs[q], dx = p.x - sh.x, dy = p.y - sh.y, d2 = dx * dx + dy * dy, nr = H.R + p.r + 2, near = d2 < nr * nr;
      if (p.sleep) { if (near && !p.stuck) { p.sleep = false; p.still = 0; } else continue; }
      const bx = cs * dx + sn * dy, by = -sn * dx + cs * dy; let hidden = false;
      if (p.sub) {
        const hs = near ? H.sd(bx, by) : 9;
        if (hs < 0.5 * p.r + 0.3) {              // under the hull: it stays roughly where it is while the ship passes over
          if (hs < 0) { subArea += p.area; hidden = true; }
          const lat = (by >= 0 ? 1 : -1) * 0.05 * Math.abs(sh.u) * p.latK, k = Math.min(1, 3 * dt);
          p.vx += (0.1 * (px.vx - px.w * dy) - sn * lat - p.vx) * k; p.vy += (0.1 * (px.vy + px.w * dx) + cs * lat - p.vy) * k;
        } else if (sim.t - p.tSub > 1) {         // clear of the hull: comes back up unless a sheet is on top of it
          let under = null; const cand = sheetsNear(sim, p.x, p.y), nc = cand ? cand.length : sheets.length;
          for (let k = 0; k < nc; k++) { const S = sheets[cand ? cand[k] : k], ex = p.x - S.x, ey = p.y - S.y; if (ex * ex + ey * ey > S.R * S.R) continue; S.toLocal(p.x, p.y, T2); if (S.sample(T2[0], T2[1]) < -0.3) { under = S; break; } }
          if (!under) { p.sub = false; p.tint = p.flip; }
          else if (under.im === 0) { p.dead = true; sim.dirtyPieces = true; continue; }
          else { const ex = p.x - under.x, ey = p.y - under.y, k = Math.min(1, 2 * dt); p.vx += (under.vx - under.w * ey - p.vx) * k; p.vy += (under.vy + under.w * ex - p.vy) * k; hidden = true; }
        }
      }
      if (sh.thr > 0.05 && near && bx < washX + 2 && bx > washX - 30 && Math.abs(by) < 6) {   // propeller wash
        const f = sh.thr * (1 - (washX - bx) / 30) * (1 - Math.abs(by) / 6), k = Math.min(1, 2 * dt * f), tx = -4 * f, ty = (by >= 0 ? 1.2 : -1.2) * f;
        p.vx += (cs * tx - sn * ty - p.vx) * k; p.vy += (sn * tx + cs * ty - p.vy) * k;
      }
      const sp = Math.hypot(p.vx, p.vy), f = 1 / (1 + (0.8 + 0.6 * sp) * dt); p.vx *= f; p.vy *= f; p.w /= 1 + 1.2 * dt;
      p.x += p.vx * dt; p.y += p.vy * dt; p.th += p.w * dt; updateVerts(p);
      if (!near && sp < 0.05 && Math.abs(p.w) < 0.02) { p.still += dt; if (p.still > 0.5) { p.sleep = true; p.vx = p.vy = p.w = 0; continue; } } else p.still = 0;
      if (hidden) continue;                      // out of sight under the hull or a floe: no collisions
      act.push(p);
      if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x; if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y;
    }
    sim.subArea = subArea;
    const na = act.length; sim.nAwake = na;
    if (na) {
      minx -= 4; miny -= 4; maxx += 4; maxy += 4;
      for (let q = 0; q < pcs.length; q++) { const p = pcs[q]; if (!p.sleep || p.stuck || p.sub || p.x < minx || p.x > maxx || p.y < miny || p.y > maxy) continue; act.push(p); }
      collide(sim, act, na, minx, miny, maxx, maxy);
    }
    // pieces against sheets, then against the hull
    for (let i = 0; i < na; i++) {
      const p = act[i]; if (p.sub) continue; const cand = sheetsNear(sim, p.x, p.y), nc = cand ? cand.length : sheets.length;
      for (let k = 0; k < nc; k++) {
        const S = sheets[cand ? cand[k] : k], ex = p.x - S.x, ey = p.y - S.y, rr = S.R + p.r; if (ex * ex + ey * ey > rr * rr) continue;
        let dm = 0.04, mx = 0, my = 0, mlx = 0, mly = 0;
        for (let v = 0; v < p.n; v++) { S.toLocal(p.wx[v], p.wy[v], T2); const d = -S.sample(T2[0], T2[1]); if (d > dm) { dm = d; mx = p.wx[v]; my = p.wy[v]; mlx = T2[0]; mly = T2[1]; } }
        if (dm > 0.04) {
          S.gradW(mlx, mly, G2); resolve(S, p, G2[0], G2[1], dm, mx, my, 0.3, 0.5); p.touch = sim.n; if (S.im) { S.asleep = false; S.still = 0; }
          S.toLocal(p.x, p.y, T2); if (-S.sample(T2[0], T2[1]) > 0.5 * p.r + 0.3) { p.sub = true; p.tSub = sim.t - 2; break; }   // mostly under the sheet already
        }
      }
    }
    const HR = H.R, key = H.key;
    for (let i = 0; i < na; i++) {
      const p = act[i]; if (p.sub) continue;
      const dx = p.x - sh.x, dy = p.y - sh.y, rr = HR + p.r; if (dx * dx + dy * dy > rr * rr) continue;
      let dm = 0.02, kb = -1, mbx = 0, mby = 0;
      for (let v = 0; v < p.n; v++) { const ex = p.wx[v] - sh.x, ey = p.wy[v] - sh.y, bx = cs * ex + sn * ey, by = -sn * ex + cs * ey, d = -H.sd(bx, by); if (d > dm) { dm = d; kb = v; mbx = bx; mby = by; } }
      let nx = 0, ny = 0, cx = 0, cy = 0;
      if (kb >= 0) { H.grad(mbx, mby, G2); nx = cs * G2[0] - sn * G2[1]; ny = sn * G2[0] + cs * G2[1]; cx = p.wx[kb]; cy = p.wy[kb]; }
      else {             // the stem can poke a piece between its corners
        dm = 0;
        for (let q = 0; q < key.length; q++) {
          const h = key[q], hx = sh.x + cs * H.px[h] - sn * H.py[h], hy = sh.y + sn * H.px[h] + cs * H.py[h]; if (!inside(p, hx, hy)) continue;
          let dmin = 1e9; for (let v = 0, j = p.n - 1; v < p.n; j = v++) { const ex = p.wx[v] - p.wx[j], ey = p.wy[v] - p.wy[j], l = Math.hypot(ex, ey) || 1, d = (ex * (hy - p.wy[j]) - ey * (hx - p.wx[j])) / l; if (d < dmin) dmin = d; }
          if (dmin > dm) { dm = dmin; nx = cs * H.nx[h] - sn * H.ny[h]; ny = sn * H.nx[h] + cs * H.ny[h]; cx = hx; cy = hy; }
        }
        if (dm <= 0) continue;
      }
      const bx = cs * dx + sn * dy, by = -sn * dx + cs * dy;
      // squeezed between the hull and intact ice, or overrun: it goes under instead of fighting the hull
      if (sim.n - p.touch < 13 || dm > 0.5 || H.sd(bx, by) < 0) { p.sub = true; p.tSub = sim.t; p.w += (sim.rand() - 0.5) * 0.8; continue; }
      resolve(px, p, nx, ny, dm, cx, cy, 0.12, 0.6);
    }
    sh.u = px.vx * cs + px.vy * sn; sh.v = -px.vx * sn + px.vy * cs; sh.r = px.w;
    sim.accX += (sh.u - u0) * sh.mx / sim.dt;       // momentum given to floes counts as ice resistance
  }

  // ------------------------------------------------------------------ stepping
  function processPending(sim) {
    const pend = sim.pending;
    for (let i = 0; i < pend.length; i++) { const ev = pend[i]; if (ev.S.dead) continue; if (ev.type === 0) breakCusp(sim, ev.S, ev.lx, ev.ly, ev.R); else split(sim, ev.S, ev); }
    pend.length = 0;
    if (sim.dirtySheets) { sim.sheets = sim.sheets.filter(s => !s.dead); sim.dirtySheets = false; }
  }

  function housekeeping(sim) {
    const list = sim.sheets.slice();
    for (const S of list) {
      if (S.topo && S.im > 0 && !S.dead) separate(sim, S, null);
      if (S.crackDirty && S.cracks.length) {
        const c = S.cracks; let w = 0;
        for (let i = 0; i < c.length; i += 4) if (S.sample((c[i] + c[i + 2]) / 2, (c[i + 1] + c[i + 3]) / 2) < -0.15 && S.sample(c[i + 2], c[i + 3]) < -0.05 && S.sample(c[i], c[i + 1]) < 0.3) { c[w++] = c[i]; c[w++] = c[i + 1]; c[w++] = c[i + 2]; c[w++] = c[i + 3]; }
        c.length = w;
      }
      S.crackDirty = false;
    }
    if (sim.dirtySheets) { sim.sheets = sim.sheets.filter(s => !s.dead); sim.dirtySheets = false; }
    const pcs = sim.pieces;
    for (const S of sim.sheets) {     // a drifting floe wakes the brash it runs into
      if (S.im === 0 || S.asleep) continue;
      for (let i = 0; i < pcs.length; i++) { const p = pcs[i]; if (!p.sleep || p.stuck) continue; const dx = p.x - S.x, dy = p.y - S.y, rr = S.R + p.r; if (dx * dx + dy * dy < rr * rr) { p.sleep = false; p.still = 0; } }
    }
    if (pcs.length > sim.o.maxPieces) { let extra = pcs.length - sim.o.maxPieces; for (let i = 0; i < pcs.length && extra > 0; i++) if (pcs[i].sleep) { pcs[i].dead = true; extra--; } sim.dirtyPieces = true; }
    if (sim.dirtyPieces) { sim.pieces = pcs.filter(p => !p.dead); sim.dirtyPieces = false; }
  }

  function step(sim) {
    const dt = sim.dt; sim.n++;
    hullIce(sim);
    if ((sim.n & 1) === 0) sheetContacts(sim);
    integrateShip(sim, dt);
    integrateSheets(sim, dt);
    if (sim.pending.length) processPending(sim);
    if (sim.n % 6 === 0) stepPieces(sim, 6 * dt);
    if (sim.n % 144 === 0) housekeeping(sim);
    sim.t += dt;
  }
  function advance(sim, T) {
    sim.acc += T; let k = 0;
    while (sim.acc >= sim.dt && k < 80) { step(sim); sim.acc -= sim.dt; k++; }
    if (k === 80) sim.acc = 0;
    return k;
  }

  // ------------------------------------------------------------------ scenarios
  function clipHalf(poly, px, py, nx, ny) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length], da = (a[0] - px) * nx + (a[1] - py) * ny, db = (b[0] - px) * nx + (b[1] - py) * ny;
      if (da <= 0) out.push(a);
      if ((da < 0 && db > 0) || (da > 0 && db < 0)) { const t = da / (da - db); out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]); }
    }
    return out;
  }
  function polyAC(poly) {
    let A = 0, cx = 0, cy = 0;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const c = poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1]; A += c; cx += (poly[j][0] + poly[i][0]) * c; cy += (poly[j][1] + poly[i][1]) * c; }
    return { area: A / 2, cx: cx / (3 * A), cy: cy / (3 * A) };
  }
  function warpXY(sim, x, y, amp, out) {   // one smooth displacement field shared by all floes, so neighbouring edges stay parallel
    const ws = sim.o.seed * 7 + 100;
    out[0] = x + amp * (vnoise(x / 46, y / 46, ws) + 0.22 * vnoise(x / 12, y / 12, ws + 2));
    out[1] = y + amp * (vnoise(x / 46, y / 46, ws + 1) + 0.22 * vnoise(x / 12, y / 12, ws + 3));
  }
  function sheetFromPoly(sim, poly, rough, amp) {
    const o = sim.o, c = o.cell, pr = polyAC(poly); if (pr.area < 0) poly = poly.slice().reverse();
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const p of poly) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
    const pad = 3 * c + 2 * rough + 1.3 * amp, ox = x0 - pad, oy = y0 - pad, nx = Math.ceil((x1 - x0 + 2 * pad) / c) + 1, ny = Math.ceil((y1 - y0 + 2 * pad) / c) + 1;
    const S = new Sheet(nx, ny, c, sim.P.h, (sim.rand() * 1e6) | 0), E = [], raw = new Float32Array(nx * ny);
    for (let i = 0; i < poly.length; i++) { const a = poly[i], b = poly[(i + 1) % poly.length], ex = b[0] - a[0], ey = b[1] - a[1], l = Math.hypot(ex, ey) || 1; E.push(ey / l, -ex / l, (ey * a[0] - ex * a[1]) / l); }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      warpXY(sim, ox + i * c, oy + j * c, amp, T2); const x = T2[0], y = T2[1]; let d = -1e9;
      for (let e = 0; e < E.length; e += 3) { const v = E[e] * x + E[e + 1] * y - E[e + 2]; if (v > d) d = v; }
      raw[j * nx + i] = d + rough * (vnoise(x / 9, y / 9, S.seed) + 0.4 * vnoise(x / 3.3, y / 3.3, S.seed + 1));
    }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {     // the warp stretches distances: divide by the local slope
      const k = j * nx + i, i0 = i > 0 ? k - 1 : k, i1 = i < nx - 1 ? k + 1 : k, j0 = j > 0 ? k - nx : k, j1 = j < ny - 1 ? k + nx : k;
      const gx = (raw[i1] - raw[i0]) / ((i1 - i0) * c), gy = (raw[j1] - raw[j0]) / ((j1 - j0) / nx * c), g = Math.hypot(gx, gy);
      S.sdf[k] = raw[k] / clamp(g, 0.5, 2);
    }
    S.x = ox; S.y = oy; S.hox = ox; S.hoy = oy; S.recompute(sim.P);
    return S;
  }
  function randomPiece(sim, x, y, size) {
    const rnd = sim.rand, n = 5 + ((rnd() * 3) | 0), xs = [], ys = [], a0 = rnd() * TAU;
    for (let i = 0; i < n; i++) { const a = a0 + TAU * (i + 0.5 * rnd()) / n, r = size * (0.65 + 0.35 * rnd()); xs.push(x + r * Math.cos(a)); ys.push(y + r * Math.sin(a)); }
    const p = addPiece(sim, xs, ys, sim.P.h, 1); if (p) p.tint = rnd() < 0.3 ? 2 : 0; return p;
  }

  function buildLevel(sim) {
    const o = sim.o, c = o.cell, W = 1100, Hh = 440, nx = Math.round(W / c) + 1, ny = Math.round(Hh / c) + 1, S = new Sheet(nx, ny, c, sim.P.h, o.seed * 13 + 5);
    for (let j = 0; j < ny; j++) {
      const y = j * c, xe = 150 + 7 * vnoise(y / 38, 0.5, o.seed) + 2.5 * vnoise(y / 9, 3.3, o.seed + 5);
      for (let i = 0; i < nx; i++) { const x = i * c; S.sdf[j * nx + i] = Math.max(xe - x, x - (W - 3), 3 - y, y - (Hh - 3)); }
    }
    S.fixed = true; S.R = Math.hypot(W, Hh); S.area = W * Hh;
    S.ridges = [{ x1: 420, y1: 0, x2: 505, y2: Hh, w: 7, gain: 0.7 }, { x1: 790, y1: 0, x2: 730, y2: Hh, w: 9, gain: 0.85 }];
    sim.sheets.push(S); sim.ship.x = 70; sim.ship.y = Hh / 2;
  }

  function buildFloes(sim, pack) {
    const o = sim.o, rnd = sim.rand, X0 = 170, X1 = pack ? 900 : 1150, Y1 = pack ? 400 : 480;
    const cutTo = (poly, target, sink) => {
      const pr = polyAC(poly), A = Math.abs(pr.area);
      if (A < target * (0.7 + 0.9 * rnd())) { sink.push(poly); return; }
      let bi = 0, bj = 1, bd = 0;
      for (let i = 0; i < poly.length; i++) for (let j = i + 1; j < poly.length; j++) { const d = (poly[i][0] - poly[j][0]) ** 2 + (poly[i][1] - poly[j][1]) ** 2; if (d > bd) { bd = d; bi = i; bj = j; } }
      const ang = Math.atan2(poly[bj][1] - poly[bi][1], poly[bj][0] - poly[bi][0]) + PI / 2 + (rnd() - 0.5) * 0.9;
      const px = pr.cx + (rnd() - 0.5) * 0.25 * Math.sqrt(A), py = pr.cy + (rnd() - 0.5) * 0.25 * Math.sqrt(A), nx = -Math.sin(ang), ny = Math.cos(ang);
      const a = clipHalf(poly, px, py, nx, ny), b = clipHalf(poly, px, py, -nx, -ny);
      if (a.length < 3 || b.length < 3) { sink.push(poly); return; }
      cutTo(a, target, sink); cutTo(b, target, sink);
    };
    const rect = [[X0, 0], [X1, 0], [X1, Y1], [X0, Y1]], leaves = [], amp = pack ? 3.2 : 6.5;
    if (pack) { const reg = []; cutTo(rect, 9000, reg); for (const r of reg) cutTo(r, 260 * Math.pow(11, rnd()), leaves); }
    else cutTo(rect, 8500, leaves);
    for (const leaf of leaves) {
      if (rnd() < 0.05) continue;
      const pr = polyAC(leaf), A = Math.abs(pr.area), gap = pack ? 2.4 + 2.6 * rnd() : 4.5 + 5 * rnd(), s = 1 - 1.1 * gap / Math.sqrt(A); if (s < 0.3) continue;
      const poly = leaf.map(p => [pr.cx + (p[0] - pr.cx) * s, pr.cy + (p[1] - pr.cy) * s]);
      if (A * s * s < o.minSheetArea * 1.3) { addPiece(sim, poly.map(p => p[0]), poly.map(p => p[1]), sim.P.h, 1); continue; }
      sim.sheets.push(sheetFromPoly(sim, poly, pack ? 0.35 : 0.6, amp));
    }
    let brash = 0; const maxBrash = pack ? 520 : 420;      // small ice in the leads
    for (const leaf of leaves) for (let i = 0; i < leaf.length && brash < maxBrash; i++) {
      const a = leaf[i], b = leaf[(i + 1) % leaf.length], len = Math.hypot(b[0] - a[0], b[1] - a[1]), k = Math.floor(len / (pack ? 9 : 6) * rnd() * 1.4);
      for (let q = 0; q < k; q++) {
        const t = rnd(), size = 0.5 + 1.6 * rnd() * rnd(), ex = a[0] + (b[0] - a[0]) * t, ey = a[1] + (b[1] - a[1]) * t; warpXY(sim, ex, ey, amp, T2);
        const x = 2 * ex - T2[0] + (rnd() - 0.5) * 3, y = 2 * ey - T2[1] + (rnd() - 0.5) * 3; let ok = x > X0 - 5;
        for (const S of sim.sheets) { const dx = x - S.x, dy = y - S.y; if (dx * dx + dy * dy > S.R * S.R) continue; S.toLocal(x, y, T2); if (S.sample(T2[0], T2[1]) < size) { ok = false; break; } }
        if (ok && randomPiece(sim, x, y, size)) brash++;
      }
    }
    sim.ship.x = 70; sim.ship.y = Y1 / 2;
  }

  // ------------------------------------------------------------------ an endless field around the ship
  // The world is cut into square chunks of CH metres. A chunk is made when the ship comes near and dropped when it is far away. What it
  // holds is asked from field.at(x, y) -> { conc, h } at its centre: one fixed sheet of level ice (concentration of 0.895 or more), floes
  // that cover that share of the water (down to 0.08), or nothing. Neighbouring sheets of level ice touch, so the ice is continuous;
  // the only trace of the cut is that a cusp breaks a little more easily right at a chunk's border.
  const CH = 256;
  function floesIn(sim, X0, Y0, X1, Y1, conc, pack) {   // as buildFloes, in any rectangle and for any share of ice; the floes are larger, to keep their number down
    const o = sim.o, rnd = sim.rand, leaves = [], amp = pack ? 3.2 : 6.5;
    const cutTo = (poly, target, sink) => {
      const pr = polyAC(poly), A = Math.abs(pr.area);
      if (A < target * (0.7 + 0.9 * rnd())) { sink.push(poly); return; }
      let bi = 0, bj = 1, bd = 0;
      for (let i = 0; i < poly.length; i++) for (let j = i + 1; j < poly.length; j++) { const d = (poly[i][0] - poly[j][0]) ** 2 + (poly[i][1] - poly[j][1]) ** 2; if (d > bd) { bd = d; bi = i; bj = j; } }
      const ang = Math.atan2(poly[bj][1] - poly[bi][1], poly[bj][0] - poly[bi][0]) + PI / 2 + (rnd() - 0.5) * 0.9;
      const px = pr.cx + (rnd() - 0.5) * 0.25 * Math.sqrt(A), py = pr.cy + (rnd() - 0.5) * 0.25 * Math.sqrt(A), nx = -Math.sin(ang), ny = Math.cos(ang);
      const a = clipHalf(poly, px, py, nx, ny), b = clipHalf(poly, px, py, -nx, -ny);
      if (a.length < 3 || b.length < 3) { sink.push(poly); return; }
      cutTo(a, target, sink); cutTo(b, target, sink);
    };
    const rect = [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]], first = sim.sheets.length;
    if (pack) { const reg = []; cutTo(rect, 14000, reg); for (const r of reg) cutTo(r, 700 * Math.pow(8, rnd()), leaves); } else cutTo(rect, 8500, leaves);
    for (const leaf of leaves) {
      const pr = polyAC(leaf), A = Math.abs(pr.area), s = Math.min(0.975, Math.sqrt(conc) * (0.95 + 0.1 * rnd()));   // each floe shrunk so that floes cover `conc` of the water
      const poly = leaf.map(p => [pr.cx + (p[0] - pr.cx) * s, pr.cy + (p[1] - pr.cy) * s]);
      if (A * s * s < o.minSheetArea * 1.3) { addPiece(sim, poly.map(p => p[0]), poly.map(p => p[1]), sim.P.h, 1); continue; }
      sim.sheets.push(sheetFromPoly(sim, poly, pack ? 0.35 : 0.6, amp));
    }
    let brash = 0; const maxBrash = Math.round(24 * conc);      // small ice in the leads
    for (const leaf of leaves) for (let i = 0; i < leaf.length && brash < maxBrash; i++) {
      const a = leaf[i], b = leaf[(i + 1) % leaf.length], len = Math.hypot(b[0] - a[0], b[1] - a[1]), k = Math.floor(len / (pack ? 9 : 6) * rnd() * 1.4);
      for (let q = 0; q < k && brash < maxBrash; q++) {
        const t = rnd(), size = 0.5 + 1.6 * rnd() * rnd(), ex = a[0] + (b[0] - a[0]) * t, ey = a[1] + (b[1] - a[1]) * t; warpXY(sim, ex, ey, amp, T2);
        const x = 2 * ex - T2[0] + (rnd() - 0.5) * 3, y = 2 * ey - T2[1] + (rnd() - 0.5) * 3; let ok = true;
        for (let m = first; m < sim.sheets.length; m++) { const S = sim.sheets[m], dx = x - S.x, dy = y - S.y; if (dx * dx + dy * dy > S.R * S.R) continue; S.toLocal(x, y, T2); if (S.sample(T2[0], T2[1]) < size) { ok = false; break; } }
        if (ok && randomPiece(sim, x, y, size)) brash++;
      }
    }
  }
  function makeChunk(sim, ci, cj) {
    const f = sim.field, x0 = ci * CH, y0 = cj * CH, q = f.at(x0 + CH / 2, y0 + CH / 2), c = sim.o.cell;
    const seed = (Math.imul(ci, 73856093) ^ Math.imul(cj, 19349663) ^ Math.imul(sim.o.seed, 83492791)) >>> 0;
    if (!(q.conc >= 0.08) || !(q.h > 0.02)) return;
    if (q.conc >= 0.895) {                      // level ice: a fixed sheet that fills the chunk
      const n = Math.round(CH / c) + 1, S = new Sheet(n, n, c, q.h, sim.o.seed * 13 + 5);
      S.sdf.fill(-50); S.fixed = true; S.x = x0; S.y = y0; S.hox = x0; S.hoy = y0; S.R = Math.hypot(CH, CH); S.area = CH * CH; sim.sheets.push(S); return;
    }
    const keep = sim.rand, h0 = sim.P.h; sim.rand = makeRng(seed); sim.P.h = q.h;
    floesIn(sim, x0, y0, x0 + CH, y0 + CH, q.conc, f.pack);
    sim.rand = keep; sim.P.h = h0;
  }
  // Keep the field around the ship: make the nearest missing chunk (all of them with `all`), and now and then drop what is far away.
  // Call it once per frame; it costs nothing when there is nothing to do.
  function follow(sim, all) {
    const f = sim.field; if (!f) return;
    const sh = sim.ship, R = f.radius;
    // Level ice is kept within R of the ship. Floes are kept within a shorter distance, because each one is a body of its own.
    const near = (ci, cj) => { const dx = Math.max(ci * CH - sh.x, 0, sh.x - (ci + 1) * CH), dy = Math.max(cj * CH - sh.y, 0, sh.y - (cj + 1) * CH); return Math.hypot(dx, dy); };
    const solid = (ci, cj) => f.at((ci + 0.5) * CH, (cj + 0.5) * CH).conc >= 0.895;
    for (;;) {
      let best = null, bd = 1e18;
      for (let cj = Math.floor((sh.y - R) / CH); cj <= Math.floor((sh.y + R) / CH); cj++) for (let ci = Math.floor((sh.x - R) / CH); ci <= Math.floor((sh.x + R) / CH); ci++) {
        if (f.chunks.has(ci + ',' + cj)) continue; const d = near(ci, cj); if (d > (solid(ci, cj) ? R : f.floeRadius)) continue; if (d < bd) { bd = d; best = [ci, cj]; }
      }
      if (!best) break;
      f.chunks.add(best[0] + ',' + best[1]); makeChunk(sim, best[0], best[1]);
      if (!all) break;
    }
    if (sim.n - f.swept < 720) return;          // every two seconds of ship time
    f.swept = sim.n; const gone = (x, y, lim) => { const dx = x - sh.x, dy = y - sh.y; return dx * dx + dy * dy > lim * lim; };
    for (const S of sim.sheets) if (S.fixed ? gone(S.x + CH / 2, S.y + CH / 2, R + 1.1 * CH) : gone(S.x, S.y, f.floeRadius + 0.9 * CH)) { S.dead = true; sim.dirtySheets = true; }
    for (const p of sim.pieces) if (gone(p.x, p.y, 460)) { p.dead = true; sim.dirtyPieces = true; }
    for (const key of f.chunks) { const q = key.split(','), ci = +q[0], cj = +q[1]; if (near(ci, cj) > (solid(ci, cj) ? R + 0.6 * CH : f.floeRadius + 0.4 * CH)) f.chunks.delete(key); }
    if (sim.dirtySheets) { sim.sheets = sim.sheets.filter(s => !s.dead); sim.dirtySheets = false; }
    if (sim.dirtyPieces) { sim.pieces = sim.pieces.filter(p => !p.dead); sim.dirtyPieces = false; }
  }
  // The ship starts inside the ice as if it had been sailing through it: its own outline is cut out, a channel with ragged edges runs
  // astern with brash in it, and broken ice lies under the hull (without it the first seconds would be too easy: see `cover`).
  function seat(sim) {
    const sh = sim.ship, o = sh.o, L = o.L, hb = o.B / 2, Lb = o.bowFrac * L, xb = L / 2, rnd = sim.rand, back = 380;
    const half = x => (x > xb - Lb ? hb * (1 - Math.pow((x - (xb - Lb)) / Lb, o.bowPow)) : hb);
    const cut = (x, y, R) => { for (const S of sim.sheets) { const dx = S.x - x, dy = S.y - y, rr = S.R + R; if (dx * dx + dy * dy > rr * rr) continue; S.toLocal(x, y, T2); S.carve(T2[0], T2[1], R, 0, 0, 0); if (S.im > 0) S.topo = true; } };
    const here = sim.field.at(0, 0), ice = here.conc >= 0.08 && here.h > 0.02;
    for (let x = -xb - back; x < xb; x += 1.5) cut(x, 0, (x < -xb ? hb + 1.2 : half(x) + 1.2));                       // the channel and the hull's own place
    for (let x = -xb - back; x < xb - Lb; x += 2.4) for (const side of [-1, 1]) cut(x + 2 * rnd(), side * (hb + 0.4 + 1.8 * rnd()), 1.4 + 2.4 * rnd());   // cusps along its edges
    if (!ice) return;
    const share = Math.min(1, here.conc / 0.9);
    for (let k = 0; k < 150 * share; k++) randomPiece(sim, -xb - back * rnd() * rnd(), (rnd() - 0.5) * 2 * (hb - 1), 0.6 + 1.5 * rnd() * rnd());   // brash in the channel
    let area = 0; const want = 0.63 * sh.hull.area * share;
    for (let k = 0; k < 400 && area < want; k++) {
      const x = -xb + 3 + (L - 14) * rnd(), p = randomPiece(sim, x, (rnd() - 0.5) * 1.7 * half(x), 1.3 + 1.2 * rnd()); if (!p) continue;
      p.sub = true; p.tSub = sim.t; area += p.area;
    }
    sim.cover = 0.9 * share;
  }
  function buildField(sim) {
    const f = sim.o.field || {};
    sim.field = { at: f.at || (() => ({ conc: 1, h: sim.P.h })), pack: f.pack !== false, radius: f.radius || 420, floeRadius: f.floeRadius || 190, chunks: new Set(), swept: 0 };
    sim.ship.x = 0; sim.ship.y = 0; follow(sim, true); seat(sim);
  }

  function create(opt) {
    const o = Object.assign({}, DEFAULTS, opt || {}); o.ship = Object.assign({}, DEFAULTS.ship, (opt && opt.ship) || {});
    const sim = {
      o, P: { h: o.h, sigF: o.sigF, sigC: o.sigC, mu: o.mu, E: o.E, nu: o.nu, rhoW: o.rhoW, rhoI: o.rhoI, g: o.g, KIc: o.KIc, Cf: o.Cf, Cl: o.Cl, Cv: o.Cv, Csplit: o.Csplit },
      dt: o.dt, t: 0, n: 0, acc: 0, rand: makeRng(o.seed), ship: makeShip(o.ship), sheets: [], pieces: [], pending: [], events: [],
      FX: 0, FY: 0, MZ: 0, accX: 0, accN: 0, subArea: 0, cover: 0, Rs0: 0, dirtySheets: false, dirtyPieces: false,
      ap: { on: true, heading: 0, hasTarget: false, tx: 0, ty: 0 }, stats: { cusps: 0, splits: 0 },
      Z: { len: 0, k: -1, max: 0, fv: 0, fh: 0, w: 0, x: 0, y: 0, nx: 0, ny: 0, vn: 0, h: 0 }, inn: new Uint8Array(24), act: [],
      proxy: { x: 0, y: 0, vx: 0, vy: 0, w: 0, im: 0, ii: 0 }, big: [], nAwake: 0, awake: new Uint8Array(64), sg: { on: false, lists: [], x0: 0, y0: 0, cell: 48, gw: 0, gh: 0 }, gcount: new Int32Array(1024), gitem: new Int32Array(1024), gcell: new Int32Array(1024), gfill: new Int32Array(1024),
    };
    sim.proxy.im = 1 / (1.2 * sim.ship.m); sim.proxy.ii = 1 / sim.ship.Izz;
    if (o.scenario === 'level') buildLevel(sim); else if (o.scenario === 'field') buildField(sim); else buildFloes(sim, o.scenario === 'pack');
    sim.ship.u = o.u0 === undefined ? 4 : o.u0; sim.Rs0 = lindqvist(sim, sim.P.h, 0).Rs;
    return sim;
  }

  function setThickness(sim, h) {
    const f = sim.P.h / h; sim.P.h = h; sim.Rs0 = lindqvist(sim, h, 0).Rs;
    for (const S of sim.sheets) { S.h = h; if (S.im > 0) { S.im *= f; S.ii *= f; S.mass /= f; } }
    for (const p of sim.pieces) { p.h = h; p.im *= f; p.ii *= f; }
  }

  return { create, step, advance, follow, lindqvist, setThickness, ellOf, vnoise, hash2, DEFAULTS, CH };
})();
if (typeof module !== 'undefined') module.exports = IceSim;
