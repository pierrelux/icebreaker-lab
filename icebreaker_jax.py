"""Icebreaker Lab in JAX: a ship breaking a continuous ice sheet, as a batched environment.

This is a port of the hull-against-ice physics in icebreaker-core.js. Every array has a fixed shape,
so `reset` and `step` can be jit-compiled and vmapped over thousands of environments.

Same as the JavaScript core
  * the ice sheet is a signed-distance field on a grid (negative inside the ice);
  * the hull waterline is sampled every ~0.5 m; overlap crushes the ice edge at pressure sigC
    on the inclined hull face;
  * when the vertical force on a contact zone exceeds P_f = C_f (theta/pi)^2 sigF h^2,
    a cusp of radius R = C_l * l * (1 + C_v * v_n) is removed;
  * 3-DOF ship (surge, sway, yaw) with azimuthing thrust.

Different from the JavaScript core
  * no rigid broken pieces. A second grid records where ice has been broken, and the share of the
    hull bottom lying over broken ice scales Lindqvist's submersion resistance. `cover_gain` is the
    one constant fitted so that this matches the rigid-piece version;
  * only the continuous sheet: no separate floes, no floe splitting;
  * at most `kb` cusps and `kc` crushing cuts per physics substep (the rest wait one substep).

Usage
    env = Icebreaker(Config())
    params = EnvParams.default()
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(state, action, params)   # action = [throttle, rudder] in [-1, 1]
    # batched: jax.vmap(env.step, in_axes=(0, 0, None))
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

DEG = math.pi / 180.0


@dataclass(frozen=True)
class Config:
    """Static settings: anything that fixes an array shape or is baked into the compiled step."""
    # domain (m): ice fills x > ~150; the ship starts at x = 70 heading +x
    W: float = 600.0
    H: float = 240.0
    cell: float = 0.5
    # time: one env step = outer * inner physics substeps of dt seconds (default 0.2 s)
    dt: float = 1.0 / 360.0
    inner: int = 6          # substeps between refreshes of thickness-at-hull and bottom coverage
    outer: int = 12
    t_max: float = 180.0
    # ship (a 76 m icebreaker)
    L: float = 76.0
    B: float = 18.0
    T: float = 6.5
    Cb: float = 0.55
    bow_frac: float = 0.36
    stern_frac: float = 0.2
    bow_pow: float = 2.0
    stern_taper: float = 0.45
    phi_mid: float = 8 * DEG      # angle of the hull-surface normal below horizontal (0 = vertical side)
    phi_bow: float = 46 * DEG
    phi_stem: float = 62 * DEG
    phi_stern: float = 30 * DEG
    ds: float = 0.5
    TB: float = 2.0e6             # bollard pull, N
    uow: float = 8.4              # open-water speed, m/s
    # ice constants
    sigC: float = 2.3e6
    mu: float = 0.15
    E: float = 5.4e9
    nu: float = 0.33
    rhoW: float = 1025.0
    rhoI: float = 880.0
    g: float = 9.81
    Cf: float = 3.1
    Cl: float = 0.30
    Cv: float = -0.14
    cover_gain: float = 0.93      # fitted: bottom coverage of the rigid-piece model in steady icebreaking
    # fixed shapes
    h_max: float = 1.3            # thickest ice (thickness x local multiplier) a cusp patch must hold, m
    zcap: int = 12                # contact zones examined per substep
    kb: int = 2                   # cusps broken per substep
    kc: int = 6                   # crushing cuts per substep
    n_ridges: int = 3
    hres: float = 2.0             # resolution of the thickness-multiplier grid, m
    # observation: local ice thickness on a body-fixed lattice (obs_ny x obs_nx points, obs_dx apart)
    obs_nx: int = 24
    obs_ny: int = 13
    obs_dx: float = 8.0

    @cached_property
    def nx(self) -> int:
        return int(round(self.W / self.cell)) + 1

    @cached_property
    def ny(self) -> int:
        return int(round(self.H / self.cell)) + 1

    def ell(self, h):
        """Characteristic length of a floating ice plate of thickness h."""
        return (self.E * h ** 3 / (12 * (1 - self.nu ** 2) * self.rhoW * self.g)) ** 0.25

    @cached_property
    def hull(self) -> dict:
        """Waterline points, outward normals and frame angles; index 0 is the middle of the transom."""
        L, hb = self.L, self.B / 2
        Lb, Ls, xb, xs = self.bow_frac * L, self.stern_frac * L, L / 2, -L / 2

        def half(x):
            x = np.asarray(x, dtype=np.float64)
            bow = hb * (1 - np.clip((x - (xb - Lb)) / Lb, 0, 1) ** self.bow_pow)
            stern = hb * (1 - self.stern_taper * np.clip((xs + Ls - x) / Ls, 0, 1) ** 2.5)
            return np.where(x > xb - Lb, bow, np.where(x < xs + Ls, stern, hb))

        M = 1500
        xa = xb - L * np.arange(M + 1) / M
        yT = float(half(xs))
        dx = np.concatenate([xa, np.full(39, xs), xa[:0:-1]])
        dy = np.concatenate([half(xa), yT * (1 - np.arange(1, 40) / 20), -half(xa[:0:-1])])
        seg = np.hypot(np.diff(np.append(dx, dx[0])), np.diff(np.append(dy, dy[0])))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        n = 2 * int(round(total / (2 * self.ds)))
        t = np.arange(n) * total / n
        px = np.interp(t, cum, np.append(dx, dx[0]))
        py = np.interp(t, cum, np.append(dy, dy[0]))
        tx, ty = np.roll(px, -1) - np.roll(px, 1), np.roll(py, -1) - np.roll(py, 1)
        ln = np.hypot(tx, ty)
        nx_, ny_ = ty / ln, -tx / ln
        flip = nx_ * px + ny_ * py < 0
        nx_, ny_ = np.where(flip, -nx_, nx_), np.where(flip, -ny_, ny_)
        xiB, xiS = (px - (xb - Lb)) / Lb, (xs + Ls - px) / Ls
        ss = lambda a, b, x: (lambda q: q * q * (3 - 2 * q))(np.clip((x - a) / (b - a), 0, 1))
        phi = np.full(n, self.phi_mid)
        phi = np.where(xiB > 0, self.phi_mid + (self.phi_bow - self.phi_mid) * ss(0, 0.3, xiB) + (self.phi_stem - self.phi_bow) * ss(0.8, 1, xiB), phi)
        phi = np.where((xiB <= 0) & (xiS > 0), self.phi_mid + (self.phi_stern - self.phi_mid) * ss(0, 0.8, xiS), phi)
        area = 0.5 * abs(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
        k = n // 2                                   # start the loop at the stern so a zone never wraps at the stem
        r = lambda a: np.roll(a, -k).astype(np.float32)
        # points inside the waterline on a 2 m lattice, used to measure how much broken ice is under the hull
        gx, gy = np.meshgrid(np.arange(xs + 1, xb, 2.0), np.arange(-hb + 1, hb, 2.0))
        inside = np.abs(gy) < half(gx) - 0.3
        return dict(n=n, ds=total / n, px=r(px), py=r(py), nx=r(nx_), ny=r(ny_), sinp=r(np.sin(phi)), cosp=r(np.cos(phi)), tanp=r(np.tan(phi)),
                    area=area, fx=gx[inside].astype(np.float32), fy=gy[inside].astype(np.float32))

    @cached_property
    def rs_per_h(self) -> float:
        """Lindqvist's submersion resistance per metre of ice thickness, at zero speed (N/m)."""
        phi = math.pi / 2 - self.phi_stem
        alpha = math.atan(self.B / (self.bow_frac * self.L))
        psi = math.atan(math.tan(phi) / math.sin(alpha))
        k = self.mu * (0.7 * self.L - self.T / math.tan(phi) - self.B / (4 * math.tan(alpha))
                       + self.T * math.cos(phi) * math.cos(psi) * math.sqrt(1 / math.sin(phi) ** 2 + 1 / math.tan(alpha) ** 2))
        return (self.rhoW - self.rhoI) * self.g * self.B * (self.T * (self.B + self.T) / (self.B + 2 * self.T) + k)

    def lindqvist(self, h, v, sigF):
        """Lindqvist (1989) level-ice resistance in N, as a reference curve."""
        phi = math.pi / 2 - self.phi_stem
        alpha = math.atan(self.B / (self.bow_frac * self.L))
        psi = math.atan(math.tan(phi) / math.sin(alpha))
        rc = 0.5 * sigF * h ** 2 * (math.tan(phi) + self.mu * math.cos(phi) / math.cos(psi)) / (1 - self.mu * math.sin(phi) / math.cos(psi))
        rb = (27 / 64 * sigF * self.B * h ** 1.5 / math.sqrt(self.E / (12 * (1 - self.nu ** 2) * self.g * self.rhoW))
              * (math.tan(psi) + self.mu * math.cos(phi) / (math.cos(psi) * math.sin(alpha))) * (1 + 1 / math.cos(psi)))
        return (rc + rb) * (1 + 1.4 * v / (self.g * h) ** 0.5) + self.rs_per_h * h * (1 + 9.4 * v / math.sqrt(self.g * self.L))


class EnvParams(NamedTuple):
    """Per-environment physical parameters; vmap over these for domain randomisation."""
    h: jax.Array            # nominal ice thickness, m
    sigF: jax.Array         # flexural strength, Pa
    hvar: jax.Array         # amplitude of the smooth thickness variation (fraction of h)
    ridge_gain: jax.Array   # extra thickness on ridges (fraction of h); 0 for none
    edge_wave: jax.Array    # waviness of the ice edge, m

    @staticmethod
    def default() -> "EnvParams":
        f = lambda v: jnp.asarray(v, jnp.float32)
        return EnvParams(h=f(0.6), sigF=f(0.55e6), hvar=f(0.15), ridge_gain=f(0.8), edge_wave=f(7.0))


class State(NamedTuple):
    x: jax.Array
    y: jax.Array
    th: jax.Array       # heading, rad (0 = +x)
    u: jax.Array        # surge speed, m/s
    v: jax.Array        # sway speed, m/s
    r: jax.Array        # yaw rate, rad/s
    rud: jax.Array      # thruster angle as a fraction of 35 degrees
    cover: jax.Array    # share of the hull bottom covered by broken ice
    t: jax.Array
    cusps: jax.Array    # number of cusps broken so far
    sdf: jax.Array      # [ny, nx] signed distance to the ice edge, m (negative inside the ice)
    brash: jax.Array    # [ny, nx] 1 where ice has been broken by bending
    hmul: jax.Array     # [my, mx] local thickness as a multiple of params.h


def _bilin(grid, gx, gy):
    """Bilinear lookup at fractional grid coordinates (clamped to the grid)."""
    ny, nx = grid.shape
    gx = jnp.clip(gx, 0.0, nx - 1.0)
    gy = jnp.clip(gy, 0.0, ny - 1.0)
    i = jnp.minimum(gx.astype(jnp.int32), nx - 2)
    j = jnp.minimum(gy.astype(jnp.int32), ny - 2)
    fx, fy = gx - i, gy - j
    return (grid[j, i] * (1 - fx) + grid[j, i + 1] * fx) * (1 - fy) + (grid[j + 1, i] * (1 - fx) + grid[j + 1, i + 1] * fx) * fy


class Icebreaker:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        hl = cfg.hull
        self.n = hl["n"]
        for k in ("px", "py", "nx", "ny", "sinp", "cosp", "tanp", "fx", "fy"):
            setattr(self, "_" + k, jnp.asarray(hl[k]))
        m = cfg.Cb * cfg.L * cfg.B * cfg.T * 1025.0
        self.mass, self.mx, self.my, self.Izz = m, 1.05 * m, 1.7 * m, 1.6 * m * (0.25 * cfg.L) ** 2
        self.kx, self.kx1 = 0.7 * cfg.TB / cfg.uow ** 2, 0.01 * cfg.TB / cfg.uow
        self.ky, self.ky1 = 0.5 * 1025 * 0.9 * cfg.L * cfg.T, 2e4
        self.kr, self.kr1 = 0.5 * 1025 * 0.9 * cfg.T * cfg.L ** 4 / 32 * 1.6, 2e7
        # patch sizes: a cusp of the largest radius, and a crushing cut of radius 1.2 m
        r_max = cfg.Cl * cfg.ell(cfg.h_max) * 1.15
        self.r_max = r_max
        kh = int(math.ceil((r_max * 1.13 + cfg.cell) / cfg.cell))
        self._off = jnp.arange(-kh, kh + 1, dtype=jnp.int32)
        kc = int(math.ceil((1.2 + cfg.cell) / cfg.cell))
        self._offc = jnp.arange(-kc, kc + 1, dtype=jnp.int32)
        a = np.arange(24) * 2 * math.pi / 24
        self._cos24, self._sin24 = jnp.asarray(np.cos(a), jnp.float32), jnp.asarray(np.sin(a), jnp.float32)
        self._idx = jnp.arange(self.n, dtype=jnp.int32)
        ox = (np.arange(cfg.obs_nx) - 4) * cfg.obs_dx                    # from 32 m astern of midships to well ahead
        oy = (np.arange(cfg.obs_ny) - (cfg.obs_ny - 1) / 2) * cfg.obs_dx
        gx, gy = np.meshgrid(ox, oy)
        self._ox, self._oy = jnp.asarray(gx, jnp.float32), jnp.asarray(gy, jnp.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, key, params: EnvParams):
        c = self.cfg
        f32 = jnp.float32
        k1, k2, k3 = jax.random.split(key, 3)
        ys = jnp.arange(c.ny, dtype=f32) * c.cell
        xs = jnp.arange(c.nx, dtype=f32) * c.cell
        ph = jax.random.uniform(k1, (3,), f32, 0.0, 2 * math.pi)
        xe = 150.0 + params.edge_wave * (0.6 * jnp.sin(2 * math.pi * ys / 97.0 + ph[0]) + 0.3 * jnp.sin(2 * math.pi * ys / 31.0 + ph[1]) + 0.1 * jnp.sin(2 * math.pi * ys / 11.0 + ph[2]))
        X, Y = xs[None, :], ys[:, None]
        sdf = jnp.maximum(jnp.maximum(xe[:, None] - X, X - (c.W - 3.0)), jnp.maximum(3.0 - Y, Y - (c.H - 3.0)))
        # thickness multiplier: smooth noise with a ~47 m length scale, plus a few ridges
        mx, my = int(round(c.W / c.hres)) + 1, int(round(c.H / c.hres)) + 1
        lat = jax.random.uniform(k2, (int(c.H / 47) + 3, int(c.W / 47) + 3), f32, -1.0, 1.0)
        hmul = 1.0 + params.hvar * jax.image.resize(lat, (my, mx), "cubic")
        hx, hy = (jnp.arange(mx, dtype=f32) * c.hres)[None, :], (jnp.arange(my, dtype=f32) * c.hres)[:, None]
        if c.n_ridges > 0:
            rr = jax.random.uniform(k3, (c.n_ridges, 4), f32)
            x1 = 240.0 + rr[:, 0] * (c.W - 290.0)
            x2 = x1 + (rr[:, 1] - 0.5) * 180.0
            gain = params.ridge_gain * (0.6 + 0.6 * rr[:, 2])
            wid = 6.0 + 3.0 * rr[:, 3]
            ex, ey = (x2 - x1)[:, None, None], c.H
            tt = jnp.clip(((hx - x1[:, None, None]) * ex + hy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
            d2 = (hx - x1[:, None, None] - ex * tt) ** 2 + (hy - ey * tt) ** 2
            hmul = hmul + jnp.sum(gain[:, None, None] * jnp.exp(-d2 / (wid[:, None, None] ** 2)), axis=0)
        hmul = jnp.maximum(hmul, 0.35)
        z = jnp.zeros((), f32)
        state = State(x=z + 70.0, y=z + c.H / 2, th=z, u=z + 4.0, v=z, r=z, rud=z, cover=z, t=z, cusps=jnp.zeros((), jnp.int32),
                      sdf=sdf.astype(f32), brash=jnp.zeros((c.ny, c.nx), jnp.uint8), hmul=hmul.astype(f32))
        return self.observe(state, params), state

    # ------------------------------------------------------------------ observation
    def observe(self, s: State, params: EnvParams):
        """Ship state (8,) and local ice thickness in metres on a body-fixed lattice (obs_ny, obs_nx)."""
        c = self.cfg
        cs, sn = jnp.cos(s.th), jnp.sin(s.th)
        wx, wy = s.x + cs * self._ox - sn * self._oy, s.y + sn * self._ox + cs * self._oy
        gx, gy = wx / c.cell, wy / c.cell
        inb = (gx >= 0) & (gy >= 0) & (gx <= c.nx - 1) & (gy <= c.ny - 1)
        ice = inb & (_bilin(s.sdf, gx, gy) < 0)
        thick = jnp.where(ice, params.h * _bilin(s.hmul, wx / c.hres, wy / c.hres), 0.0)
        ship = jnp.stack([s.u / 8.0, s.v / 2.0, s.r * 10.0, sn, cs, s.rud, s.cover, (s.y - c.H / 2) / (c.H / 2)])
        return {"ship": ship, "ice": thick}

    # ------------------------------------------------------------------ physics
    def _substep(self, carry, thr, rud_cmd, hL, cover_target, params: EnvParams):
        c = self.cfg
        x, y, th, u, v, r, rud, cover, sdf, brash, acc_x, cusps = carry
        inv, dt, ds, n = 1.0 / c.cell, c.dt, c.hull["ds"], self.n
        cs, sn = jnp.cos(th), jnp.sin(th)
        wx, wy = x + cs * self._px - sn * self._py, y + sn * self._px + cs * self._py
        gx, gy = wx * inv, wy * inv
        inb = (gx >= 0) & (gy >= 0) & (gx <= c.nx - 1) & (gy <= c.ny - 1)
        d = jnp.where(inb, -_bilin(sdf, gx, gy), -1e3)        # overlap of each hull point with the ice
        contact = d > 0
        pen = jnp.where(contact, d, 0.0)
        nxw, nyw = cs * self._nx - sn * self._ny, sn * self._nx + cs * self._ny
        rx, ry = wx - x, wy - y
        vxs, vys = u * cs - v * sn, u * sn + v * cs
        rvx, rvy = vxs - r * ry, vys + r * rx
        vn, vt = rvx * nxw + rvy * nyw, -rvx * nyw + rvy * nxw
        # contact band on the inclined hull face: the edge is crushed from the top down
        geo = jnp.minimum(hL, pen / self._tanp) / self._cosp * ds
        dN = c.sigC * geo * jnp.clip(vn / 0.08, 0.0, 1.0)      # crushing is plastic: no force while unloading
        vd = jnp.where(vn > 0, vn * self._sinp, 0.0)
        sl = jnp.sqrt(vt * vt + vd * vd) + 1e-4
        sd, st = vd / sl, vt / sl
        fn = dN * (self._cosp + c.mu * self._sinp * sd)
        ft = c.mu * dN * st
        fz = dN * (self._sinp - c.mu * self._cosp * sd)
        fx, fy = -fn * nxw + ft * nyw, -fn * nyw - ft * nxw
        FX, FY, MZ = fx.sum(), fy.sum(), (rx * fy - ry * fx).sum()

        # contact zones: runs of consecutive contacting points, cut every ~2.2 cusp radii
        m = jnp.maximum(jnp.ceil(2.2 * c.Cl * c.ell(params.h) / ds), 1.0).astype(jnp.int32)
        last_gap = lax.cummax(jnp.where(contact, -1, self._idx))
        start = contact & (((self._idx - last_gap - 1) % m) == 0)
        zid = jnp.where(contact, jnp.minimum(jnp.cumsum(start) - 1, c.zcap - 1), c.zcap)
        feat = jnp.stack([fz, geo, geo * wx, geo * wy, geo * vn, geo * hL], axis=1)
        Z = jax.ops.segment_sum(feat, zid, num_segments=c.zcap + 1)[: c.zcap]
        zw = Z[:, 1]
        ok = (zw > 0) & (Z[:, 0] > 0)
        zs = jnp.where(zw > 0, zw, 1.0)
        zx, zy, zvn = Z[:, 2] / zs, Z[:, 3] / zs, Z[:, 4] / zs
        zh = jnp.where(ok, Z[:, 5] / zs, params.h)
        R = jnp.minimum(c.Cl * c.ell(zh) * jnp.clip(1.0 + c.Cv * zvn, 0.4, 1.15), self.r_max)
        # opening angle of the ice wedge around each zone: share of a circle of radius R that is ice
        sx, sy = (zx[:, None] + R[:, None] * self._cos24) * inv, (zy[:, None] + R[:, None] * self._sin24) * inv
        sin_b = (sx >= 0) & (sy >= 0) & (sx <= c.nx - 1) & (sy <= c.ny - 1)
        cnt = jnp.sum(sin_b & (_bilin(sdf, sx, sy) < 0), axis=1)
        thn = jnp.clip(cnt / 24.0 * 2 * math.pi, 0.15 * math.pi, 1.25 * math.pi) / math.pi
        Pf = c.Cf * thn * thn * params.sigF * zh * zh
        ratio = jnp.where(ok, Z[:, 0] / Pf, 0.0)
        top, zi = lax.top_k(ratio, c.kb)
        brk = top > 1.0

        # bending failure: remove a slightly irregular disc and record the area as broken ice
        cx, cy, Rz = zx[zi], zy[zi], R[zi]
        ph1 = 2 * math.pi * jnp.mod(jnp.sin(cx * 12.9898 + cy * 78.233) * 43758.5453, 1.0)
        ph2 = 2 * math.pi * jnp.mod(jnp.sin(cy * 39.3468 + cx * 11.135) * 24634.6345, 1.0)
        ii = jnp.round(cx * inv).astype(jnp.int32)[:, None, None] + self._off[None, None, :]
        jj = jnp.round(cy * inv).astype(jnp.int32)[:, None, None] + self._off[None, :, None]
        ddx, ddy = ii * c.cell - cx[:, None, None], jj * c.cell - cy[:, None, None]
        dist = jnp.sqrt(ddx * ddx + ddy * ddy)
        dsafe = jnp.maximum(dist, 1e-6)
        ca, sa = ddx / dsafe, ddy / dsafe
        c3, s3 = 4 * ca ** 3 - 3 * ca, 3 * sa - 4 * sa ** 3                          # cos and sin of 3x the polar angle
        c5, s5 = 16 * ca ** 5 - 20 * ca ** 3 + 5 * ca, 16 * sa ** 5 - 20 * sa ** 3 + 5 * sa
        p1, p2 = ph1[:, None, None], ph2[:, None, None]
        Re = Rz[:, None, None] * (1 + 0.13 * (0.6 * (s3 * jnp.cos(p1) + c3 * jnp.sin(p1)) + 0.4 * (s5 * jnp.cos(p2) + c5 * jnp.sin(p2))))
        good = brk[:, None, None] & (ii >= 0) & (ii < c.nx) & (jj >= 0) & (jj < c.ny)
        val = jnp.where(good, Re - dist, -jnp.inf)
        iic, jjc = jnp.clip(ii, 0, c.nx - 1), jnp.clip(jj, 0, c.ny - 1)
        iic, jjc = jnp.broadcast_arrays(iic, jjc)
        old = sdf[jjc, iic]
        removed = ((old < 0) & (val >= 0)).astype(jnp.uint8)
        # Tie the value written to the value read (never true inside the grid). Without this the compiler is free to
        # schedule the read after the write, and then it copies the whole grid every substep instead of updating in place.
        val = jnp.where(old > 1e30, old, val)

        # through-thickness crushing: ice deeper inside the hull than dmax no longer exists
        dmax = hL * self._tanp + 0.03
        ex, ei = lax.top_k(jnp.where(contact, d - dmax, -1.0), c.kc)
        rc = 1.2
        qx, qy = wx[ei] - nxw[ei] * (dmax[ei] + rc), wy[ei] - nyw[ei] * (dmax[ei] + rc)
        ci = jnp.round(qx * inv).astype(jnp.int32)[:, None, None] + self._offc[None, None, :]
        cj = jnp.round(qy * inv).astype(jnp.int32)[:, None, None] + self._offc[None, :, None]
        cdx, cdy = ci * c.cell - qx[:, None, None], cj * c.cell - qy[:, None, None]
        cgood = (ex > 0)[:, None, None] & (ci >= 0) & (ci < c.nx) & (cj >= 0) & (cj < c.ny)
        cval = jnp.where(cgood, rc - jnp.sqrt(cdx * cdx + cdy * cdy), -jnp.inf)
        cic, cjc = jnp.broadcast_arrays(jnp.clip(ci, 0, c.nx - 1), jnp.clip(cj, 0, c.ny - 1))

        sdf = sdf.at[jjc, iic].max(val).at[cjc, cic].max(cval)
        brash = brash.at[jjc, iic].max(removed)
        cusps = cusps + jnp.sum(brk).astype(jnp.int32)

        # ship: surge, sway, yaw with azimuthing thrusters at the stern
        rud = rud + jnp.clip(rud_cmd - rud, -1.5 * dt, 1.5 * dt)
        Xi, Yi = FX * cs + FY * sn, -FX * sn + FY * cs
        thrust = jnp.where(thr >= 0, thr * c.TB * (1 - 0.3 * jnp.clip(u / c.uow, 0.0, 1.5)), thr * 0.6 * c.TB)
        dl, au = rud * 35 * DEG, jnp.abs(u)
        Xt, Yt, Nt = thrust * jnp.cos(dl), -thrust * jnp.sin(dl), 0.45 * c.L * thrust * jnp.sin(dl)
        Xh, Yh, Nh = -(self.kx * au + self.kx1) * u, -(self.ky * jnp.abs(v) + self.ky1) * v, -(self.kr * jnp.abs(r) + self.kr1) * r
        cover = cover + (cover_target - cover) * min(1.0, dt / 0.5)
        Xs = -cover * c.rs_per_h * params.h * (1 + 9.4 * au / math.sqrt(c.g * c.L)) * jnp.tanh(u / 0.25)   # broken ice sliding under the hull
        du = (Xi + Xt + Xh + Xs + self.my * v * r) / self.mx
        dv = (Yi + Yt + Yh - self.mx * u * r) / self.my
        dr = (MZ + Nt + Nh) / self.Izz
        u, v, r = u + du * dt, v + dv * dt, r + dr * dt
        x, y, th = x + (u * cs - v * sn) * dt, y + (u * sn + v * cs) * dt, th + r * dt
        return (x, y, th, u, v, r, rud, cover, sdf, brash, acc_x + Xi + Xs, cusps)

    def step(self, state: State, action, params: EnvParams):
        """Advance outer * inner physics substeps. action = [throttle, rudder command], each in [-1, 1]."""
        c = self.cfg
        thr, rud_cmd = jnp.clip(action[0], -1.0, 1.0), jnp.clip(action[1], -1.0, 1.0)

        def outer(carry, _):
            x, y, th, brash = carry[0], carry[1], carry[2], carry[9]
            cs, sn = jnp.cos(th), jnp.sin(th)
            wx, wy = x + cs * self._px - sn * self._py, y + sn * self._px + cs * self._py
            hL = params.h * _bilin(state.hmul, wx / c.hres, wy / c.hres)             # ice thickness at each hull point
            bx = jnp.clip(jnp.round((x + cs * self._fx - sn * self._fy) / c.cell).astype(jnp.int32), 0, c.nx - 1)
            by = jnp.clip(jnp.round((y + sn * self._fx + cs * self._fy) / c.cell).astype(jnp.int32), 0, c.ny - 1)
            target = jnp.clip(c.cover_gain * jnp.mean(brash[by, bx].astype(jnp.float32)), 0.0, 1.0)
            carry = lax.fori_loop(0, c.inner, lambda i, cr: self._substep(cr, thr, rud_cmd, hL, target, params), carry)
            return carry, None

        z = jnp.zeros((), jnp.float32)
        carry = (state.x, state.y, state.th, state.u, state.v, state.r, state.rud, state.cover, state.sdf, state.brash, z, state.cusps)
        carry, _ = lax.scan(outer, carry, None, length=c.outer)
        x, y, th, u, v, r, rud, cover, sdf, brash, acc_x, cusps = carry
        t = state.t + c.outer * c.inner * c.dt
        new = State(x=x, y=y, th=th, u=u, v=v, r=r, rud=rud, cover=cover, t=t, cusps=cusps, sdf=sdf, brash=brash, hmul=state.hmul)
        reward = x - state.x                                                           # metres made good along +x
        done = (x > c.W - c.L) | (t >= c.t_max) | (jnp.abs(y - c.H / 2) > c.H / 2 - 25.0) | (x < 20.0)
        info = {"ice_resistance": -acc_x / (c.outer * c.inner), "cusps": cusps}        # mean longitudinal ice force over the step, N
        return self.observe(new, params), new, reward, done, info


if __name__ == "__main__":
    import time

    env = Icebreaker(Config())
    params = EnvParams.default()
    step = jax.jit(env.step)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    act = jnp.array([1.0, 0.0])
    obs, state, rew, done, info = step(state, act, params)            # compile
    t0 = time.time()
    for i in range(150):
        obs, state, rew, done, info = step(state, act, params)
    wall = time.time() - t0
    print(f"30 s simulated in {wall:.2f} s; x = {float(state.x):.1f} m, speed = {float(state.u):.2f} m/s, "
          f"cusps = {int(state.cusps)}, ice resistance = {float(info['ice_resistance']) / 1e3:.0f} kN")
