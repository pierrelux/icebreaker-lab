"""JAX port on uniform ice with a nearly straight edge: speed and ice resistance, 8 runs per thickness.

Compare with checks/compare_js.js. Uses 32 small environments (about 0.5 GB of RAM on a CPU).

    python checks/compare_jax.py            # physics at 360 Hz
    python checks/compare_jax.py --fast     # physics at 180 Hz
"""
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from icebreaker_jax import Config, EnvParams, Icebreaker  # noqa: E402

fast = "--fast" in sys.argv
cfg = Config(W=700.0, H=200.0, outer=1, n_ridges=0, **(dict(dt=1 / 180, inner=3) if fast else {}))
env = Icebreaker(cfg)
R = 8
hs = jnp.repeat(jnp.array([0.3, 0.6, 0.9, 1.2], jnp.float32), R)
B = hs.shape[0]
params = EnvParams(h=hs, sigF=jnp.full(B, 0.55e6), hvar=jnp.zeros(B), ridge_gain=jnp.zeros(B), edge_wave=jnp.full(B, 0.6))
_, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(1), B), params)
steps = int(round(70 / (cfg.outer * cfg.inner * cfg.dt)))


def rollout(s, p):
    def f(s, _):
        rudder = jnp.clip(2.2 * (-s.th) - 9 * s.r, -1, 1)          # same heading hold as the JavaScript autopilot
        _, s, _, _, info = env.step(s, jnp.stack([1.0, rudder]), p)
        return s, (s.u, info["ice_resistance"], s.t)
    return jax.lax.scan(f, s, None, length=steps)


_, (u, ice, t) = jax.jit(jax.vmap(rollout))(state, params)
u, ice, late = np.asarray(u), np.asarray(ice), np.asarray(t[0] > 30)
for k in range(4):
    sl = slice(R * k, R * k + R)
    um, fm = u[sl][:, late].mean(1), ice[sl][:, late].mean(1) / 1e3
    print(json.dumps({"core": "jax", "h": round(float(hs[R * k]), 2), "speed_mean_sd": [round(float(um.mean()), 2), round(float(um.std(ddof=1)), 2)],
                      "ice_kN_mean_sd": [round(float(fm.mean())), round(float(fm.std(ddof=1)))]}))
