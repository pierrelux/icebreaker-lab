"""Throughput of the JAX icebreaker environment: batched rollouts with random steering at full throttle.

    python bench_icebreaker_jax.py 1 8            # batch sizes; keep them small on a laptop
    python bench_icebreaker_jax.py 256 1024 4096  # on a GPU node
    python bench_icebreaker_jax.py --fast 256     # 180 Hz physics instead of 360 Hz (within ~5% on speed and resistance)

Memory: about 3 MB of state per environment with the default 600 m x 240 m grid, times two to three for copies.
"""
import json
import resource
import sys
import time

import jax
import jax.numpy as jnp

from icebreaker_jax import Config, EnvParams, Icebreaker

args = sys.argv[1:]
fast = "--fast" in args
cfg = Config(dt=1 / 180, inner=3) if fast else Config()
env = Icebreaker(cfg)
T = 50                                              # env steps per rollout (10 s of simulated time)
step_s = cfg.outer * cfg.inner * cfg.dt


def rollout(state, params, key):
    def f(s, k):
        action = jnp.stack([1.0, 0.3 * jax.random.uniform(k, (), jnp.float32, -1, 1)])
        _, s, reward, _, _ = env.step(s, action, params)
        return s, reward
    return jax.lax.scan(f, state, jax.random.split(key, T))


print("device:", jax.devices()[0], "| physics rate:", round(1 / cfg.dt), "Hz", flush=True)
for B in [int(a) for a in args if a != "--fast"] or [1, 8]:
    try:
        params = jax.tree.map(lambda v: jnp.broadcast_to(v, (B,)), EnvParams.default())
        keys = jax.random.split(jax.random.key(B), B)
        _, state = jax.jit(jax.vmap(env.reset))(keys, params)
        run = jax.jit(jax.vmap(rollout))
        state, r = run(state, params, keys)             # compiles; the ships reach the ice during these 10 s
        jax.block_until_ready(r)
        t0 = time.time()
        state, r = run(state, params, keys)             # timed: 10 s of icebreaking
        jax.block_until_ready(r)
        wall = time.time() - t0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
        print(json.dumps({"batch": B, "env_steps_per_s": round(B * T / wall), "simulated_s_per_wall_s": round(B * T * step_s / wall, 1),
                          "wall_s": round(wall, 2), "cusps_broken": int(state.cusps.sum()), "peak_host_ram_GB": round(peak, 2)}), flush=True)
    except Exception as e:                            # typically out of memory at a batch too large for the device
        print(json.dumps({"batch": B, "failed": type(e).__name__ + ": " + str(e)[:160]}), flush=True)
        break
