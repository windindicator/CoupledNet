import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_collections
from jax.tree_util import tree_leaves, tree_map

from couplednet.samplers import UniformSampler

jax.config.update("jax_default_matmul_precision", "highest")


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_config(raw, arch_name, depth, seed):
    overrides = raw.get("architecture_overrides", {}).get(arch_name, {})
    depth = int(overrides.get("depth", depth))
    if "num_layers" in overrides:
        num_layers = int(overrides["num_layers"])
    elif arch_name == "PirateNet":
        num_layers = max(depth // 3, 1)
    else:
        num_layers = depth

    arch = raw["arch"].copy()
    arch.update({
        "arch_name": arch_name,
        "num_layers": num_layers,
        "hidden_dim": int(overrides.get("hidden_dim", raw["hidden_dim"])),
        "out_dim": 1,
    })
    if arch_name == "PirateNet":
        arch.setdefault("nonlinearity", 0.0)
    else:
        arch.setdefault("nonlinearity", None)
    arch.setdefault("pi_init", None)

    config = ml_collections.ConfigDict()
    config.mode = "train"
    config.seed = int(seed)
    config.input_dim = 2
    config.wandb = ml_collections.ConfigDict({"use": False, "project": "couplednet", "name": "local"})
    config.arch = ml_collections.ConfigDict(arch)
    config.optim = ml_collections.ConfigDict(raw["optim"])
    config.training = ml_collections.ConfigDict({
        "max_steps": int(raw["max_steps"]),
        "batch_size_per_device": int(raw["batch_size_per_device"]),
    })
    config.weighting = ml_collections.ConfigDict(raw["weighting"])
    config.logging = ml_collections.ConfigDict({
        "log_every_steps": 100,
        "log_errors": True,
        "log_losses": True,
        "log_weights": True,
        "log_grads": False,
        "log_ntk": False,
        "log_preds": False,
    })
    config.saving = ml_collections.ConfigDict({"save_every_steps": None, "num_keep_ckpts": 1})
    return config


def train_one(model_cls, solution_fn, raw_config, arch_name, depth, seed, output_dir):
    config = make_config(raw_config, arch_name, depth, seed)
    x_star = jnp.linspace(-1.0, 1.0, int(raw_config["eval_grid_size"]))
    y_star = jnp.linspace(-1.0, 1.0, int(raw_config["eval_grid_size"]))
    model = model_cls(config, solution_fn, x_star, y_star)

    dom = jnp.array([[-1.0, 1.0], [-1.0, 1.0]])
    sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    params_single = tree_map(lambda x: x[0], model.state.params)
    n_params = sum(x.size for x in tree_leaves(params_single))
    if arch_name == "PirateNet":
        display_name = f"PirateNet {config.arch.num_layers}B ({depth} linear-layer budget)"
    else:
        display_name = f"{arch_name} {config.arch.num_layers}L"

    schedule = getattr(config.optim, "schedule", "exponential")
    warmup_steps = int(getattr(config.optim, "warmup_steps", 0))
    print(f"\n{display_name} | width={config.arch.hidden_dim} | params={n_params:,} | seed={seed}", flush=True)
    print(
        f"steps={config.training.max_steps} batch/device={config.training.batch_size_per_device} "
        f"weighting={config.weighting.scheme} schedule={schedule} warmup={warmup_steps}",
        flush=True,
    )

    batch = next(sampler)
    _ = model.step(model.state, batch)
    if config.weighting.scheme in ("ntk", "grad_norm"):
        _ = model.update_weights(model.state, batch)

    best_l2 = float("inf")
    final_l2 = None
    history = []
    start = time.perf_counter()
    eval_every = int(raw_config["eval_every_steps"])
    max_steps = int(config.training.max_steps)

    for step in range(1, max_steps + 1):
        batch = next(sampler)
        model.state = model.step(model.state, batch)
        if config.weighting.scheme in ("ntk", "grad_norm") and step % int(config.weighting.update_every_steps) == 0:
            model.state = model.update_weights(model.state, batch)

        if step % eval_every == 0 or step == max_steps:
            params_single = jax.device_get(tree_map(lambda x: x[0], model.state.params))
            l2 = float(model.compute_l2_error(params_single, x_star, y_star))
            final_l2 = l2
            best_l2 = min(best_l2, l2)
            elapsed = time.perf_counter() - start
            print(f"step={step:6d} l2={l2:.6e} best={best_l2:.6e} time={elapsed:.1f}s", flush=True)
            history.append({"step": step, "l2": l2, "best_l2": best_l2, "time_sec": elapsed})
    total_time = time.perf_counter() - start
    ms_per_step = total_time / max_steps * 1000.0

    result = {
        "run_name": raw_config.get("run_name"),
        "arch_name": arch_name,
        "display_name": display_name,
        "depth": depth,
        "num_layers": int(config.arch.num_layers),
        "hidden_dim": int(config.arch.hidden_dim),
        "params": int(n_params),
        "seed": int(seed),
        "learning_rate": float(config.optim.learning_rate),
        "schedule": str(schedule),
        "warmup_steps": warmup_steps,
        "best_l2": best_l2,
        "final_l2": final_l2,
        "ms_per_step": ms_per_step,
        "total_time_sec": total_time,
        "history": history,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = raw_config.get("run_name") or f"{arch_name}_depth{depth}_seed{seed}"
    path = output_dir / f"{output_stem}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
