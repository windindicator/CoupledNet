import argparse
import functools
from pathlib import Path

from couplednet.pdes import FirstOrderPDE, high_frequency_solution
from experiments.common import load_json_config, train_one


def main():
    parser = argparse.ArgumentParser(description="Section 5.1 high-frequency first-order PDE.")
    parser.add_argument("--config", default="experiments/configs/section_5_1_8layer.json")
    parser.add_argument("--arch", choices=["CoupledMlp", "ModifiedMlp", "PirateNet"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/section_5_1")
    args = parser.parse_args()

    raw = load_json_config(args.config)
    depth = int(raw.get("depth", 0))
    archs = [args.arch] if args.arch else raw["architectures"]
    omega = float(raw.get("omega", 30.0))
    solution = functools.partial(high_frequency_solution, omega=omega)

    for arch_name in archs:
        arch_depth = raw.get("architecture_overrides", {}).get(arch_name, {}).get("depth", depth)
        train_one(FirstOrderPDE, solution, raw, arch_name, arch_depth, args.seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
