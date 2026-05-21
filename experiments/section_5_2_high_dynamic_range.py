import argparse
import copy
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from couplednet.pdes import FirstOrderPDE, high_dynamic_range_solution
from experiments.common import load_json_config, train_one


def main():
    parser = argparse.ArgumentParser(description="Section 5.2 high-dynamic-range first-order PDE.")
    parser.add_argument("--config", default="experiments/configs/section_5_2_default.json")
    parser.add_argument("--arch", choices=["CoupledMlp", "PirateNet"], default=None)
    parser.add_argument("--run", default=None, help="Optional run name from the config's runs list.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/section_5_2")
    args = parser.parse_args()

    raw = load_json_config(args.config)
    runs = raw.get("runs")
    if runs:
        for run in runs:
            arch_name = run["arch_name"]
            if args.arch and arch_name != args.arch:
                continue
            if args.run and run["name"] != args.run:
                continue

            run_raw = copy.deepcopy(raw)
            run_raw["run_name"] = run["name"]
            run_raw["optim"]["learning_rate"] = float(run["learning_rate"])
            run_raw["architecture_overrides"] = {
                arch_name: {
                    "depth": int(run["depth"]),
                    "num_layers": int(run["num_layers"]),
                    "hidden_dim": int(run.get("hidden_dim", raw["hidden_dim"])),
                }
            }
            train_one(
                FirstOrderPDE,
                high_dynamic_range_solution,
                run_raw,
                arch_name,
                int(run["depth"]),
                args.seed,
                Path(args.output_dir),
            )
        return

    depths = raw["depths"]
    archs = [args.arch] if args.arch else raw["architectures"]

    for depth in depths:
        for arch_name in archs:
            train_one(FirstOrderPDE, high_dynamic_range_solution, raw, arch_name, depth, args.seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
