# CoupledNet

Minimal JAX implementation for the ICML 2026 paper **Deep Coupling Learning for Solving PDEs**.

This repository contains the code needed to run the experiments from Sections 4.1 and 4.2.

## Installation

Create or activate a Python environment with JAX installed, then install this package:

```bash
pip install -e .
```

For GPU runs, install the JAX/JAXLIB build matching your CUDA version before installing this package.

## Repository Layout

```text
couplednet_minimal/        Core model, PDE, sampler, and training utilities
experiments/               Experiment entry points
experiments/configs/       Reproducible experiment configs
experiments/records/       Search records retained for traceability
```

## Run Section 4.1

Run a CoupledNet depth config:

```bash
python -m experiments.section_4_1_high_frequency \
  --config experiments/configs/section_4_1_couplednet/couplednet_8L_best.json
```

Other CoupledNet depth configs are in:

```text
experiments/configs/section_4_1_couplednet/
```

Run the Appendix baseline architectures:

```bash
python -m experiments.section_4_1_high_frequency \
  --config experiments/configs/section_4_1_appendix_baselines/appendix_baselines_paper.json
```

## Run Section 4.2

Run all Section 4.2 configs:

```bash
python -m experiments.section_4_2_high_dynamic_range \
  --config experiments/configs/section_4_2_default.json
```

Run one Section 4.2 config by name:

```bash
python -m experiments.section_4_2_high_dynamic_range \
  --config experiments/configs/section_4_2_default.json \
  --run couplednet_8L_lr1e-3
```

Available run names are listed in:

```text
experiments/configs/section_4_2_default.json
```

## Outputs

Experiment outputs are written to `results/` by default. Use `--output-dir` to choose another directory:

```bash
python -m experiments.section_4_1_high_frequency \
  --config experiments/configs/section_4_1_couplednet/couplednet_8L_best.json \
  --output-dir results/section_4_1_8L
```

## License

This code is released under the MIT License.
