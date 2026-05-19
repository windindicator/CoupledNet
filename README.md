# CoupledNet Minimal Release

Minimal JAX implementation for the ICML 2026 paper **Deep Coupling Learning for Solving PDEs**.

This folder is a minimal code release for the ICML CoupledNet experiments. It keeps only the two first-order PDE experiments corresponding to Sections 4.1 and 4.2, plus the architecture baselines needed for comparison.

## Contents

- `couplednet_minimal/`: small JAX/Flax PINN implementation with `CoupledMlp`, `ModifiedMlp`, and `PirateNet`.
- `experiments/section_4_1_high_frequency.py`: high-frequency PDE experiment.
- `experiments/section_4_2_high_dynamic_range.py`: high-dynamic-range PDE experiment.
- `experiments/configs/section_4_1_8layer.json`: Section 4.1 8-layer config.
- `experiments/configs/section_4_1_16layer.json`: Section 4.1 16-layer config.
- `experiments/configs/section_4_1_24layer.json`: Section 4.1 24-layer config.
- `experiments/configs/section_4_1_couplednet/`: best CoupledNet config for each reported depth.
- `experiments/configs/section_4_1_appendix_baselines/`: single Appendix baseline config for `ModifiedMlp` and `PirateNet`.
- `experiments/records/section_4_1_warmup_search_results.json`: retained warmup search record, not a formal experiment entry.
- `experiments/configs/section_4_2_default.json`: Section 4.2 Appendix G shallow/deep configs.

## Installation

```bash
pip install -e .
```

For GPU runs, install the JAX/JAXLIB build matching your CUDA version before installing this package.

## Run Section 4.1

Run the best CoupledNet config for a reported depth:

```bash
python -m experiments.section_4_1_high_frequency \
  --config experiments/configs/section_4_1_couplednet/couplednet_8L_best.json
```

Available depth configs are stored in:

```text
experiments/configs/section_4_1_couplednet/
```

Run the Appendix baseline architectures:

```bash
python -m experiments.section_4_1_high_frequency \
  --config experiments/configs/section_4_1_appendix_baselines/appendix_baselines_paper.json
```

Best CoupledNet depth configs are stored in:

```text
experiments/configs/section_4_1_couplednet/
```

Current CoupledNet depth configs use the best setting selected for each depth:

| Depth | Warmup | Best L2 | Final L2 |
|---|---:|---:|---:|
| 4L | 2000 | 1.9985e-2 | 2.3062e-2 |
| 8L | 0 | 1.1745e-2 | 1.1745e-2 |
| 16L | 2000 | 1.4741e-2 | 1.5116e-2 |
| 24L | 0 | 1.1635e-2 | 1.1745e-2 |

## Run Section 4.2

```bash
python -m experiments.section_4_2_high_dynamic_range \
  --config experiments/configs/section_4_2_default.json
```

Section 4.2 follows Appendix G: Adam, exponential decay period `2000` with ratio `0.9`,
batch size `512`, width `256`, `50,000` steps, `tanh`, no Fourier features, no NTK
re-weighting, and no output-layer constraint for CoupledNet.

The config contains these reproducibility entries:

| Run | Layers | LR |
|---|---:|---:|
| CoupledNet shallow | 8L | 1e-3 |
| CoupledNet deep | 16L | 1e-4 |
| PirateNet shallow | 9L / 3 blocks | 1e-3 |
| PirateNet deep Appendix extra | 18L / 6 blocks | 1e-3 |

## Notes

- `PirateNet` uses blocks; the scripts convert a linear-layer budget to blocks with `depth // 3`.
- Results are written to `results/`.
- CoupledNet configs use Adam, initial learning rate `1e-3`, cosine decay for `50,000` steps, batch size `1024`, 128 hidden units, Fourier features, random weight factorization, and no loss reweighting. Warmup is depth-specific because the reported CoupledNet row keeps the best result for each depth.
- Appendix baseline configs use Adam, initial learning rate `1e-3`, exponential decay with period `2000` and ratio `0.9`, batch size `1024`, 256 hidden units, Fourier features, random weight factorization, NTK reweighting, and no warmup. `ModifiedMlp` uses `4Lx256`; `PirateNet` uses `6Bx256`.
- Warmup search results are kept as a record in `experiments/records/` so the depth-specific CoupledNet choices are traceable.

## Citation

```bibtex
@inproceedings{meng2026couplednet,
  title={Deep Coupling Learning for Solving PDEs},
  author={MENG, Lingshi and Shi, Haosen and Pan, Sinno Jialin},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

This code is released under the MIT License.
