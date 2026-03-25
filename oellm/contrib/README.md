# Contrib Benchmark Registry

Community-contributed benchmarks integrated into the ELLIOT evaluation platform. Each benchmark runs as a self-contained plugin -- no changes to core scheduling code required.

To add your own benchmark, see the [Contributing Guide](CONTRIBUTING.md).

## Benchmarks

| Benchmark | Task Group | Description | Paper | Code |
|---|---|---|---|---|
| RegionDial-Bench | `regiondial-bench` | Multi-round region grounding and segmentation on RefCOCOg and RefCOCO+. Evaluates robustness to error accumulation across dialogue turns. | [arXiv:2602.03733](https://arxiv.org/abs/2602.03733) | [lmsdss/RegionReasoner](https://github.com/lmsdss/RegionReasoner) |

### RegionDial-Bench

**Metrics:** gIoU (primary), cIoU, bbox_AP, pass_rate@0.3/0.5/0.7/0.9, per-round R1–R7

```bash
oellm schedule-eval \
  --models lmsdss/RegionReasoner-7B \
  --task_groups regiondial-bench \
  --venv_path ~/elliot-venv
```

Requires cluster-specific setup (`REGION_REASONER_DIR`, etc.). See the full [RegionDial-Bench README](regiondial_bench/README.md) for prerequisites and configuration.
