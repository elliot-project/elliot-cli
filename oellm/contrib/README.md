# Contrib Benchmark Registry

Community-contributed benchmarks integrated into the ELLIOT evaluation platform. Each benchmark runs as a self-contained plugin -- no changes to core scheduling code required.

To add your own benchmark, see the [Contributing Guide](CONTRIBUTING.md).

## Benchmarks

| Benchmark | Task Group | Description | Paper | Code |
|---|---|---|---|---|
| RegionReasoner | `region-reasoner` | Multi-turn region grounding and segmentation on RefCOCOg. Evaluates a model's ability to locate and segment objects described in multi-turn conversations. | [arXiv:2602.03733](https://arxiv.org/abs/2602.03733) | [lmsdss/RegionReasoner](https://github.com/lmsdss/RegionReasoner) |

### RegionReasoner

**Metrics:** gIoU (primary), cIoU, bbox_AP, pass_rate@0.3/0.5/0.7/0.9

```bash
oellm schedule-eval \
  --models lmsdss/RegionReasoner-7B \
  --task_groups region-reasoner \
  --venv_path ~/elliot-venv
```

Requires cluster-specific setup (`REGION_REASONER_DIR`, etc.). See the full [RegionReasoner README](region_reasoner/README.md) for prerequisites and configuration.
