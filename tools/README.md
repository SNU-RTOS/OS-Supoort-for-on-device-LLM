# tools/ Directory Guide

`tools/` contains analysis, profiling, and utility scripts used by the Flash-SLiM workflow.

## Directory map

- `ebpf/`: eBPF collectors and probe programs
- `model_prefetch_planner/`: prefetch plan generation from CMT + profiling logs
- `model_dump/`: model structure dump/analysis utilities
- `benchmark/`: CSV/log post-processing helpers
- `plot/`: visualization scripts
- `cache/`: cache-clear helpers
- `flamegraph/`: flamegraph utilities
- `power_analysis/`: power-related analysis scripts
- `prompt/`: JSON prompt parsing/update helpers

## Core features

- **USDT + eBPF runtime tracing**
  - Phase-level and operator-level event capture from Flash-SLiM binaries
  - I/O interval and latency-oriented runtime inspection for bottleneck analysis

- **CMT-based prefetch planning**
  - Consumes chunk metadata (`weight_chunks_metadata_table.json`)
  - Merges profiling evidence and generates runtime prefetch plans
  - Supports multiple strategies (`simple`, `rechunk`)

- **Model structure introspection**
  - Dumps model/operator structure for LiteRT/TFLite artifacts
  - Supports downstream report generation via visualization scripts

- **Benchmark result normalization**
  - Post-processes raw profiling CSV outputs into cleaner analysis inputs

- **Experiment utility toolbox**
  - Prompt preprocessing helpers, cache utilities, plotting/flamegraph helpers
  - Supports repeatable benchmarking and analysis workflows

## eBPF tools

- `ebpf/profile_phase.py`
  - Collects phase-level events from USDT probes
  - Input args: `<target_binary_path> <output_report_path>`

- `ebpf/profile_ops.py`
  - Collects operator-level events from USDT probes
  - Input args: `<target_binary_path> <output_report_path>`

Example:

```bash
sudo python3 ./tools/ebpf/profile_phase.py ./bin/text_generator_main bpf_phase_profile.log
sudo python3 ./tools/ebpf/profile_ops.py ./bin/cmt_generator bpf_ops_profile.log
```

Important:

- eBPF collection requires binaries built with `--config=ebpf`
- root privileges are required for BPF attach

## Prefetch planning

- Script: `model_prefetch_planner/prefetch_planner.py`
- Reads CMT (`--cmt`) and profiling logs (`--profile-pattern`)
- Writes prefetch plan JSON (`--output`)
- Strategies: `simple`, `rechunk`

Example:

```bash
python3 ./tools/model_prefetch_planner/prefetch_planner.py \
  --cmt weight_chunks_metadata_table.json \
  --profile-pattern "bpf_profile_ops_results_*" \
  --output prefetch_plan.json \
  --strategy simple
```

## Benchmark post-processing

- `benchmark/fix_profile_report.py`: normalizes/fixes generated CSV profiling outputs

Example:

```bash
python3 ./tools/benchmark/fix_profile_report.py input.csv output.csv
```

## Related docs

- Root overview: `../README.md`
- Core source guide: `../flash-slim/README.md`
- Model dump details: `model_dump/README.md`
