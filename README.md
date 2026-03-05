# Flash-SLiM: OS-Aware On-Device LLM Inference Framework

Flash-SLiM is a LiteRT(TensorFlow Lite)-based research framework for running decoder-only LLMs under memory constraints.  
The project focuses on **storage-aware weight streaming** with **I/O–Compute overlap**, and provides end-to-end tooling from metadata generation to runtime inference/profiling.

## Core Features (Tech at a glance)

- **One-line stack summary**: LiteRT/TFLite + XNNPACK weight cache + CMT/prefetch planning + double-buffered weight streaming + eBPF(USDT) profiling + Direct I/O (`io_uring` / parallel `pread`) + Bazel-based reproducible builds.
- **Model runtime**: Decoder-only LLM inference with prefill/decode split, token sampling, and optional LoRA path.
- **Streaming runtime**: Controller/prefetcher architecture for chunk lifecycle orchestration and compute–I/O overlap.
- **Profiling stack**: TFLite profiling summaries + custom scope metrics + eBPF phase/operator tracing.

## What is implemented now

- **Main runtime binary**: `text_generator_main`
  - Prompt prefill + decode loop
  - Sampling: temperature / top-k / top-p / repetition penalty
  - Multi-level profiling output (custom scope + TFLite profiling summary CSV)
  - Weight-streaming runtime path behind build config (`--config=weight_streaming`)
- **Pre-runtime binary**: `cmt_generator`
  - Runs model and records weight-chunk access metadata
  - Writes `weight_chunks_metadata_table.json`
- **Weight streaming runtime components** (active development)
  - `WeightChunkController`
  - `WeightChunkPrefetcher`
  - `WeightChunkIOEngine` (`io_uring`)
  - JSON loader/writer utilities in `weight_chunk_controller_utils.{h,cc}`

## Architecture (current code path)

### 1) Pre-runtime (offline)

1. Run `cmt_generator`
2. Collect weight chunk access per mode (`PREFILL`, `DECODE`)
3. Save JSON metadata:
   - `metadata` (version, per-mode chunk stats, max aligned size, buffer size)
   - `weight_chunks` (per-mode chunk entries)
4. Generate prefetch plans using `tools/model_prefetch_planner/prefetch_planner.py` from CMT JSON and ops profiling logs (`bpf_profile_ops_results_*`), producing plan JSON files (for example, `prefetch_plan_simple_<threads>_<tokens>.json`).

> Note: `JsonPrefetchPlanLoader` can read CMT-only JSON and can also consume an optional `prefetch_plan` section when present.

### 2) Runtime (online)

1. `text_generator_main` loads model/tokenizer and builds interpreter
2. Applies XNNPACK delegate (with weight cache path if provided)
3. Loads the generated chunk/prefetch plan JSON (`--prefetch_plan_path`)
4. Controller + prefetcher orchestrate chunk I/O and buffer switching
5. Prefill/decode executes while profiling metrics are collected

## Core source map

- `flash-slim/text_generator_main.cc`: main LLM inference executable
- `flash-slim/cmt_generator.cc`: CMT generation executable
- `flash-slim/common.{h,cc}`: shared runtime helpers
- `flash-slim/profiler.{h,cc}`: profiling helpers / scope events
- `flash-slim/sampler.{h,cc}`: token sampling functions
- `flash-slim/weight_chunk_controller.{h,cc}`: runtime chunk orchestration
- `flash-slim/weight_chunk_prefetcher.{h,cc}`: async prefetch scheduling and worker logic
- `flash-slim/weight_chunk_io_engine.{h,cc}`: `io_uring` I/O engine abstraction
- `flash-slim/weight_chunk_controller_utils.{h,cc}`: JSON writer/loader for chunk metadata/plans
- `flash-slim/BUILD`: Bazel targets for libraries and binaries

## Build targets

- `//flash-slim:text_generator_main`
- `//flash-slim:cmt_generator`

Top-level wrappers:

- `./build.sh` → builds `text_generator_main`
- `./build_cmt_generator.sh` → builds `cmt_generator`

Both scripts copy outputs to `bin/`.

## Prerequisites

- Bazel/Bazelisk toolchain
- C++ build toolchain compatible with LiteRT/XNNPACK
- Linux environment is the primary target for provided scripts (`taskset`, cgroups, `/proc`, `io_uring` usage)

For repository scripts:

- Create env file:

```bash
cp .env.sample .env
```

- Update paths in `.env` as needed (`ROOT_PATH`, `MODEL_PATH`, `PROMPT_PATH`, `BAZEL_ROOT`, etc.)

- (Optional) Run dependency script:

```bash
./scripts/install_prerequisites.sh
```

> `install_prerequisites.sh` is apt/dpkg-oriented and intended for Debian/Ubuntu-like systems.

## Quick start

### 1) Build binaries

```bash
./build.sh
./build_cmt_generator.sh
```

### 2) Generate CMT JSON

```bash
./run_cmt_generator.sh
```

Default output file:

- `weight_chunks_metadata_table.json`

### 2-1) Generate Prefetch Plan (current workflow)

Prefetch plans are currently generated with `tools/model_prefetch_planner/prefetch_planner.py`.

When `run_cmt_generator.sh` runs, the following steps are executed in sequence:

1. Run `cmt_generator` → generate `weight_chunks_metadata_table.json`
2. Run `tools/ebpf/profile_ops.py` → generate ops profiling logs
3. Run `tools/model_prefetch_planner/prefetch_planner.py`
   - `--cmt weight_chunks_metadata_table.json`
   - `--profile-pattern <ops_log_file>`
   - `--strategy simple`, `--strategy rechunk`

Generated outputs (default):

- `prefetch_plan_simple_<threads>_<tokens>.json`
- `prefetch_plan_rechunk_<threads>_<tokens>.json`

Manual run example:

```bash
python3 ./tools/model_prefetch_planner/prefetch_planner.py \
  --cmt weight_chunks_metadata_table.json \
  --profile-pattern "bpf_profile_ops_results_*" \
  --output prefetch_plan.json \
  --strategy simple
```

To apply a plan at runtime, pass the generated file path to `text_generator_main` via `--prefetch_plan_path`.
(`run.sh` default is `prefetch_plan.json`)

### 3) Run LLM inference

```bash
./run.sh
```

Useful options:

```bash
./run.sh --help
./run.sh --log
./run.sh --threads 4 --core 0-3
./run.sh --memory 512M --memory 1G --log
```

## eBPF Profiling

Flash-SLiM supports USDT + eBPF profiling for phase-level and operator-level runtime analysis.

- Build-time flag: `EBPF_TRACE_ENABLED` (Bazel config: `--config=ebpf`)
- USDT probes are emitted from profiling hooks in `flash-slim/profiler.h`
- eBPF collectors live in `tools/ebpf/`
- **Important**: profiling with eBPF works only when the binary is built with eBPF enabled (`--config=ebpf`).

### 1) Phase-level profiling (inference runtime)

- Script: `tools/ebpf/profile_phase.py`
- Typical runner integration: `run.sh`
- Output example: `bpf_profile_phase_results_<threads>threads_prefill_<tokens>.log`

`run.sh` has `BPF_PHASE_LOGGING=false` by default, so phase-level eBPF collection is disabled unless enabled in the script/config.

### 2) Operator-level profiling (CMT generation path)

- Script: `tools/ebpf/profile_ops.py`
- Integration: `run_cmt_generator.sh` starts/stops eBPF profiler around `bin/cmt_generator`
- Output example: `bpf_profile_ops_results_<threads>threads_prefill_<tokens>.log`

The generated ops-level logs are consumed by `tools/model_prefetch_planner/prefetch_planner.py` in `run_cmt_generator.sh`.

### 3) Manual eBPF run (direct)

You can run the collectors manually by passing:

1. target binary path
2. report output file path

```bash
sudo python3 ./tools/ebpf/profile_phase.py ./bin/text_generator_main bpf_phase_profile.log
sudo python3 ./tools/ebpf/profile_ops.py ./bin/cmt_generator bpf_ops_profile.log
```

### eBPF prerequisites

- Linux environment with eBPF support
- `sudo` privileges for BPF attach
- Python BCC runtime (`bcc` / `python3-bcc` package family)
- Binary built with USDT probes enabled (`--config=ebpf` when building manually)

If you build without `--config=ebpf`, regular inference and CSV profiling still work, but eBPF probe collection is not available.

## Direct binary execution (example)

You can run binaries directly from `bin/` as well:

```bash
bin/text_generator_main \
  --tflite_model <model.tflite> \
  --sentencepiece_model <tokenizer.model> \
  --prompt "Write an email:" \
  --max_decode_steps 32 \
  --num_threads 4 \
  --weight_cache_path <model.xnnpack_cache> \
  --prefetch_plan_path prefetch_plan.json \
  --io_engine io_uring \
  --io_ring_depth 64 \
  --io_subread_bytes 524288 \
  --io_min_block_size 524288 \
  --io_max_threads 4
```

For the full `text_generator_main` flag list, see the `ABSL_FLAG(...)` definitions in `flash-slim/text_generator_main.cc`.

## Outputs

- Inference logs and CSV summaries: `benchmark/llm_infer_results/...`
- CMT metadata JSON: `weight_chunks_metadata_table.json` (or a custom path set by `--output_cmt_path`)
- Additional analysis/profiling utilities: subdirectories under `tools/`

## Notes

- The repository still contains experimental/prototype code under `test/` and analysis utilities under `tools/`.
- Weight streaming runtime is actively evolving; interfaces and JSON schema may continue to change.

## Development Management

### Requirement Specs

- Memory profiling requirement: `docs/requirements/memory-01-decoder-memory-profiling.md`
- Requirement filename format: `docs/requirements/meta-01-requirement-filename-format.md`
- Development tracking summary: `docs/dev_log.md`
