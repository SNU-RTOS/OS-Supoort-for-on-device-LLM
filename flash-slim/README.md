# flash-slim/ Source Guide

`flash-slim/` contains the core C++ implementation for Flash-SLiM runtime inference, CMT generation, and weight-streaming components.

## Binaries

- `text_generator_main`
  - Main inference executable (prefill + decode)
  - Sampling controls: temperature, top-k, top-p, repetition penalty
  - Optional weight-streaming runtime path (`USE_WEIGHT_STREAMING`)
  - Optional eBPF USDT probe emission (`EBPF_TRACE_ENABLED`)

- `cmt_generator`
  - Pre-runtime CMT generation executable
  - Records per-mode (`PREFILL`, `DECODE`) weight-chunk metadata
  - Writes `weight_chunks_metadata_table.json` (or `--output_cmt_path`)

## Key modules

- `common.{h,cc}`: shared runtime helpers (interpreter/decode/prefill/KV cache setup)
- `utils.{h,cc}`: utility functions for model/tensor/runtime helpers
- `sampler.{h,cc}`: token sampling logic
- `profiler.{h,cc}`: timing/profiling + USDT probe macros
- `lora_adapter.{h,cc}`: LoRA adapter integration helpers
- `aligned_allocator.h`: aligned allocation helpers

### Weight-streaming runtime modules

- `weight_chunk_controller.{h,cc}`
  - Runtime orchestration for chunk access and buffer management
  - Loads chunk metadata/prefetch plan through loader utilities

- `weight_chunk_prefetcher.{h,cc}`
  - Worker-based prefetch orchestration
  - Supports `io_uring` and parallel pread mode configuration

- `weight_chunk_io_engine.{h,cc}`
  - `io_uring` submission/completion abstraction

- `weight_chunk_controller_utils.{h,cc}`
  - `JsonWeightChunkMetaDataWriter`
  - `JsonPrefetchPlanLoader`

## Build targets

`flash-slim/BUILD` provides:

- Libraries: `utils`, `sampler`, `profiler`, `common`, `weight_chunk_*`, etc.
- Binaries: `text_generator_main`, `cmt_generator`

Top-level wrappers from repository root:

```bash
./build.sh
./build_cmt_generator.sh
```

## Runtime feature toggles

From Bazel configs:

- `--config=weight_streaming` → enables `USE_WEIGHT_STREAMING`
- `--config=ebpf` → enables `EBPF_TRACE_ENABLED`

Note: eBPF probe collection requires binaries built with `--config=ebpf`.

## Related docs

- Root guide: `../README.md`
- Tools overview: `../tools/README.md`
