# 📓 Flash-SLiM Development Log

- Author: Geonha Park
- Project: Flash-SLiM (On-device LLM weight streaming)
- Scope: LiteRT runtime, profiling stack, CMT/prefetch planning, streaming runtime
- Last updated: 2026-03-05

---

## 📆 Phase 1 — LiteRT Baseline Runtime

**Period**: 2025-06 to 2025-07  
**Goal**: Stabilize a LiteRT-based decoder-only LLM baseline

- ✅ Migrated baseline runtime to LiteRT/TFLite flow
- ✅ Consolidated prompt/tokenizer/inference execution path
- ✅ Added initial profiling and metrics plumbing

---

## 📆 Phase 2 — USDT + eBPF Profiling Foundation

**Period**: 2025-07  
**Goal**: Introduce system-observable profiling on runtime execution

- ✅ Added USDT probe integration in runtime profiling path
- ✅ Added eBPF profiling scripts under `tools/ebpf/`
  - `profile_phase.py` (phase-level)
  - `profile_ops.py` (operator-level)
- ✅ Enabled build toggle for probe emission via Bazel config (`--config=ebpf`)

---

## 📆 Phase 3 — CMT Generation and Planning Pipeline

**Period**: 2025-08 to 2025-10  
**Goal**: Build pre-runtime metadata and prefetch planning path

- ✅ Implemented `cmt_generator` to record weight chunk metadata
- ✅ Implemented JSON writer/loader utilities in `weight_chunk_controller_utils.{h,cc}`
- ✅ Integrated planner workflow with `tools/model_prefetch_planner/prefetch_planner.py`
- ✅ Connected ops profiling logs (`bpf_profile_ops_results_*`) as planner input

Artifacts:

- `weight_chunks_metadata_table.json`
- `prefetch_plan_simple_<threads>_<tokens>.json`
- `prefetch_plan_rechunk_<threads>_<tokens>.json`

---

## 📆 Phase 4 — Streaming Runtime Components

**Period**: 2025-10 to 2026-03  
**Goal**: Build runtime orchestration for chunk prefetch + compute overlap

- ✅ Added `WeightChunkController` runtime orchestration layer
- ✅ Added `WeightChunkPrefetcher` worker and plan lookup path
- ✅ Added `WeightChunkIOEngine` (`io_uring`) and parallel pread options
- ✅ Integrated runtime loading via `--prefetch_plan_path`
- 🚧 Ongoing optimization and behavior tuning for chunk scheduling/buffer policy

---

## ✅ Current Workflow Snapshot

### Pre-runtime

1. Run `cmt_generator`
2. Collect CMT metadata (`PREFILL`, `DECODE`)
3. Collect ops-level profiling logs (eBPF)
4. Generate prefetch plan with `tools/model_prefetch_planner/prefetch_planner.py`

### Runtime

1. Run `text_generator_main`
2. Load generated plan via `--prefetch_plan_path`
3. Execute prefill/decode with controller + prefetcher orchestration
4. Collect profiling outputs (TFLite/custom scope and optional eBPF)

---

## 📊 Current Status Matrix

| Area | Status | Notes |
| --- | --- | --- |
| LiteRT inference baseline | ✅ Stable | Main execution path in `text_generator_main` |
| CMT generation | ✅ Implemented | `cmt_generator` + JSON metadata writer |
| Prefetch planning | ✅ Implemented | `model_prefetch_planner` with `simple`/`rechunk` |
| eBPF profiling | ✅ Implemented | Requires `--config=ebpf` build |
| Streaming runtime core | ✅ Implemented | Controller/prefetcher/io-engine integrated |
| Runtime tuning & validation | 🚧 Ongoing | Scheduling and performance tuning in progress |

---

## 📌 Next Steps

- [ ] Improve scheduling heuristics for better compute–I/O overlap consistency
- [ ] Expand controlled benchmark matrix (model/thread/memory constraints)
- [ ] Strengthen runtime validation for plan mismatch and fallback paths
- [ ] Add more post-processing/visualization around chunk-level timeline behavior
- [ ] Implement memory profiling requirements tracked in [docs/requirements/memory-01-decoder-memory-profiling.md](requirements/memory-01-decoder-memory-profiling.md)

### Requirement Specs

- Memory profiling requirement: [docs/requirements/memory-01-decoder-memory-profiling.md](requirements/memory-01-decoder-memory-profiling.md)
- Requirement filename format: [docs/requirements/meta-01-requirement-filename-format.md](requirements/meta-01-requirement-filename-format.md)

---
