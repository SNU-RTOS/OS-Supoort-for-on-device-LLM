# memory-01: Decoder Memory Profiling

- ID: `memory-01`
- Title: Decoder-only LLM memory profiling (immutable + mutable)
- Status: `Planned`
- Owner: Flash-SLiM runtime/profiling

## Objective

Add accurate memory accounting for decoder-only LLM execution, separating:

1. **Immutable memory** (model weights)
2. **Mutable memory** (runtime-changing memory)

## Background

The optional `Model Dump` block in `flash-slim/cmt_generator.cc` provides structural inspection, but it is not sufficient for precise per-layer weight ownership accounting and mutable-vs-immutable phase analysis.

## Scope

### In-scope

- Layer-grouped immutable weight accounting for decoder-only models
- Mutable memory accounting across prefill/decode
- Structured report outputs (JSON/CSV)
- Integration with existing profiling flow (`cmt_generator`, planner, runtime)

### Out-of-scope

- Training-time memory profiling
- GPU memory profiling
- Kernel-level allocator redesign

## Functional Requirements

### FR-1: Immutable weight grouping

System must report immutable weight memory for the following 3 groups:

1. **QKV projection weights**
2. **Attention-block FFN weights**
3. **Output projection / vocab-mapping weights** used in decode logits path

For each group, report at least:

- total bytes
- percentage of total immutable weight bytes
- cumulative bytes (ordered report)

### FR-2: Immutable source filtering

Immutable accounting must use stable tensor identifiers and only include immutable allocation classes (for example, `kTfLiteMmapRo`, `kTfLitePersistentRo`).

### FR-3: Mutable memory accounting

System must track mutable memory categories including:

- KV cache growth
- Arena allocations
- Temporary tensors
- Runtime streaming buffers (weight chunk buffers and related runtime allocations)

### FR-4: Phase-aware reporting

Mutable memory must be reported separately for:

- prefill phase
- decode phase

and include a derived split:

- immutable bytes vs mutable bytes

### FR-5: Machine-readable outputs

System must export memory profiling results in:

- JSON (primary)
- CSV (summary)

## Non-Functional Requirements

- Reproducibility: same model + config should produce stable accounting within expected runtime noise
- Low overhead: profiling instrumentation should not materially distort runtime behavior
- Extensibility: grouping rules should be maintainable for additional decoder architectures

## Proposed Integration Points

- `flash-slim/cmt_generator.cc`
  - upgrade optional model dump path into structured memory accounting path
- `flash-slim/weight_chunk_controller_utils.{h,cc}`
  - reuse metadata/report serialization patterns
- `tools/model_dump/*`
  - optional helper scripts for offline aggregation/visualization

## Acceptance Criteria

1. For a target decoder-only model, report includes all 3 immutable groups with bytes and percentages.
2. Mutable memory report includes prefill/decode snapshots and category-level bytes.
3. A combined report clearly shows immutable vs mutable split.
4. Output artifacts are generated automatically in profiling workflow and are parseable by downstream tools.

## Deliverables

- Memory profiling implementation in runtime/pre-runtime code
- JSON and CSV output schema documentation
- At least one sample output artifact under benchmark results
- README/dev_log references updated to point to this requirement
