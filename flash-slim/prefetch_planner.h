// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef FLASH_SLIM_PREFETCH_PLANNER_H_
#define FLASH_SLIM_PREFETCH_PLANNER_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "weight_chunk_prefetcher.h"  // WeightChunkInfo

namespace flash_slim {
namespace streaming {

// Configuration for the PrefetchPlanner.
struct PlannerConfig {
    // Target DRAM budget in bytes.
    // 0 (default) means "auto-compute the minimum feasible budget".
    size_t memory_budget_bytes = 0;
};

// One planned I/O group: a contiguous set of weight chunks that are loaded
// into the double-buffer together.
struct PlannedChunkGroup {
    size_t group_index        = 0;
    size_t total_weight_bytes = 0;  // sum of origin_size for all chunks in group
    size_t peak_memory_bytes  = 0;  // estimated peak DRAM while this group is active
    std::vector<size_t> chunk_indices;  // ordered by execution sequence
};

// Memory-aware prefetch planner.
//
// Inputs:
//   - An ordered list of WeightChunkInfo (one per XNNPACK kernel invocation,
//     in execution order).
//   - The total KV-cache bytes (constant across operators, pre-allocated at init).
//   - A PlannerConfig specifying the DRAM budget.
//
// The greedy grouping algorithm (adapted from
// test/conceptual_streaming_prototype/partitioner.py):
//   For each chunk i:
//     peak = accumulated_weight_in_current_group
//            + chunk[i].origin_size          (chunk being added)
//            + kv_cache_bytes                (always resident)
//            + chunk[i+1].origin_size        (lookahead: next group prefetch)
//     If peak > budget AND current group is non-empty → close group, start new.
class PrefetchPlanner {
 public:
    // Returns the theoretical minimum feasible budget:
    //   max over i of (chunk[i].origin_size + kv_cache_bytes + chunk[i+1].origin_size)
    // No plan can have a peak lower than this value (even with one-chunk groups).
    size_t ComputeMinimumPeakMemory(
        const std::vector<WeightChunkInfo>& chunks,
        size_t kv_cache_bytes) const;

    // Build chunk groups subject to config.memory_budget_bytes.
    // If budget == 0, ComputeMinimumPeakMemory is used as the budget.
    //
    // Returns:
    //   first  - vector of PlannedChunkGroup (all chunks assigned, in order)
    //   second - the budget that was actually used
    std::pair<std::vector<PlannedChunkGroup>, size_t> BuildPlan(
        const std::vector<WeightChunkInfo>& chunks,
        size_t kv_cache_bytes,
        const PlannerConfig& config) const;
};

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_PREFETCH_PLANNER_H_
