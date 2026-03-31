// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "prefetch_planner.h"

#include <algorithm>
#include <iostream>

namespace flash_slim {
namespace streaming {

size_t PrefetchPlanner::ComputeMinimumPeakMemory(
    const std::vector<WeightChunkInfo>& chunks,
    size_t kv_cache_bytes) const {
    size_t minimum = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        // Even if this chunk forms a group by itself, we still need to hold
        // the next group's first chunk in the inactive buffer (double-buffering).
        const size_t next_weight =
            (i + 1 < chunks.size()) ? chunks[i + 1].origin_size : 0;
        const size_t single_peak =
            chunks[i].origin_size + kv_cache_bytes + next_weight;
        minimum = std::max(minimum, single_peak);
    }
    return minimum;
}

std::pair<std::vector<PlannedChunkGroup>, size_t> PrefetchPlanner::BuildPlan(
    const std::vector<WeightChunkInfo>& chunks,
    size_t kv_cache_bytes,
    const PlannerConfig& config) const {

    size_t budget = config.memory_budget_bytes;
    if (budget == 0) {
        budget = ComputeMinimumPeakMemory(chunks, kv_cache_bytes);
        std::cout << "[PrefetchPlanner] memory_budget_bytes=0: using computed "
                     "minimum peak = "
                  << budget << " bytes\n";
    } else {
        const size_t minimum = ComputeMinimumPeakMemory(chunks, kv_cache_bytes);
        if (budget < minimum) {
            std::cerr << "[PrefetchPlanner] Warning: requested budget " << budget
                      << " bytes is below the minimum feasible peak " << minimum
                      << " bytes. Raising budget to minimum.\n";
            budget = minimum;
        }
    }

    std::vector<PlannedChunkGroup> groups;
    if (chunks.empty()) {
        return {groups, budget};
    }

    PlannedChunkGroup current;
    current.group_index = 0;
    size_t accumulated_weight = 0;
    size_t current_peak       = 0;

    for (size_t i = 0; i < chunks.size(); ++i) {
        const WeightChunkInfo& chunk = chunks[i];
        const size_t next_weight =
            (i + 1 < chunks.size()) ? chunks[i + 1].origin_size : 0;

        // Peak if we add this chunk to the current group.
        const size_t peak_if_added =
            accumulated_weight + chunk.origin_size + kv_cache_bytes + next_weight;

        if (!current.chunk_indices.empty() && peak_if_added > budget) {
            // Close the current group.
            current.total_weight_bytes = accumulated_weight;
            current.peak_memory_bytes  = current_peak;
            groups.push_back(std::move(current));

            // Start a new group with this chunk.
            current              = PlannedChunkGroup{};
            current.group_index  = groups.size();
            accumulated_weight   = 0;
            current_peak         = 0;
        }

        current.chunk_indices.push_back(chunk.chunk_index);
        accumulated_weight += chunk.origin_size;

        // Re-compute peak now that the chunk is in the group.
        const size_t peak_now =
            accumulated_weight + kv_cache_bytes + next_weight;
        current_peak = std::max(current_peak, peak_now);
    }

    // Flush the last group.
    if (!current.chunk_indices.empty()) {
        current.total_weight_bytes = accumulated_weight;
        current.peak_memory_bytes  = current_peak;
        groups.push_back(std::move(current));
    }

    std::cout << "[PrefetchPlanner] Built " << groups.size()
              << " chunk groups (budget=" << budget << " bytes, "
              << chunks.size() << " chunks)\n";

    return {groups, budget};
}

}  // namespace streaming
}  // namespace flash_slim
