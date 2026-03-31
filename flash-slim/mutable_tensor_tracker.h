// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef FLASH_SLIM_MUTABLE_TENSOR_TRACKER_H_
#define FLASH_SLIM_MUTABLE_TENSOR_TRACKER_H_

#include <cstddef>
#include <vector>

// Forward declaration — avoids pulling TFLite headers into every consumer.
namespace tflite { class Interpreter; }

namespace flash_slim {
namespace streaming {

// Per-node breakdown of tensor memory by allocation class.
struct NodeMutableMemoryInfo {
    int    node_index       = -1;
    size_t weight_bytes     = 0;  // kTfLiteMmapRo + kTfLitePersistentRo
    size_t activation_bytes = 0;  // kTfLiteArenaRw + kTfLiteArenaRwPersistent
    size_t kv_cache_bytes   = 0;  // kTfLiteCustom (KV cache via custom allocation)
};

// Aggregated memory profile for one subgraph, with per-node detail.
struct SubgraphMutableMemoryProfile {
    std::vector<NodeMutableMemoryInfo> nodes;
    size_t total_weight_bytes     = 0;
    size_t total_activation_bytes = 0;
    size_t total_kv_cache_bytes   = 0;
};

// Walk every node in `subgraph_idx` of `interpreter` and classify its
// input / output / temporary tensors by TfLiteAllocationType.
// Must be called AFTER AllocateTensors() so allocation types are finalized.
SubgraphMutableMemoryProfile CollectSubgraphMutableMemory(
    tflite::Interpreter* interpreter, int subgraph_idx);

}  // namespace streaming
}  // namespace flash_slim

#endif  // FLASH_SLIM_MUTABLE_TENSOR_TRACKER_H_
