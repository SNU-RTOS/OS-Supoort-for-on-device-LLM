// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "mutable_tensor_tracker.h"

#include "tflite/core/subgraph.h"
#include "tflite/interpreter.h"

namespace flash_slim {
namespace streaming {

SubgraphMutableMemoryProfile CollectSubgraphMutableMemory(
    tflite::Interpreter* interpreter, int subgraph_idx) {
    SubgraphMutableMemoryProfile profile;
    if (!interpreter) return profile;

    tflite::Subgraph* subgraph = interpreter->subgraph(subgraph_idx);
    if (!subgraph) return profile;

    const auto& execution_plan = subgraph->execution_plan();
    for (int node_idx : execution_plan) {
        const auto* nr = subgraph->node_and_registration(node_idx);
        if (!nr) continue;
        const TfLiteNode* node = &nr->first;

        NodeMutableMemoryInfo info;
        info.node_index = node_idx;

        // Classifies one tensor index and accumulates into `info`.
        auto classify = [&](const TfLiteIntArray* arr) {
            if (!arr) return;
            for (int i = 0; i < arr->size; ++i) {
                const int tidx = arr->data[i];
                if (tidx < 0) continue;
                const TfLiteTensor* t = subgraph->tensor(tidx);
                if (!t || t->bytes == 0) continue;

                switch (t->allocation_type) {
                    case kTfLiteMmapRo:
                    case kTfLitePersistentRo:
                        info.weight_bytes += t->bytes;
                        break;
                    case kTfLiteArenaRw:
                    case kTfLiteArenaRwPersistent:
                        info.activation_bytes += t->bytes;
                        break;
                    case kTfLiteCustom:
                        // KV cache tensors bound via SetCustomAllocationForInputTensor.
                        info.kv_cache_bytes += t->bytes;
                        break;
                    default:
                        break;
                }
            }
        };

        classify(node->inputs);
        classify(node->outputs);
        classify(node->temporaries);

        profile.total_weight_bytes     += info.weight_bytes;
        profile.total_activation_bytes += info.activation_bytes;
        profile.total_kv_cache_bytes   += info.kv_cache_bytes;
        profile.nodes.push_back(std::move(info));
    }

    return profile;
}

}  // namespace streaming
}  // namespace flash_slim
