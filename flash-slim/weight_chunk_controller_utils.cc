// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Prefetch Planner for Weight Streaming
// This module provides tools to record and plan I/O prefetch operations
// for weight chunks during model execution.

#include "weight_chunk_controller_utils.h"
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace flash_slim
{
    namespace streaming
    {
        //* ==================== JsonWeightChunkMetaDataWriter ==================== */

        JsonWeightChunkMetaDataWriter::JsonWeightChunkMetaDataWriter(const std::string &output_path)
            : output_path_(output_path),
              json_root_(nlohmann::ordered_json::object()),
              finalized_(false),
              max_aligned_size_(0),
              weight_chunk_buffer_size_(0)
        {
            // Initialize containers
            json_root_["weight_chunks"] = nlohmann::ordered_json::object();
            std::cout << "[JsonWeightChunkMetaDataWriter] Initialized: " << output_path_ << std::endl;
        }

        JsonWeightChunkMetaDataWriter::~JsonWeightChunkMetaDataWriter()
        {
            if (!finalized_)
            {
                std::cerr << "[JsonWeightChunkMetaDataWriter] Warning: Destroyed without calling Finalize()" << std::endl;
                Finalize();
            }
        }

        void JsonWeightChunkMetaDataWriter::WriteChunkInfo(
            const WeightChunkInfo &chunk_info,
            PrefetchMode prefetch_mode)
        {
            if (finalized_)
            {
                std::cerr << "[JsonWeightChunkMetaDataWriter] Error: Cannot write after Finalize()" << std::endl;
                return;
            }

            nlohmann::ordered_json chunk_json;
            chunk_json["chunk_index"] = chunk_info.chunk_index;
            chunk_json["origin_offset"] = chunk_info.origin_offset;
            chunk_json["origin_size"] = chunk_info.origin_size;
            chunk_json["aligned_offset"] = chunk_info.aligned_offset;
            chunk_json["aligned_size"] = chunk_info.aligned_size;
            chunk_json["offset_adjust"] = chunk_info.offset_adjust;
            chunk_json["weights_id"] = chunk_info.weights_id;
            chunk_json["prefetch_mode"] = static_cast<int>(prefetch_mode);
            chunk_json["prefetch_mode_str"] = WeightChunkPrefetcher::PrefetchModeName(prefetch_mode);

            // Update max aligned size
            if (chunk_info.aligned_size > max_aligned_size_)
            {
                max_aligned_size_ = chunk_info.aligned_size;
            }
            // Group by prefetch mode
            const std::string mode_key = WeightChunkPrefetcher::PrefetchModeName(prefetch_mode);
            if (!json_root_["weight_chunks"].contains(mode_key))
            {
                json_root_["weight_chunks"][mode_key] = nlohmann::ordered_json::array();
            }
            json_root_["weight_chunks"][mode_key].push_back(chunk_json);

            weight_chunk_buffer_size_ =
                std::max(weight_chunk_buffer_size_,
                         static_cast<size_t>(chunk_info.aligned_size));
            auto &last_aligned = last_chunk_aligned_size_[mode_key];
            if (last_aligned > 0)
            {
                weight_chunk_buffer_size_ =
                    std::max(weight_chunk_buffer_size_, last_aligned + chunk_info.aligned_size);
            }
            last_aligned = chunk_info.aligned_size;

            // Counters
            per_mode_counts_[mode_key] += 1;
            per_mode_total_aligned_size_[mode_key] += static_cast<size_t>(chunk_info.aligned_size);

            // Mirror into recorded_chunks_ for use in Finalize().
            recorded_chunks_[mode_key].push_back(chunk_info);
        }

        void JsonWeightChunkMetaDataWriter::Finalize()
        {
            if (finalized_)
            {
                return;
            }

            std::ofstream output_file(output_path_);
            if (!output_file.is_open())
            {
                std::cerr << "[JsonWeightChunkMetaDataWriter] Error: Failed to open: " << output_path_ << std::endl;
                finalized_ = true;
                return;
            }

            // Attach metadata before writing
            nlohmann::ordered_json meta;
            meta["version"] = "1.0";
            meta["max_aligned_size"] = max_aligned_size_;
            meta["weight_chunk_buffer_size"] = weight_chunk_buffer_size_;
            if (!model_path_.empty())
            {
                meta["model"] = model_path_;
            }
            // Per-mode counts
            nlohmann::ordered_json mode_counts = nlohmann::ordered_json::object();
            for (const auto &kv : per_mode_counts_)
            {
                mode_counts[kv.first] = kv.second;
            }
            meta["chunk_count_by_mode"] = mode_counts;

            // Per-mode total sizes
            nlohmann::ordered_json mode_total_sizes = nlohmann::ordered_json::object();
            for (const auto &kv : per_mode_total_aligned_size_)
            {
                mode_total_sizes[kv.first] = kv.second;
            }
            meta["total_aligned_size_by_mode"] = mode_total_sizes;

            // --- Mutable memory profile ---
            if (!mutable_profiles_.empty())
            {
                nlohmann::ordered_json mmp = nlohmann::ordered_json::object();
                for (const auto &kv : mutable_profiles_)
                {
                    nlohmann::ordered_json entry;
                    entry["total_weight_bytes"]     = kv.second.total_weight_bytes;
                    entry["total_activation_bytes"] = kv.second.total_activation_bytes;
                    entry["total_kv_cache_bytes"]   = kv.second.total_kv_cache_bytes;
                    mmp[kv.first] = std::move(entry);
                }
                meta["mutable_memory_profile"] = std::move(mmp);
            }

            // --- Planner output ---
            if (planned_peak_memory_bytes_ > 0)
            {
                meta["planned_peak_memory_bytes"] = planned_peak_memory_bytes_;
                meta["memory_budget_bytes"]       = memory_budget_bytes_;
            }

            // Build final root with metadata first, then weight_chunks (ordered_json preserves insertion order)
            nlohmann::ordered_json final_root = nlohmann::ordered_json::object();
            final_root["metadata"] = meta;
            if (json_root_.contains("weight_chunks"))
            {
                final_root["weight_chunks"] = std::move(json_root_["weight_chunks"]);
            }
            else
            {
                final_root["weight_chunks"] = nlohmann::ordered_json::object();
            }

            // --- prefetch_plan section ---
            // Serialize PlannedChunkGroups, computing byte-level offsets from
            // the recorded chunk metadata.
            if (!planned_groups_.empty())
            {
                nlohmann::ordered_json prefetch_plan = nlohmann::ordered_json::object();

                for (const auto &mode_kv : planned_groups_)
                {
                    const std::string &mode          = mode_kv.first;
                    const auto &groups               = mode_kv.second;
                    const auto &chunks_for_mode_it   = recorded_chunks_.find(mode);

                    // Build chunk_index -> WeightChunkInfo lookup.
                    std::unordered_map<size_t, const WeightChunkInfo *> idx_to_chunk;
                    if (chunks_for_mode_it != recorded_chunks_.end())
                    {
                        for (const auto &c : chunks_for_mode_it->second)
                        {
                            idx_to_chunk[c.chunk_index] = &c;
                        }
                    }

                    nlohmann::ordered_json mode_plan = nlohmann::ordered_json::object();
                    for (const auto &group : groups)
                    {
                        nlohmann::ordered_json g;

                        // Compute start offsets and total_aligned_size from chunk data.
                        size_t start_origin_offset  = 0;
                        size_t start_aligned_offset = 0;
                        size_t max_extent           = 0;
                        bool first_chunk_seen       = false;

                        std::vector<size_t> relative_offsets;
                        relative_offsets.reserve(group.chunk_indices.size());

                        for (size_t ci : group.chunk_indices)
                        {
                            auto it = idx_to_chunk.find(ci);
                            if (it == idx_to_chunk.end()) continue;
                            const WeightChunkInfo &c = *it->second;

                            if (!first_chunk_seen)
                            {
                                start_origin_offset  = c.origin_offset;
                                start_aligned_offset = c.aligned_offset;
                                first_chunk_seen     = true;
                            }
                            const size_t rel = (c.aligned_offset >= start_aligned_offset)
                                                   ? c.aligned_offset - start_aligned_offset
                                                   : 0;
                            relative_offsets.push_back(rel);
                            max_extent = std::max(max_extent, rel + c.aligned_size);
                        }

                        g["start_origin_offset"]    = start_origin_offset;
                        g["start_aligned_offset"]   = start_aligned_offset;
                        g["total_aligned_size"]     = max_extent;
                        g["peak_memory_bytes"]      = group.peak_memory_bytes;
                        g["chunk_indices"]          = group.chunk_indices;
                        g["chunk_relative_offsets"] = relative_offsets;

                        mode_plan[std::to_string(group.group_index)] = std::move(g);
                    }
                    prefetch_plan[mode] = std::move(mode_plan);
                }
                final_root["prefetch_plan"] = std::move(prefetch_plan);
            }

            // Write root object with pretty formatting (indent=2)
            output_file << final_root.dump(2) << std::endl;
            output_file.close();

            std::cout << "\n\n"
                      << "[JsonWeightChunkMetaDataWriter] Wrote chunk metadata to: " << output_path_ << std::endl;

            finalized_ = true;
        }

        void JsonWeightChunkMetaDataWriter::SetMutableMemoryProfile(
            const std::string &mode,
            const SubgraphMutableMemoryProfile &profile)
        {
            mutable_profiles_[mode] = profile;
        }

        void JsonWeightChunkMetaDataWriter::SetPrefetchPlan(
            const std::string &mode,
            const std::vector<PlannedChunkGroup> &groups,
            size_t planned_peak_memory_bytes,
            size_t memory_budget_bytes)
        {
            planned_groups_[mode]        = groups;
            planned_peak_memory_bytes_   = std::max(planned_peak_memory_bytes_,
                                                    planned_peak_memory_bytes);
            memory_budget_bytes_         = memory_budget_bytes;
        }

        //* ==================== JsonPrefetchPlanLoader ==================== */

        JsonPrefetchPlanLoader::JsonPrefetchPlanLoader()
            : root_(nlohmann::ordered_json::object()),
              version_(),
              model_path_(),
              max_aligned_size_(0),
              count_by_mode_(),
              size_by_mode_(),
              weight_chunks_()
        {
            std::cout << "[JsonPrefetchPlanLoader] Initialized" << std::endl;
        }
        JsonPrefetchPlanLoader::~JsonPrefetchPlanLoader()
        {
            Clear();
            std::cout << "[JsonPrefetchPlanLoader] Destroyed" << std::endl;
        }

        void JsonPrefetchPlanLoader::Clear()
        {
            root_.clear();
            version_.clear();
            model_path_.clear();
            max_aligned_size_ = 0;
            weight_chunk_buffer_size_ = 0;
            count_by_mode_.clear();
            size_by_mode_.clear();
            weight_chunks_.clear();
            io_order_groups_.clear();
        }

        std::vector<std::string> JsonPrefetchPlanLoader::KeysOf(const nlohmann::ordered_json &obj)
        {
            std::vector<std::string> keys;
            if (!obj.is_object())
                return keys;
            keys.reserve(obj.size());
            for (auto it = obj.begin(); it != obj.end(); ++it)
            {
                keys.emplace_back(it.key());
            }
            return keys;
        }

        bool JsonPrefetchPlanLoader::LoadFromFile(const std::string &path)
        {
            Clear();

            if (!LoadRootFromFile(path))
            {
                return false;
            }

            if (!ParseMetadataSection(root_.at("metadata")))
            {
                return false;
            }

            if (!ParseChunkGroups(root_.at("weight_chunks")))
            {
                return false;
            }

            if (root_.contains("prefetch_plan"))
            {
                if (!ParsePrefetchPlan(root_.at("prefetch_plan")))
                {
                    return false;
                }
            }
            else
            {
                if (!ParsePrefetchPlan(nlohmann::ordered_json::object()))
                {
                    return false;
                }
            }

            // Optional new sections — tolerate absence for backward compatibility.
            const auto &meta = root_.at("metadata");
            if (meta.contains("planned_peak_memory_bytes"))
            {
                planned_peak_memory_bytes_ = meta.at("planned_peak_memory_bytes").get<uint64_t>();
            }
            if (meta.contains("memory_budget_bytes"))
            {
                memory_budget_bytes_loader_ = meta.at("memory_budget_bytes").get<uint64_t>();
            }
            if (meta.contains("mutable_memory_profile"))
            {
                ParseMutableMemoryProfile(meta.at("mutable_memory_profile"));
            }

            DeriveWeightChunkBufferSize();

            return true;
        }

        bool JsonPrefetchPlanLoader::LoadRootFromFile(const std::string &path)
        {
            std::ifstream ifs(path);
            if (!ifs.is_open())
            {
                std::cerr << "[JsonPrefetchPlanLoader] failed to open file: " << path << std::endl;
                return false;
            }

            try
            {
                ifs >> root_;
            }
            catch (const std::exception &e)
            {
                std::cerr << "[JsonPrefetchPlanLoader] json parse error: " << e.what() << std::endl;
                return false;
            }

            if (!root_.is_object() || !root_.contains("metadata") || !root_.contains("weight_chunks"))
            {
                std::cerr << "[JsonPrefetchPlanLoader] invalid schema: expect { metadata, weight_chunks }" << std::endl;
                return false;
            }

            return true;
        }

        bool JsonPrefetchPlanLoader::ParseMetadataSection(const nlohmann::ordered_json &meta)
        {
            if (!meta.is_object())
            {
                std::cerr << "[JsonPrefetchPlanLoader] invalid metadata: expect object" << std::endl;
                return false;
            }

            if (meta.contains("version"))
            {
                version_ = meta.at("version").get<std::string>();
            }
            if (meta.contains("model"))
            {
                model_path_ = meta.at("model").get<std::string>();
            }
            if (meta.contains("max_aligned_size"))
            {
                max_aligned_size_ = meta.at("max_aligned_size").get<uint64_t>();
            }
            if (meta.contains("weight_chunk_buffer_size"))
            {
                weight_chunk_buffer_size_ = meta.at("weight_chunk_buffer_size").get<uint64_t>();
            }

            if (meta.contains("chunk_count_by_mode") && meta.at("chunk_count_by_mode").is_object())
            {
                for (auto it = meta.at("chunk_count_by_mode").begin(); it != meta.at("chunk_count_by_mode").end(); ++it)
                {
                    count_by_mode_[it.key()] = it.value().get<size_t>();
                }
            }

            if (meta.contains("total_aligned_size_by_mode") && meta.at("total_aligned_size_by_mode").is_object())
            {
                for (auto it = meta.at("total_aligned_size_by_mode").begin(); it != meta.at("total_aligned_size_by_mode").end(); ++it)
                {
                    size_by_mode_[it.key()] = it.value().get<uint64_t>();
                }
            }

            return true;
        }

        bool JsonPrefetchPlanLoader::ParseChunkGroups(const nlohmann::ordered_json &groups)
        {
            if (!groups.is_object())
            {
                std::cerr << "[JsonPrefetchPlanLoader] invalid schema: weight_chunks must be an object" << std::endl;
                return false;
            }

            for (auto git = groups.begin(); git != groups.end(); ++git)
            {
                const std::string mode = git.key();
                const auto &arr = git.value();
                if (!arr.is_array())
                {
                    continue;
                }

                auto &vec = weight_chunks_[mode];
                vec.reserve(arr.size());
                for (const auto &j : arr)
                {
                    WeightChunkInfo c{};
                    if (j.contains("chunk_index"))
                        c.chunk_index = static_cast<size_t>(j.at("chunk_index").get<uint64_t>());
                    if (j.contains("origin_offset"))
                        c.origin_offset = static_cast<size_t>(j.at("origin_offset").get<uint64_t>());
                    if (j.contains("origin_size"))
                        c.origin_size = static_cast<size_t>(j.at("origin_size").get<uint64_t>());
                    if (j.contains("aligned_offset"))
                        c.aligned_offset = static_cast<size_t>(j.at("aligned_offset").get<uint64_t>());
                    if (j.contains("aligned_size"))
                        c.aligned_size = static_cast<size_t>(j.at("aligned_size").get<uint64_t>());
                    if (j.contains("offset_adjust"))
                        c.offset_adjust = static_cast<size_t>(j.at("offset_adjust").get<uint64_t>());
                    if (j.contains("weights_id"))
                        c.weights_id = static_cast<size_t>(j.at("weights_id").get<uint64_t>());
                    vec.emplace_back(c);
                }
            }

            return true;
        }

        bool JsonPrefetchPlanLoader::ParsePrefetchPlan(const nlohmann::ordered_json &plan_root)
        {
            io_order_groups_.clear();

            if (!plan_root.is_object())
            {
                return true;
            }

            for (auto pit = plan_root.begin(); pit != plan_root.end(); ++pit)
            {
                const std::string mode = pit.key();
                const auto &mode_plan = pit.value();
                if (!mode_plan.is_object())
                {
                    continue;
                }

                auto &groups = io_order_groups_[mode];
                groups.clear();
                groups.reserve(mode_plan.size());

                for (auto it = mode_plan.begin(); it != mode_plan.end(); ++it)
                {
                    const auto &entry = it.value();
                    if (!entry.is_object())
                    {
                        continue;
                    }
                    WeightChunkGroupInfo group;
                    try
                    {
                        group.group_index = static_cast<size_t>(std::stoull(it.key()));
                    }
                    catch (const std::exception &)
                    {
                        std::cerr << "[JsonPrefetchPlanLoader] Warning: invalid group_index key \"" << it.key()
                                  << "\" for mode " << mode << std::endl;
                        continue;
                    }

                    if (entry.contains("start_origin_offset"))
                    {
                        group.start_origin_offset = entry.at("start_origin_offset").get<uint64_t>();
                    }
                    if (entry.contains("start_aligned_offset"))
                    {
                        group.start_aligned_offset = entry.at("start_aligned_offset").get<uint64_t>();
                    }
                    if (entry.contains("total_aligned_size"))
                    {
                        group.total_aligned_size = entry.at("total_aligned_size").get<uint64_t>();
                    }
                    if (entry.contains("chunk_indices") && entry.at("chunk_indices").is_array())
                    {
                        const auto &indices = entry.at("chunk_indices");
                        group.chunk_indices.reserve(indices.size());
                        for (const auto &idx_val : indices)
                        {
                            group.chunk_indices.push_back(static_cast<size_t>(idx_val.get<uint64_t>()));
                        }
                    }
                    if (entry.contains("chunk_relative_offsets") && entry.at("chunk_relative_offsets").is_array())
                    {
                        const auto &rel_offsets = entry.at("chunk_relative_offsets");
                        group.chunk_relative_offsets.clear();
                        group.chunk_relative_offsets.reserve(rel_offsets.size());
                        for (const auto &offset_val : rel_offsets)
                        {
                            group.chunk_relative_offsets.push_back(static_cast<size_t>(offset_val.get<uint64_t>()));
                        }
                    }
                    else
                    {
                        group.chunk_relative_offsets.clear();
                    }

                    groups.emplace_back(std::move(group));
                }

                std::sort(groups.begin(), groups.end(), [](const WeightChunkGroupInfo &lhs, const WeightChunkGroupInfo &rhs) {
                    return lhs.group_index < rhs.group_index;
                });

                if (!groups.empty())
                {
                    if (groups.front().group_index != 0)
                    {
                        std::cerr << "[JsonPrefetchPlanLoader] Invalid prefetch plan for mode " << mode
                                  << ": group_index values must start at 0" << std::endl;
                        return false;
                    }
                    for (size_t idx = 1; idx < groups.size(); ++idx)
                    {
                        if (groups[idx].group_index != groups[idx - 1].group_index + 1)
                        {
                            std::cerr << "[JsonPrefetchPlanLoader] Invalid prefetch plan for mode " << mode
                                      << ": group_index=" << groups[idx].group_index
                                      << " is not contiguous after " << groups[idx - 1].group_index << std::endl;
                            return false;
                        }
                    }
                }

                for (size_t group_index = 0; group_index < groups.size(); ++group_index)
                {
                    groups[group_index].group_index = group_index;
                }
            }

            return true;
        }

        void JsonPrefetchPlanLoader::DeriveWeightChunkBufferSize()
        {
            uint64_t derived_pair_max = 0;
            for (const auto &kv : weight_chunks_)
            {
                const auto &vec = kv.second;
                if (vec.empty())
                {
                    continue;
                }
                size_t mode_max = 0;
                for (const auto &chunk : vec)
                {
                    mode_max = std::max(mode_max, chunk.aligned_size);
                }
                size_t mode_pair_max = mode_max;
                for (size_t i = 0; i + 1 < vec.size(); ++i)
                {
                    const size_t pair_sum = vec[i].aligned_size + vec[i + 1].aligned_size;
                    mode_pair_max = std::max(mode_pair_max, pair_sum);
                }
                derived_pair_max = std::max<uint64_t>(derived_pair_max, mode_pair_max);
            }

            uint64_t derived_io_order_max = 0;
            for (const auto &kv : io_order_groups_)
            {
                for (const auto &group : kv.second)
                {
                    derived_io_order_max = std::max<uint64_t>(derived_io_order_max, group.total_aligned_size);
                }
            }

            const uint64_t derived_max = std::max<uint64_t>(derived_pair_max, derived_io_order_max);
            if (weight_chunk_buffer_size_ == 0)
            {
                weight_chunk_buffer_size_ = derived_max;
            }
            else if (derived_max > weight_chunk_buffer_size_)
            {
                std::cerr << "[JsonPrefetchPlanLoader] Warning: weight_chunk_buffer_size from metadata ("
                          << weight_chunk_buffer_size_
                          << ") is smaller than derived maximum requirement (" << derived_max
                          << "). Using derived value.\n";
                weight_chunk_buffer_size_ = derived_max;
            }
        }

        std::vector<std::string> JsonPrefetchPlanLoader::modes() const
        {
            std::vector<std::string> k = KeysOf(root_.at("weight_chunks"));
            return k;
        }

        const std::vector<WeightChunkInfo> &
        JsonPrefetchPlanLoader::chunks(const std::string &mode) const
        {
            static const std::vector<WeightChunkInfo> kEmpty;
            auto it = weight_chunks_.find(mode);
            if (it == weight_chunks_.end())
                return kEmpty;
            return it->second;
        }

        void JsonPrefetchPlanLoader::PrintMetadata(std::ostream &os) const
        {
            os << "[INFO] Prefetch Plan - Metadata" << std::endl;
            os << "  version: " << version_ << std::endl;
            if (!model_path_.empty())
                os << "  model: " << model_path_ << std::endl;
            os << "  max_aligned_size: " << max_aligned_size_ << std::endl;
            os << "  weight_chunk_buffer_size: " << weight_chunk_buffer_size_ << std::endl;

            os << "  chunk_count_by_mode:" << std::endl;
            for (const auto &kv : count_by_mode_)
            {
                os << "    - " << kv.first << ": " << kv.second << std::endl;
            }
            os << "  total_aligned_size_by_mode:" << std::endl;
            for (const auto &kv : size_by_mode_)
            {
                os << "    - " << kv.first << ": " << kv.second << std::endl;
            }
        }

        void JsonPrefetchPlanLoader::PrintMetadata() const
        {
            PrintMetadata(std::cout);
        }

        std::unordered_map<size_t, size_t>
        JsonPrefetchPlanLoader::BuildOffsetToIndexForMode(const std::string &mode) const
        {
            std::unordered_map<size_t, size_t> map;
            auto it = weight_chunks_.find(mode);
            if (it == weight_chunks_.end())
                return map;
            const auto &vec = it->second;
            map.reserve(vec.size());
            for (size_t i = 0; i < vec.size(); ++i)
            {
                const size_t key = vec[i].origin_offset;
                // first occurrence wins; if duplicates exist, keep first index
                map.emplace(key, i);
            }
            return map;
        }

        std::vector<WeightChunkInfo>
        JsonPrefetchPlanLoader::BuildIndexToChunkVectorForMode(const std::string &mode) const
        {
            std::vector<WeightChunkInfo> out;
            auto it = weight_chunks_.find(mode);
            if (it == weight_chunks_.end())
                return out;
            const auto &vec = it->second;
            out = vec; // 사본 반환
            return out;
        }

        const std::vector<WeightChunkGroupInfo> &
        JsonPrefetchPlanLoader::PrefetchChunkGroups(const std::string &mode) const
        {
            static const std::vector<WeightChunkGroupInfo> kEmpty;
            auto it = io_order_groups_.find(mode);
            if (it == io_order_groups_.end())
            {
                return kEmpty;
            }
            return it->second;
        }

        JsonPrefetchPlanLoader::ModeChunkPlan JsonPrefetchPlanLoader::BuildModeChunkPlan(const std::string &mode) const
        {
            JsonPrefetchPlanLoader::ModeChunkPlan plan;
            auto chunks_it = weight_chunks_.find(mode);
            if (chunks_it != weight_chunks_.end())
            {
                plan.chunks = chunks_it->second;
                plan.offset_to_index.reserve(plan.chunks.size());
                for (size_t i = 0; i < plan.chunks.size(); ++i)
                {
                    plan.offset_to_index.emplace(plan.chunks[i].origin_offset, i);
                }
            }

            auto groups_it = io_order_groups_.find(mode);
            if (groups_it != io_order_groups_.end())
            {
                plan.io_order_groups = groups_it->second;
            }

            if (!plan.chunks.empty() && !plan.io_order_groups.empty())
            {
                std::unordered_map<size_t, size_t> chunk_index_to_vector_index;
                chunk_index_to_vector_index.reserve(plan.chunks.size());
                for (size_t i = 0; i < plan.chunks.size(); ++i)
                {
                    chunk_index_to_vector_index.emplace(plan.chunks[i].chunk_index, i);
                }

                for (auto &group : plan.io_order_groups)
                {
                    if (group.chunk_indices.empty())
                    {
                        group.chunk_relative_offsets.clear();
                        continue;
                    }

                    group.chunk_relative_offsets.resize(group.chunk_indices.size(), 0);

                    const size_t first_chunk_index = group.chunk_indices.front();
                    auto first_it = chunk_index_to_vector_index.find(first_chunk_index);
                    bool start_initialized = false;
                    if (first_it != chunk_index_to_vector_index.end())
                    {
                        const auto &first_chunk = plan.chunks[first_it->second];
                        group.start_origin_offset = first_chunk.origin_offset;
                        group.start_aligned_offset = first_chunk.aligned_offset;
                        start_initialized = true;
                    }

                    size_t max_extent = 0;
                    for (size_t i = 0; i < group.chunk_indices.size(); ++i)
                    {
                        const size_t chunk_index = group.chunk_indices[i];
                        auto chunk_it = chunk_index_to_vector_index.find(chunk_index);
                        if (chunk_it == chunk_index_to_vector_index.end())
                        {
                            std::cerr << "[JsonPrefetchPlanLoader] Warning: chunk_index " << chunk_index
                                          << " referenced by group_index " << group.group_index
                                      << " not found in weight_chunks for mode " << mode << std::endl;
                            continue;
                        }
                        const auto &chunk = plan.chunks[chunk_it->second];
                        if (!start_initialized)
                        {
                            group.start_aligned_offset = chunk.aligned_offset;
                            group.start_origin_offset = chunk.origin_offset;
                            start_initialized = true;
                        }
                        const size_t relative_offset =
                            chunk.aligned_offset >= group.start_aligned_offset
                                ? chunk.aligned_offset - group.start_aligned_offset
                                : 0;
                        group.chunk_relative_offsets[i] = relative_offset;
                        max_extent = std::max(max_extent, relative_offset + chunk.aligned_size);
                    }

                    if (group.total_aligned_size == 0)
                    {
                        group.total_aligned_size = max_extent;
                    }
                    else if (max_extent > 0 && group.total_aligned_size < max_extent)
                    {
                        std::cerr << "[JsonPrefetchPlanLoader] Warning: total_aligned_size for group_index "
                                  << group.group_index << " (" << group.total_aligned_size
                                  << ") smaller than derived extent (" << max_extent
                                  << "). Using derived value.\n";
                        group.total_aligned_size = max_extent;
                    }
                }
            }

            return plan;
        }
        bool JsonPrefetchPlanLoader::ParseMutableMemoryProfile(
            const nlohmann::ordered_json &profile_root)
        {
            if (!profile_root.is_object()) return true; // tolerate missing section

            for (auto it = profile_root.begin(); it != profile_root.end(); ++it)
            {
                const std::string &mode = it.key();
                const auto &entry       = it.value();
                SubgraphMutableMemoryProfile p;
                if (entry.contains("total_weight_bytes"))
                    p.total_weight_bytes = entry.at("total_weight_bytes").get<uint64_t>();
                if (entry.contains("total_activation_bytes"))
                    p.total_activation_bytes = entry.at("total_activation_bytes").get<uint64_t>();
                if (entry.contains("total_kv_cache_bytes"))
                    p.total_kv_cache_bytes = entry.at("total_kv_cache_bytes").get<uint64_t>();
                mutable_profiles_[mode] = std::move(p);
            }
            return true;
        }

        SubgraphMutableMemoryProfile JsonPrefetchPlanLoader::mutable_memory_profile(
            const std::string &mode) const
        {
            auto it = mutable_profiles_.find(mode);
            if (it == mutable_profiles_.end()) return {};
            return it->second;
        }

    } // namespace streaming
} // namespace flash_slim
