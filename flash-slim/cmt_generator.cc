#include "cmt_generator.h"
#include "mutable_tensor_tracker.h"
#include "prefetch_planner.h"

using flash_slim::streaming::CollectSubgraphMutableMemory;
using flash_slim::streaming::PrefetchPlanner;
using flash_slim::streaming::PlannerConfig;

// ----------------------
// absl::FLAGS definition
// ----------------------
ABSL_FLAG(std::string, tflite_model, "", "Two-signature tflite model for text generation using ODML tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to the SentencePiece model file.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt for the model.");
ABSL_FLAG(int, max_decode_steps, -1, "Number of tokens to generate. Defaults to the KV cache limit.");
ABSL_FLAG(std::string, start_token, "", "Optional start token appended to the beginning of the input prompt.");
ABSL_FLAG(std::string, stop_token, "", "Optional stop token that stops the decoding loop if encountered.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "", "Path for XNNPACK weight caching, e.g., /tmp/model.xnnpack_cache.");
ABSL_FLAG(std::string, lora_path, "", "Optional path to a LoRA artifact.");
ABSL_FLAG(float, temperature, 0.8f, "Temperature for sampling. Higher values make output more random. Defaults to 0.8");
ABSL_FLAG(int, top_k, 40, "Top-k sampling parameter. Only consider the top k tokens. Defaults to 40.");
ABSL_FLAG(float, top_p, 0.9f, "Top-p (nucleus) sampling parameter. Only consider tokens with cumulative probability <= top_p. Defaults to 0.9.");
ABSL_FLAG(float, repetition_penalty, 1.2f, "Repetition penalty for sampling. Higher values reduce repetition. Defaults to 1.2.");
ABSL_FLAG(bool, enable_repetition_penalty, false, "Enable repetition penalty. Defaults to false.");
ABSL_FLAG(std::string, csv_profile_output_path, "", "Path to save the profiling results in CSV format. If empty, no CSV output is generated.");
ABSL_FLAG(bool, dump_tensor_details, false, "Whether to dump detailed tensor information for each node.");
ABSL_FLAG(bool, op_tensor_byte_stats, false, "Whether to append per-operator aggregated tensor bytes (Mmap/Arena) on each operator line.");
ABSL_FLAG(std::string, model_dump_file_path, "", "Path to save the log file. If empty, no log file is generated.");
ABSL_FLAG(std::string, output_cmt_path, "weight_chunks_metadata_table.json", "Path to the weight chunk prefetch plan JSON file.");
ABSL_FLAG(int, profile_steps, 25, "Number of decoding steps to profile. If 0, profiling is disabled. default is 25.");
ABSL_FLAG(int64_t, memory_budget_bytes, 0,
    "DRAM budget (bytes) for memory-aware prefetch planning. "
    "0 (default) = automatically use the computed minimum feasible peak memory.");

namespace
{
    // Small helper to map: subgraph -> [tensor_idx -> buffer_id]
    // Maps a subgraph-local tensor index to the FlatBuffer buffer identifier.
    class TensorToBufferIdMap
    {
    public:
        void BuildFromFlatBufferModel(const tflite::FlatBufferModel *fb_model_holder)
        {
            map_.clear();
            if (!fb_model_holder)
                return;
            const ::tflite::Model *fb_model = fb_model_holder->GetModel();
            if (!fb_model)
                return;
            auto subgraphs = fb_model->subgraphs();
            if (!subgraphs)
                return;
            map_.resize(subgraphs->size());
            for (size_t i = 0; i < subgraphs->size(); ++i)
            {
                const ::tflite::SubGraph *subgraph = subgraphs->Get(i);
                auto tensors = subgraph ? subgraph->tensors() : nullptr;
                size_t n = tensors ? tensors->size() : 0;
                auto &vec = map_[i];
                vec.resize(n, -1);
                for (size_t i = 0; i < n; ++i)
                {
                    const ::tflite::Tensor *tensor = tensors->Get(i);
                    vec[i] = tensor ? static_cast<int>(tensor->buffer()) : -1;
                }
            }
        }

        int GetBufferId(int subgraph_idx, int tensor_idx) const
        {
            if (subgraph_idx < 0 || subgraph_idx >= static_cast<int>(map_.size()))
                return -1;
            const auto &v = map_[subgraph_idx];
            if (tensor_idx < 0 || tensor_idx >= static_cast<int>(v.size()))
                return -1;
            return v[tensor_idx];
        }

        // Number of subgraphs represented in the map (0 if empty)
        size_t NumSubgraphs() const { return map_.size(); }

    private:
        std::vector<std::vector<int>> map_;
    };

    // Human-readable byte formatting helper
    std::string FormatBytes(size_t bytes)
    {
        const double kb = 1024.0;
        const double mb = kb * 1024.0;
        const double gb = mb * 1024.0;
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(2);
        if (bytes >= (size_t)gb)
        {
            oss << (bytes / gb) << "GB";
        }
        else if (bytes >= (size_t)mb)
        {
            oss << (bytes / mb) << "MB";
        }
        else if (bytes >= (size_t)kb)
        {
            oss << (bytes / kb) << "KB";
        }
        else
        {
            oss.unsetf(std::ios::floatfield);
            oss << bytes << "B";
        }
        return oss.str();
    }
    // --------------------------------------------------------------------------
    // Print detailed tensor information (migrated from text_generator_main.cc)
    // --------------------------------------------------------------------------
    void PrintTensorDetail(int tensor_idx, TfLiteTensor *tensor, std::ostream *output = nullptr)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        if (!tensor)
        {
            out << "Tensor " << tensor_idx << " is NULL\n";
            return;
        }

        void *tensor_data_address = tensor->data.raw;
        out << "Data Address: " << tensor_data_address << " ";

        // Tensor Type
        const char *type_name = TfLiteTypeGetName(tensor->type);
        out << "Type: " << (type_name ? type_name : "Unknown") << " ";

        // Tensor Allocation Type
        out << "Allocation Type: ";
        switch (tensor->allocation_type)
        {
        case kTfLiteArenaRw:
            out << "Arena RW " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteArenaRwPersistent:
            out << "Arena Persistent " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteMmapRo:
            out << "Mmap " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteDynamic:
            out << "Dynamic " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteCustom:
            out << "Custom " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLitePersistentRo:
            out << "PersistentRo " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteVariantObject:
            out << "Variant " << "Bytes: " << tensor->bytes << " ";
            break;
        case kTfLiteMemNone:
            out << "MemNone " << "Bytes: 0 ";
            break;
        default:
            out << "Unknown " << "Bytes: 0 ";
            break;
        }

        // Tensor Shape
        out << "Shape: [";
        if (tensor->dims && tensor->dims->size > 0)
        {
            for (int dim_idx = 0; dim_idx < tensor->dims->size; ++dim_idx)
            {
                out << tensor->dims->data[dim_idx];
                if (dim_idx < tensor->dims->size - 1)
                    out << ", ";
            }
        }
        out << "]\n";
    }

    void InspectExecutionPlan(tflite::Interpreter *interpreter, int subgraph_idx, const TensorToBufferIdMap &tbm, std::ostream *output = nullptr, int indent = 0)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;
        std::string indent_str(indent, ' ');

        struct State
        {
            int32_t subgraph_index;
            bool subgraph_has_dynamic_output_tensors = false;
        };

        auto *subgraph = interpreter->subgraph(subgraph_idx);
        auto execution_plan = subgraph->execution_plan();

        int cnt_stablehlo_composite = 0;
        int cnt_atomic_ops = 0;     // 일반 atomic operator 개수
        int cnt_delegate_nodes = 0; // DELEGATE 노드 개수
        int cnt_xnn_ops = 0;        // XNNPACK operator 개수

        // Helper function to count XNNPACK operators from delegate output
        auto count_xnnpack_ops = [&](void *delegate_data) -> int
        {
            std::ostringstream temp_stream;
            TfLiteXNNPackDelegateInspect(delegate_data, &temp_stream, "");
            std::string delegate_output = temp_stream.str();

            int xnn_op_count = 0;
            std::istringstream iss(delegate_output);
            std::string line;

            while (std::getline(iss, line))
            {
                // XNNPACK operator lines typically start with "[XXXX]" format
                if (line.find("[") != std::string::npos && line.find("]") != std::string::npos)
                {
                    xnn_op_count++;
                }
            }
            return xnn_op_count;
        };

        // Recursive function to count nodes including nested STABLEHLO_COMPOSITE subgraphs
        std::function<int(int, int)> count_all_nodes = [&](int sg_idx, int depth) -> int
        {
            tflite::Subgraph *sg = interpreter->subgraph(sg_idx);
            auto plan = sg->execution_plan();
            int total = 0; // 0으로 시작하여 명시적으로 카운트

            for (int id : plan)
            {
                const auto *nr = sg->node_and_registration(id);
                auto *n = &nr->first;
                auto *r = &nr->second;

                if (r->builtin_code == tflite::BuiltinOperator_STABLEHLO_COMPOSITE)
                {
                    // count this composite node
                    cnt_stablehlo_composite++;

                    // STABLEHLO_COMPOSITE 노드 자체는 카운트하지 않고, 내부 서브그래프만 카운트
                    if (n->user_data)
                    {
                        auto *op_state = reinterpret_cast<State *>(n->user_data);
                        int child_subgraph_idx = op_state->subgraph_index;
                        total += count_all_nodes(child_subgraph_idx, depth + 1);
                    }
                }
                // DELEGATE 노드는 XNNPACK operators로 치환되므로 delegate 내부의 operator 개수를 세고, delegate 노드 자체는 제외
                else if (r->builtin_code == tflite::BuiltinOperator_DELEGATE)
                {
                    cnt_delegate_nodes++;
                    int xnn_ops = count_xnnpack_ops(n->user_data);
                    cnt_xnn_ops += xnn_ops;
                    total += xnn_ops;
                }
                // 일반 노드들만 카운트 (STABLEHLO_COMPOSITE, DELEGATE 제외)
                else
                {
                    total++;
                    cnt_atomic_ops++;
                }
            }

            return total;
        };

        int total_nodes_including_nested = count_all_nodes(subgraph_idx, 0);

        out << indent_str << "Subgraph " << subgraph_idx << ": " << subgraph->GetName() << " "
            << "(Total Nodes: " << total_nodes_including_nested
            << ", Nodes only in execution_plan: " << execution_plan.size()
            << ", StableHLO Composite Nodes: " << cnt_stablehlo_composite
            << ", DELEGATE Nodes: " << cnt_delegate_nodes
            << ", Atomic Nodes (Default ops): " << cnt_atomic_ops
            << ", XNNPACK Nodes: " << cnt_xnn_ops
            << ")" << std::endl;

        int absolute_op_counter = 0; // 절대 operator 번호 (DELEGATE 제외)
        for (int idx : execution_plan)
        {
            const auto *node_and_reg = subgraph->node_and_registration(idx);
            auto *node = &node_and_reg->first;
            auto *reg = &node_and_reg->second;
            std::string op_name =
                tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(reg->builtin_code));

            // Optional per-op tensor byte stats
            size_t op_bytes_mmap = 0;
            size_t op_bytes_arena = 0;
            if (absl::GetFlag(FLAGS_op_tensor_byte_stats))
            {
                auto collect = [&](const TfLiteIntArray *arr)
                {
                    if (!arr)
                        return;
                    for (int i = 0; i < arr->size; ++i)
                    {
                        int tidx = arr->data[i];
                        if (tidx < 0)
                            continue;
                        TfLiteTensor *t = subgraph->tensor(tidx);
                        if (!t)
                            continue;
                        switch (t->allocation_type)
                        {
                        case kTfLiteMmapRo:
                            op_bytes_mmap += t->bytes;
                            break;
                        case kTfLiteArenaRw:
                        case kTfLiteArenaRwPersistent:
                            op_bytes_arena += t->bytes;
                            break;
                        default:
                            break;
                        }
                    }
                };
                collect(node->inputs);
                collect(node->outputs);
                collect(node->temporaries);
            }

            out << indent_str << "  " << std::setw(4) << std::setfill(' ') << absolute_op_counter << ": "
                << "[" << std::setw(4) << std::setfill('0') << idx << "] " << op_name;
            if (absl::GetFlag(FLAGS_op_tensor_byte_stats))
            {
                out << " (Mmap=" << FormatBytes(op_bytes_mmap) << ", Arena=" << FormatBytes(op_bytes_arena) << ")";
            }
            absolute_op_counter++;

            // Dump detailed tensor information if requested
            if (absl::GetFlag(FLAGS_dump_tensor_details))
            {
                const auto *node_and_reg = subgraph->node_and_registration(idx);

                std::string tensor_indent_str(indent + 2, ' ');
                if (node_and_reg)
                {
                    const TfLiteNode *node = &node_and_reg->first;

                    // Print input tensors
                    if (node->inputs && node->inputs->size > 0)
                    {
                        out << std::endl
                            << tensor_indent_str << "    Input Tensors:" << std::endl;
                        for (int i = 0; i < node->inputs->size; ++i)
                        {
                            int tensor_idx = node->inputs->data[i];
                            if (tensor_idx >= 0)
                            {
                                auto *tensor = subgraph->tensor(tensor_idx);
                                int buffer_idx = tbm.GetBufferId(subgraph_idx, tensor_idx);
                                out << tensor_indent_str << "      Input " << i << ": " << tensor_idx
                                    << " (buffer " << buffer_idx << ") ";
                                PrintTensorDetail(tensor_idx, tensor, output);
                            }
                        }
                    }

                    // Print output tensors
                    if (node->outputs && node->outputs->size > 0)
                    {
                        out << tensor_indent_str << "    Output Tensors:" << std::endl;
                        for (int i = 0; i < node->outputs->size; ++i)
                        {
                            int tensor_idx = node->outputs->data[i];
                            if (tensor_idx >= 0)
                            {
                                auto *tensor = subgraph->tensor(tensor_idx);
                                int buffer_idx = tbm.GetBufferId(subgraph_idx, tensor_idx);
                                out << tensor_indent_str << "      Output " << i << ": " << tensor_idx
                                    << " (buffer " << buffer_idx << ") ";
                                PrintTensorDetail(tensor_idx, tensor, output);
                            }
                        }
                    }

                    // Print temporary tensors if any
                    if (node->temporaries && node->temporaries->size > 0)
                    {
                        out << tensor_indent_str << "    Temporary Tensors:" << std::endl;
                        for (int i = 0; i < node->temporaries->size; ++i)
                        {
                            int tensor_idx = node->temporaries->data[i];
                            if (tensor_idx >= 0)
                            {
                                auto *tensor = subgraph->tensor(tensor_idx);
                                int buffer_idx = tbm.GetBufferId(subgraph_idx, tensor_idx);
                                out << tensor_indent_str << "      Temporary " << i << ": " << tensor_idx
                                    << " (buffer " << buffer_idx << ") ";
                                PrintTensorDetail(tensor_idx, tensor, output);
                            }
                        }
                    }
                }
            }

            std::string delegate_indent_str(indent + 6, ' ');
            if (reg->builtin_code == tflite::BuiltinOperator_DELEGATE)
            {
                TfLiteXNNPackDelegateInspect(node->user_data, output, delegate_indent_str.c_str());
            }
            else if (reg->builtin_code == tflite::BuiltinOperator_STABLEHLO_COMPOSITE)
            {
                auto *op_state = reinterpret_cast<State *>(node->user_data);
                int subgraph_idx_child = op_state->subgraph_index;
                auto decomposition_subgraph = interpreter->subgraph(subgraph_idx_child);
                auto tmp_name = decomposition_subgraph->GetName();
                out << " (→ Subgraph " << subgraph_idx_child << " :" << tmp_name << ")" << std::endl;
                out << delegate_indent_str << "-------------------" << std::endl;
                // increase indent by 2 spaces for nested subgraph
                InspectExecutionPlan(interpreter, subgraph_idx_child, tbm, output, indent + 4);
                out << delegate_indent_str << "-------------------" << std::endl;
            }
            else
            {
                out << std::endl;
            }
        }
    }

    void InspectSignatureExecutionPlan(tflite::Interpreter *interpreter, const std::string &signature_key, const TensorToBufferIdMap &tbm, std::ostream *output = nullptr, int indent = 0)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        // Inspect the execution plan with signature context
        int subgraph_idx = interpreter->GetSubgraphIndexFromSignature(signature_key.c_str());
        if (subgraph_idx == -1)
        {
            out << "Failed to get subgraph index for: " << signature_key << std::endl;
            return;
        }
        out << std::endl;
        InspectExecutionPlan(interpreter, subgraph_idx, tbm, output, indent);
        out << std::endl; // spacing after subgraph dump
    }

    void InspectSelectedSignature(tflite::Interpreter *interpreter, int sig_index, const TensorToBufferIdMap &tbm, std::ostream *output = nullptr, int indent = 2)
    {
        std::ostream &out = (output != nullptr) ? *output : std::cout;

        const auto &signature_keys = interpreter->signature_keys();
        if (sig_index < 0 || sig_index >= signature_keys.size())
        {
            out << "Invalid signature index: " << sig_index << std::endl;
            return;
        }

        const std::string &signature_key = *signature_keys[sig_index];
        InspectSignatureExecutionPlan(interpreter, signature_key, tbm, output, indent);
    }

    int GetSignatureIndexFromUser(tflite::Interpreter *interpreter)
    {
        // Print model signature keys (console only)
        std::cout << "\n=== Model Signature ===" << std::endl;
        int sig_index = -1;
        if (interpreter)
        {
            const std::vector<const std::string *> &keys = interpreter->signature_keys();
            std::cout << "The Model contains " << keys.size() << " signature key(s)." << std::endl;
            if (!keys.empty())
            {
                for (int i = 0; i < keys.size(); ++i)
                {
                    const std::string *key = keys[i];
                    std::cout << "  " << std::setfill('0') << i << ": "
                              << *key << std::endl;
                }
            }

            std::cout << "Please select a signature index (0 to " << keys.size() - 1 << "): ";
            std::cin >> sig_index;
            if (sig_index > keys.size() - 1)
            {
                std::cout << "Invalid signature index. Please try again." << std::endl;
                sig_index = -1;
            }
        }
        else
        {
            std::cout << "The Model does not contain any signature keys.";
        }
        std::cout << std::endl;
        return sig_index;
    }

    // ----------------------------------------------------------------------------------
    // Validate Weight Cache Mappings
    // ----------------------------------------------------------------------------------
    void ValidateWeightCacheMappings(tflite::Interpreter *interpreter,
                                     const std::string &selected_signature_key,
                                     const TensorToBufferIdMap &tbm,
                                     StreamingWeightCacheProvider *weight_cache_provider,
                                     const std::string &save_path = "weight_cache_validation.log")
    {
        // 0) Get Subgraph and tensors
        int sg_idx = interpreter->GetSubgraphIndexFromSignature(selected_signature_key.c_str());
        tflite::Subgraph *sg = interpreter->subgraph(sg_idx);
        const size_t num_tensors = static_cast<size_t>(sg->tensors_size());

        // 1) Get TensorBufferAddress → identifier map (provider snapshot)
        auto buffer_address_to_identifier = weight_cache_provider->GetBufferAddressToIdentifier();

        // 2) Snapshot provider mappings once and reuse them per tensor.
        auto cache_key_to_offset = weight_cache_provider->GetCacheKeyToOffset();

        // Build helper maps used by the validation logic.
        // id_map: tensor_index -> weights_id
        std::unordered_map<size_t, size_t> id_map;
        id_map.reserve(num_tensors);

        for (size_t i = 0; i < num_tensors; ++i)
        {
            TfLiteTensor *t_ptr = sg->tensor(static_cast<int>(i));
            if (!t_ptr)
                continue;
            const TfLiteTensor &t = *t_ptr;

            if ((t.allocation_type != kTfLiteMmapRo && t.allocation_type != kTfLitePersistentRo) || !t.data.data)
                continue;

            // Prefer provider's address -> identifier mapping
            auto pit = buffer_address_to_identifier.find(t.data.data);
            if (pit != buffer_address_to_identifier.end())
            {
                id_map[i] = static_cast<size_t>(pit->second);
            }
            else
            {
                // Fallback to FlatBuffer buffer id
                int fb_id = tbm.GetBufferId(sg_idx, static_cast<int>(i));
                if (fb_id >= 0)
                {
                    id_map[i] = static_cast<size_t>(fb_id);
                }
            }
        }

        // Build reverse index: weights_id -> list of (pack_algorithm_id, bias_id)
        std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> weights_to_candidates;
        for (const auto &kv : cache_key_to_offset)
        {
            const auto &pack_id = kv.first; // has pack_algorithm_id, weights_id, bias_id
            const size_t w_id = static_cast<size_t>(pack_id.weights_id);
            const size_t algo = static_cast<size_t>(pack_id.pack_algorithm_id);
            const size_t bias = static_cast<size_t>(pack_id.bias_id);
            weights_to_candidates[w_id].emplace_back(algo, bias);
        }

        // 3) Validate a subset of constant tensors with cache LookUp/OffsetToAddr
        std::ofstream vout(save_path);
        vout << "\n=== Validation (signature=" << selected_signature_key << ") ===\n";
        vout << "\n";
        // Human-friendly aligned header
        vout << std::left << std::setw(8) << "Tensor" << " | "
             << std::setw(18) << "Ptr" << " | -> | "
             << std::setw(15) << "Pack Algo ID" << " | "
             << std::setw(18) << "Weights(Buffer) ID" << " | "
             << std::setw(11) << "Bias ID" << " | -> | "
             << std::setw(18) << "Offset" << " | -> | "
             << std::setw(18) << "Addr(used)" << "  ||  "
             << std::setw(18) << "MmappedAddr" << "\n";
        vout << std::string(160, '-') << "\n";
        size_t validated = 0, attempted = 0;

        for (size_t i = 0; i < num_tensors; ++i)
        {
            TfLiteTensor *t_ptr = sg->tensor(static_cast<int>(i));
            if (!t_ptr)
                continue;
            const TfLiteTensor &t = *t_ptr;

            if (t.allocation_type != kTfLiteMmapRo && t.allocation_type != kTfLitePersistentRo)
                continue;
            if (!t.data.data)
                continue;

            auto it_id = id_map.find(i);
            if (it_id == id_map.end())
                continue;
            size_t weights_id = it_id->second;

            auto it_cands = weights_to_candidates.find(weights_id);
            if (it_cands == weights_to_candidates.end() || it_cands->second.empty())
                continue;

            // Try candidate algos until a match is found
            bool ok = false;
            for (const auto &cand : it_cands->second)
            {
                const size_t algo = cand.first;
                const size_t bias_id = cand.second;
                size_t offset = weight_cache_provider->LookUpByIds(algo, weights_id, bias_id);
                if (offset == SIZE_MAX)
                    continue;

                void *addr = weight_cache_provider->OffsetToAddr(offset);
                void *mmaped_addr = weight_cache_provider->GetMmappedAddr(offset);
                {
                    std::ostringstream ptrs, useds, mmaps;
                    ptrs << t.data.data;
                    useds << addr;
                    mmaps << mmaped_addr;
                    const std::string bias_id_str = (bias_id == SIZE_MAX) ? std::string("None") : std::to_string(bias_id);
                    vout << std::left << std::setw(8) << i << " | "
                         << std::setw(18) << ptrs.str() << " | -> | "
                         << std::setw(15) << algo << " | "
                         << std::setw(18) << weights_id << " | "
                         << std::setw(11) << bias_id_str << " | -> | "
                         << std::setw(18) << offset << " | -> | "
                         << std::setw(18) << useds.str() << "  ||  "
                         << std::setw(18) << mmaps.str() << "\n";
                }
                ok = true;
                ++validated;
                break;
            }
            ++attempted;
            if (!ok)
            {
                vout << "Tensor[" << i << "] ptr=" << t.data.data
                     << " Buffer id=" << weights_id
                     << " -> cache MISS (no matching algo)\n";
            }
        }
        vout << "\n";
        vout << "Validation summary: validated=" << validated
             << " attempted=" << attempted << "\n";
        vout.close();
    }

}

// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    // Set precision
    std::cout.precision(5);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout << std::boolalpha;
    std::cout << "\n[INFO] Prefetch Planner " << std::endl;

    // Parse flags
    std::cout << "\n[INFO] Preparing Required Components" << std::endl;
    absl::ParseCommandLine(argc, argv);

    //* ============================================== Initialization ========================================================= */

    //* ============ Start Main ============ */
    // Declare local variables
    std::vector<int> prompt_tokens;
    std::string prompt, start_token, stop_token;
    std::unordered_set<int> previously_generated_tokens;
    TfLiteStatus status = kTfLiteOk;
    int stop_token_id = -1;
    {
        //* ============ [Phase] 1. Load Model ============ */
        std::unique_ptr<tflite::FlatBufferModel> model;
        model = tflite::FlatBufferModel::BuildFromFile(absl::GetFlag(FLAGS_tflite_model).c_str());
        MINIMAL_CHECK(model != nullptr);

        // Initialize tensor->buffer map from FlatBuffer model
        TensorToBufferIdMap tensor_buffer_map;
        tensor_buffer_map.BuildFromFlatBufferModel(model.get());

        //* ============ [Phase] 2. Build Interpreter ============ */
        std::unique_ptr<tflite::Interpreter> interpreter;
        // Register Ops
        tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
        tflite::ops::custom::GenAIOpsRegisterer(&resolver); // Register GenAI custom ops

        // Build the interpreter
        tflite::InterpreterBuilder builder(*model, resolver);
        builder.SetNumThreads(absl::GetFlag(FLAGS_num_threads));
        builder(&interpreter);
        MINIMAL_CHECK(interpreter != nullptr);

        //* ============ [Phase] 2.5. Setup Profiler ============ */
        // Init Tflite Internal Op-level Profiler
        std::unique_ptr<tflite::profiling::BufferedProfiler> op_profiler; // Create op_profiler pointer

        // Create profiler if profiling is enabled
        constexpr int kProfilingBufferHeadrooms = 512;
        int total_nodes = flash_slim::util::CountTotalNodes(interpreter.get());
        if (total_nodes > kProfilingBufferHeadrooms)
        {
            total_nodes += kProfilingBufferHeadrooms;
        }
        op_profiler = std::make_unique<tflite::profiling::BufferedProfiler>(total_nodes, true);

        // Set profiler to interpreter
        interpreter->SetProfiler(op_profiler.get());

        //* ============ [Phase] 3. Define Weight Cache Provider and Prefetcher ============ */
        std::unique_ptr<StreamingWeightCacheProvider> weight_cache_provider = std::make_unique<StreamingWeightCacheProvider>();
        std::unique_ptr<WeightChunkController> weight_chunk_controller = std::make_unique<WeightChunkController>(weight_cache_provider.get());
        std::unique_ptr<WeightChunkPrefetcher> weight_chunk_prefetcher = std::make_unique<WeightChunkPrefetcher>();


        weight_chunk_controller->UpdateProviderMode(StreamingWeightCacheProvider::ProviderMode::PRE_RUN_WARMUP);
        weight_chunk_controller->AttachPrefetcher(std::move(weight_chunk_prefetcher));


        // Json handler for weight chunk info
        JsonWeightChunkMetaDataWriter cmt_writer(absl::GetFlag(FLAGS_output_cmt_path).c_str());
        cmt_writer.WriteModelInfo(absl::GetFlag(FLAGS_tflite_model).c_str());
        weight_chunk_controller->AttachMetadataWriter(&cmt_writer);

        //* ============ [Phase] 3.5 Apply Delegate ============ */
        if (!absl::GetFlag(FLAGS_weight_cache_path).empty())
        {
            ApplyXNNPACKWithWeightCachingProvider(interpreter.get(), weight_cache_provider.get());
        }
        std::cout << std::endl;

        //* ============ [Phase] 4. Load Tokenizer ============ */
        std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
        sp_processor = LoadSentencePieceProcessor();

        //* ============ [Phase] 5. Allocate KV Cache ============ */
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
        kv_cache = AllocateKVCache(interpreter.get());
        MINIMAL_CHECK(!kv_cache.empty());

        //* ============ [Phase] 6. Prepare Prompt ============ */
        prompt = absl::GetFlag(FLAGS_prompt);
        MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

        // Initialize start and stop tokens
        start_token = absl::GetFlag(FLAGS_start_token);
        if (!start_token.empty())
        {
            prompt_tokens.insert(prompt_tokens.begin(), sp_processor->PieceToId(start_token));
        }

        stop_token = absl::GetFlag(FLAGS_stop_token);
        if (!stop_token.empty())
        {
            stop_token_id = sp_processor->PieceToId(stop_token);
        }

        // Initialize previously generated tokens with prompt tokens for repetition penalty
        if (absl::GetFlag(FLAGS_enable_repetition_penalty))
        {
            for (int token : prompt_tokens)
            {
                previously_generated_tokens.insert(token);
            }
        }
        std::cout << "[INFO] Stop token ID: " << stop_token_id << " for token: " << stop_token << std::endl;

        //* ============ [Phase] 7. Prepare Signature Runners ============ */

        tflite::SignatureRunner *prefill_runner = nullptr;
        tflite::SignatureRunner *decode_runner = nullptr;
        std::size_t effective_prefill_token_size = (prompt_tokens.size() > 0) ? (prompt_tokens.size() - 1) : 0;

        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::PREFILL);
        prefill_runner = GetPrefillRunner(interpreter.get(), effective_prefill_token_size, kv_cache, nullptr);

        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::DECODE);
        decode_runner = GetDecodeRunner(interpreter.get(), kv_cache, nullptr);

        MINIMAL_CHECK(prefill_runner != nullptr || decode_runner != nullptr);
        std::cout << "[INFO] Prefill Signature: " << prefill_runner->signature_key() << std::endl;
        std::cout << "[INFO] Decode Signature: " << decode_runner->signature_key() << std::endl;

        //* ============ [Phase] 7.5. Collect Mutable Tensor Memory Profiles ============ */
        // Walk each signature's subgraph to measure KV-cache and activation bytes.
        // Must be called after AllocateTensors() so allocation types are finalized.
        {
            auto collect_and_store = [&](tflite::SignatureRunner *runner,
                                         const std::string &mode_name)
            {
                if (!runner) return;
                const int sg_idx = interpreter->GetSubgraphIndexFromSignature(
                    runner->signature_key());
                if (sg_idx < 0)
                {
                    std::cerr << "[INFO] Cannot find subgraph for signature: "
                              << runner->signature_key() << std::endl;
                    return;
                }
                auto profile = CollectSubgraphMutableMemory(interpreter.get(), sg_idx);
                std::cout << "[INFO] Mutable memory profile [" << mode_name << "]:"
                          << " weight=" << profile.total_weight_bytes
                          << " activation=" << profile.total_activation_bytes
                          << " kv_cache=" << profile.total_kv_cache_bytes
                          << " bytes\n";
                cmt_writer.SetMutableMemoryProfile(mode_name, profile);
            };
            collect_and_store(prefill_runner, "PREFILL");
            collect_and_store(decode_runner, "DECODE");
        }

        //* ============ [Phase] 8. Prepare Input Tensors ============ */
        TfLiteTensor *prefill_input = nullptr;
        TfLiteTensor *prefill_input_pos = nullptr;
        TfLiteTensor *decode_input = nullptr;
        TfLiteTensor *decode_input_pos = nullptr;
        TfLiteTensor *kv_cache_k_0 = nullptr;
        int max_seq_size = 0;
        int kv_cache_max_size = 0;
        int prefill_seq_size = 0;
        int seq_dim_index = 0;

        prefill_input = prefill_runner->input_tensor("tokens");
        prefill_input_pos = prefill_runner->input_tensor("input_pos");
        decode_input = decode_runner->input_tensor("tokens");
        decode_input_pos = decode_runner->input_tensor("input_pos");
        kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");
        max_seq_size = prefill_input->dims->data[1];

        // Detect KV cache sequence dimension and set max size accordingly
        seq_dim_index = flash_slim::util::DetectKVCacheSequenceDimension(kv_cache_k_0);
        if (seq_dim_index >= 0 && seq_dim_index < kv_cache_k_0->dims->size)
        {
            kv_cache_max_size = kv_cache_k_0->dims->data[seq_dim_index];
        }
        else // Fallback to default behavior
        {
            kv_cache_max_size = kv_cache_k_0->dims->data[1];
        }

        prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);

        // Zero out the input tensors
        std::memset(prefill_input->data.i32, 0, prefill_input->bytes);
        std::memset(prefill_input_pos->data.i32, 0, prefill_input_pos->bytes);

        // Prefill uses all but the last token from the prompt
        for (int i = 0; i < prefill_seq_size - 1; ++i)
        {
            prefill_input->data.i32[i] = prompt_tokens[i];
            prefill_input_pos->data.i32[i] = i;
        }

        std::cout << "[INFO] KV Cache Max Size: " << kv_cache_max_size << " (from dimension index " << seq_dim_index << ")" << std::endl;

        //* ============================================== Model Dump  ========================================================= */

        //* ============ [Optional 1] Inspect Model ============ */

        std::string prefill_selected_signature_key = prefill_runner->signature_key();
        std::string decode_selected_signature_key = decode_runner->signature_key();

        std::ofstream dump_file_prefill(absl::GetFlag(FLAGS_model_dump_file_path)+"_prefill.log");
        std::ofstream dump_file_decode(absl::GetFlag(FLAGS_model_dump_file_path)+"_decode.log");
        if (!dump_file_prefill.is_open())
        {
            std::cerr << "❌ Failed to open log file: " << absl::GetFlag(FLAGS_model_dump_file_path) << "_prefill.log" << std::endl;
            return 1;
        }
        if (!dump_file_decode.is_open())
        {
            std::cerr << "❌ Failed to open log file: " << absl::GetFlag(FLAGS_model_dump_file_path) << "_decode.log" << std::endl;
            return 1;
        }
        dump_file_prefill << "\n=== After Applying Delegate ===" << std::endl;
        InspectSignatureExecutionPlan(interpreter.get(), prefill_selected_signature_key, tensor_buffer_map, &dump_file_prefill);
        dump_file_prefill.close();
        dump_file_decode << "\n=== After Applying Delegate ===" << std::endl;
        InspectSignatureExecutionPlan(interpreter.get(), decode_selected_signature_key, tensor_buffer_map, &dump_file_decode);
        dump_file_decode.close();

        //* ============ [Optional 2] Dump Weight Cache ============ */
        weight_cache_provider->DumpWeightCacheStructureToFile("weight_cache_structure.log");
        weight_cache_provider->DumpTensorIdentifierMapToFile("weight_cache_tensor_id_map.log");
        ValidateWeightCacheMappings(interpreter.get(),
                                    prefill_selected_signature_key,
                                    tensor_buffer_map,
                                    weight_cache_provider.get(),
                                    "weight_cache_validation.log");

        //* ============ [Optional 3] Buffer Test ============ */
        std::cout << "Verifying Buffer in weight cache" << std::endl;
        weight_cache_provider->VerifyAllBuffers();
        std::cout << "Verification done" << std::endl;

        //* ============================================== Generate Prefetch Plan ========================================================= */

        //* ============ [Phase] 9. Prefill Phase ============ */

        std::cout << "[INFO] Prefill Phase started" << std::endl;
        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::PREFILL);
        MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk); // Invoke the prefill runner

        std::cout << "[INFO] Prefill Phase completed" << std::endl;

        //* ============ [Phase] 10. Decoding Phase ============ */
        // Determine how many tokens to generate
        int max_decode_steps = (absl::GetFlag(FLAGS_max_decode_steps) == -1)
                                   ? kv_cache_max_size
                                   : absl::GetFlag(FLAGS_max_decode_steps);

        int next_token_id = prompt_tokens[prefill_seq_size - 1];
        int next_position = prefill_seq_size - 1;
        int decode_steps = std::min<int>(max_decode_steps, kv_cache_max_size - prefill_seq_size);

        std::cout << "[INFO] Tokens in Prompt: " << prompt_tokens.size() << "\n";
        std::cout << "[INFO] Tokens to Generate: " << decode_steps << "\n";
        std::cout << "[INFO] Limits of Tokens to Generate: " << kv_cache_max_size << "\n";
        std::cout << "\nPrompt:\n"
                  << prompt
                  << "\n\nOutput Text for Test:\n"
                  << std::endl;

        MINIMAL_CHECK(decode_steps > 0);

        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::DECODE);

        // Decode a single token
        std::string single_decoded_text;

        // 1) Model Inference
        decode_input->data.i32[0] = next_token_id;
        decode_input_pos->data.i32[0] = next_position;
        MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);

        // 2) Token Sampling
        next_token_id = flash_slim::sampler::temperature_top_k_top_p_sampler(
            decode_runner->output_tensor("logits"),
            absl::GetFlag(FLAGS_temperature),
            absl::GetFlag(FLAGS_top_k),
            absl::GetFlag(FLAGS_top_p));

        // 3) Token Detokenization
        std::vector<int> next_token = {next_token_id};
        MINIMAL_CHECK(sp_processor->Decode(next_token, &single_decoded_text).ok());

        std::cout << single_decoded_text << std::flush;

        std::cout << "\n\n\ntest finished.\n";
        std::cout << "---------------------------------------------------\n\n";
        
        //* ============ [Phase] 11. Profiling Phase ============ */

        std::cout << "---------------------------------------------------\n\n";
        std::cout << "[INFO] Profiling phase" << std::endl;

        // Reset Page Cache before profiling
        std::cout << "[INFO] Resetting page cache via /proc/sys/vm/drop_caches" << std::endl;

        // Release weight cache provider before dropping caches (unmap model file used before profiling)
        weight_cache_provider->Release();
        // Drop caches and wait for a moment
        flash_slim::util::drop_page_cache();
        sleep(1);
        
        // Print current page cache usage
        flash_slim::util::print_current_page_cache_kb();

        // Allocate weight chunk buffer for profiling
        size_t buf_size = cmt_writer.GetMaxAlignedSize();
        weight_chunk_controller->UpdateProviderMode(StreamingWeightCacheProvider::ProviderMode::PRE_RUN_PROFILE);
        weight_chunk_controller->AllocWeightChunkBuffer(buf_size);
        
        // Re-apply delegate with weight caching provider
        if (!absl::GetFlag(FLAGS_weight_cache_path).empty())
        {
            ApplyXNNPACKWithWeightCachingProvider(interpreter.get(), weight_cache_provider.get());
        }
        std::cout << std::endl;

        // Print current page cache usage
        flash_slim::util::print_current_page_cache_kb();

        // Start profiling prefill and decode separately
        // Internally, It uses dummy buffers and only measures pure computation time and io time. 
        // These measurements are for generate io prefecth plan only.
        std::cout << "[INFO] Profiling Prefill" << std::endl;
        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::PREFILL);
        for (int i = 0; i < absl::GetFlag(FLAGS_profile_steps); ++i)
        {
            prefill_runner->Invoke();
        }
        weight_chunk_controller->ResetBPFProbe();
        flash_slim::util::print_current_page_cache_kb();


        std::cout << "[INFO] Profiling Decode" << std::endl;
        weight_chunk_controller->UpdatePrefetcherMode(WeightChunkPrefetcher::PrefetchMode::DECODE);
        for (int i = 0; i < absl::GetFlag(FLAGS_profile_steps); ++i)
        {
            decode_runner->Invoke();
        }
        weight_chunk_controller->ResetBPFProbe();

        // Print current page cache usage
        flash_slim::util::print_current_page_cache_kb();

        //* ============ [Phase] 12. Memory-Aware Prefetch Planning ============ */
        {
            PrefetchPlanner planner;
            const size_t budget =
                static_cast<size_t>(absl::GetFlag(FLAGS_memory_budget_bytes));

            // Collect KV-cache size from the mutable profile of the decode subgraph.
            // The KV-cache tensors are classified as kTfLiteCustom, so they appear in
            // total_kv_cache_bytes.  If that is 0 (e.g. profile not collected or no
            // custom tensors), we use the PREFILL profile's kv_cache bytes as a fallback.
            // In the worst case we fall back to 0 (no KV-cache contribution), which
            // under-estimates peak memory but is still safe for grouping.
            const auto decode_profile = cmt_writer.GetMutableMemoryProfile("DECODE");
            const size_t kv_cache_bytes =
                (decode_profile.total_kv_cache_bytes > 0)
                    ? decode_profile.total_kv_cache_bytes
                    : cmt_writer.GetMutableMemoryProfile("PREFILL").total_kv_cache_bytes;

            for (const std::string &mode_name :
                     std::initializer_list<std::string>{"PREFILL", "DECODE"})
            {
                const auto &chunks = cmt_writer.GetRecordedChunks(mode_name);
                if (chunks.empty())
                {
                    std::cout << "[INFO] No chunks recorded for mode " << mode_name
                              << ", skipping planner.\n";
                    continue;
                }

                auto [groups, used_budget] = planner.BuildPlan(
                    chunks, kv_cache_bytes, PlannerConfig{budget});

                // Compute the maximum peak across all groups for the metadata field.
                size_t max_peak = 0;
                for (const auto &g : groups)
                    max_peak = std::max(max_peak, g.peak_memory_bytes);

                std::cout << "[INFO] Prefetch plan [" << mode_name << "]: "
                          << groups.size() << " groups, peak=" << max_peak
                          << " bytes, budget=" << used_budget << " bytes\n";

                cmt_writer.SetPrefetchPlan(mode_name, groups, max_peak, used_budget);
            }
        }

        // Release resources in reverse order of allocation
        cmt_writer.Finalize();
        weight_cache_provider->Release();
    }

    return 0;
}