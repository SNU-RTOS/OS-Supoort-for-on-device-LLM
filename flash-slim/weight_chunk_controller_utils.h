#ifndef FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_UTILS_H_
#define FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_UTILS_H_

#include <string>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <map>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

#include "mutable_tensor_tracker.h"
#include "prefetch_planner.h"
#include "weight_chunk_prefetcher.h"

namespace flash_slim::streaming
{


    //* ==================== Interfaces ==================== */
    class WeightChunkMetaDataWriter
    {
    public:
        virtual ~WeightChunkMetaDataWriter() = default;
        virtual void WriteChunkInfo(
            const WeightChunkInfo &chunk_info,
            PrefetchMode prefetch_mode) = 0;
        virtual void Finalize() = 0;
    };

    class PrefetchPlanLoader
    {

    public:
        virtual ~PrefetchPlanLoader() = default;
        virtual bool LoadFromFile(const std::string &plan_file_path) = 0;
    };

    //* ==================== JsonWeightChunkInfoWriter ==================== */

    // JSON-based implementation of WeightChunkMetaDataWriter
    // Uses nlohmann/json for serialization
    class JsonWeightChunkMetaDataWriter : public flash_slim::streaming::WeightChunkMetaDataWriter
    {
    public:
        explicit JsonWeightChunkMetaDataWriter(const std::string &output_path);
        ~JsonWeightChunkMetaDataWriter() override;

        // Delete copy constructor and assignment operator
        JsonWeightChunkMetaDataWriter(const JsonWeightChunkMetaDataWriter &) = delete;
        JsonWeightChunkMetaDataWriter &operator=(const JsonWeightChunkMetaDataWriter &) = delete;

        void WriteChunkInfo(
            const WeightChunkInfo &chunk_info,
            flash_slim::streaming::WeightChunkPrefetcher::PrefetchMode prefetch_mode) override;

        void Finalize() override;

        // Optional: write model info into metadata (e.g., model path)
        void WriteModelInfo(const std::string &model_path) { model_path_ = model_path; }

        size_t GetMaxAlignedSize() const { return max_aligned_size_; }

        // Read-back accessors (used by cmt_generator to feed the planner).
        const std::vector<WeightChunkInfo> &GetRecordedChunks(const std::string &mode) const
        {
            static const std::vector<WeightChunkInfo> kEmpty;
            auto it = recorded_chunks_.find(mode);
            return (it != recorded_chunks_.end()) ? it->second : kEmpty;
        }

        SubgraphMutableMemoryProfile GetMutableMemoryProfile(const std::string &mode) const
        {
            auto it = mutable_profiles_.find(mode);
            return (it != mutable_profiles_.end()) ? it->second : SubgraphMutableMemoryProfile{};
        }

        // Store the mutable-tensor memory profile collected for a given mode
        // (e.g. "PREFILL", "DECODE"). Serialized into "mutable_memory_profile".
        void SetMutableMemoryProfile(
            const std::string &mode,
            const SubgraphMutableMemoryProfile &profile);

        // Store a planned prefetch schedule produced by PrefetchPlanner.
        // Serializes into "prefetch_plan" in the output JSON.
        // planned_peak_memory_bytes : peak actually achieved by the plan
        // memory_budget_bytes       : budget that was requested / used
        void SetPrefetchPlan(
            const std::string &mode,
            const std::vector<PlannedChunkGroup> &groups,
            size_t planned_peak_memory_bytes,
            size_t memory_budget_bytes);

    private:
        std::string output_path_;
        nlohmann::ordered_json json_root_; // Root JSON object
        bool finalized_;
        size_t max_aligned_size_;
        size_t weight_chunk_buffer_size_;
        std::map<std::string, size_t> per_mode_counts_;
        std::map<std::string, size_t> per_mode_total_aligned_size_;
        std::map<std::string, size_t> last_chunk_aligned_size_;
        std::string model_path_;

        // Mirror of recorded chunks (needed to compute group offsets in Finalize).
        std::map<std::string, std::vector<WeightChunkInfo>> recorded_chunks_;

        // Planned prefetch schedule: mode -> groups.
        std::map<std::string, std::vector<PlannedChunkGroup>> planned_groups_;
        size_t planned_peak_memory_bytes_ = 0;
        size_t memory_budget_bytes_ = 0;

        // Mutable tensor profiles: mode -> profile.
        std::map<std::string, SubgraphMutableMemoryProfile> mutable_profiles_;
    };

    //* ==================== JsonPrefetchPlanLoader ==================== */
    class JsonPrefetchPlanLoader : public flash_slim::streaming::PrefetchPlanLoader
    {
    public:
        struct ModeChunkPlan
        {
            std::vector<WeightChunkInfo> chunks;
            std::unordered_map<size_t, size_t> offset_to_index;
            std::vector<WeightChunkGroupInfo> io_order_groups;
        };

        JsonPrefetchPlanLoader();
        ~JsonPrefetchPlanLoader() override;

        // JSON 파일 로드(성공 시 true)
        bool LoadFromFile(const std::string &path) override;

        // 메타데이터 접근자
        const std::string &version() const { return version_; }
        const std::string &model() const { return model_path_; }
        uint64_t max_aligned_size() const { return max_aligned_size_; }
        uint64_t weight_chunk_buffer_size() const { return weight_chunk_buffer_size_; }

        // 모드 목록("PREFILL", "DECODE", ...)
        std::vector<std::string> modes() const;

        // 모드별 chunk 벡터(모드가 없으면 빈 벡터 반환)
        const std::vector<WeightChunkInfo> &chunks(const std::string &mode) const;

        // 편의 접근자: PREFILL / DECODE 전용 벡터
        const std::vector<WeightChunkInfo> &prefill_chunks() const { return chunks("PREFILL"); }
        const std::vector<WeightChunkInfo> &decode_chunks() const { return chunks("DECODE"); }

        // 메타데이터 출력
        void PrintMetadata(std::ostream &os) const;

        // 표준 출력으로 메타데이터 출력 (구현은 .cc에서 std::cout 사용)
        void PrintMetadata() const;

        // 특정 모드: origin_offset -> index 맵(사본) 반환
        std::unordered_map<size_t, size_t> BuildOffsetToIndexForMode(const std::string &mode) const;

        // 특정 모드: index -> weight_chunk_info_t 벡터(사본) 반환
        std::vector<WeightChunkInfo> BuildIndexToChunkVectorForMode(const std::string &mode) const;

        // io_order 기반 청크 범위를 반환
        const std::vector<WeightChunkGroupInfo> &PrefetchChunkGroups(const std::string &mode) const;

        // 모드에 대한 전체 계획 구조 반환
        ModeChunkPlan BuildModeChunkPlan(const std::string &mode) const;

        // 모드별 개수/총 aligned_size
        const std::map<std::string, size_t> &chunk_count_by_mode() const { return count_by_mode_; }
        const std::map<std::string, uint64_t> &total_aligned_size_by_mode() const { return size_by_mode_; }

        // 원본 JSON에 접근이 필요하면 제공
        const nlohmann::ordered_json &raw_json() const { return root_; }

        // Accessors for new fields written by the planner.
        size_t planned_peak_memory_bytes() const { return planned_peak_memory_bytes_; }
        size_t memory_budget_bytes() const { return memory_budget_bytes_loader_; }

        // Returns the mutable memory profile for the given mode, or a zeroed
        // default if the JSON does not contain the section.
        SubgraphMutableMemoryProfile mutable_memory_profile(const std::string &mode) const;

    private:
        void Clear();
        bool LoadRootFromFile(const std::string &path);
        bool ParseMetadataSection(const nlohmann::ordered_json &meta);
        bool ParseChunkGroups(const nlohmann::ordered_json &groups);
        bool ParsePrefetchPlan(const nlohmann::ordered_json &plan_root);
        void DeriveWeightChunkBufferSize();

        std::string version_;
        nlohmann::ordered_json root_;
        std::string model_path_;
        uint64_t max_aligned_size_ = 0;
        uint64_t weight_chunk_buffer_size_ = 0;
        std::map<std::string, size_t> count_by_mode_;
        std::map<std::string, uint64_t> size_by_mode_;
        std::map<std::string, std::vector<WeightChunkInfo>> weight_chunks_;
        std::map<std::string, std::vector<WeightChunkGroupInfo>> io_order_groups_;

        // Fields populated from the new planner-written JSON sections.
        size_t planned_peak_memory_bytes_ = 0;
        size_t memory_budget_bytes_loader_ = 0;
        std::map<std::string, SubgraphMutableMemoryProfile> mutable_profiles_;

        bool ParseMutableMemoryProfile(const nlohmann::ordered_json &profile_root);

        static std::vector<std::string> KeysOf(const nlohmann::ordered_json &obj);
    };

} // namespace flash_slim::streaming

#endif // FLASH_SLIM_WEIGHT_CHUNK_CONTROLLER_UTILS_H_
