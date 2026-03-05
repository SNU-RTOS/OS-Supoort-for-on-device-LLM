// Copyright 2025 Flash-SLiM Research Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <thread>
#include <limits>
#include <optional>
#include <utility>
#include <unistd.h>
#include <vector>

#include <pthread.h>

#include "tflite/minimal_logging.h"
#include "weight_chunk_prefetcher.h"

namespace flash_slim {
namespace streaming {

static constexpr uint32_t kPlanIndexShift = 30;
static constexpr size_t kPlanIndexMask = (static_cast<size_t>(1) << kPlanIndexShift) - static_cast<size_t>(1);

static inline size_t EncodeRangeId(int plan_index, size_t range_index) {
  return (static_cast<size_t>(plan_index) << kPlanIndexShift) | range_index;
}

static inline int DecodePlanIndex(size_t encoded) {
  return static_cast<int>(encoded >> kPlanIndexShift);
}

static inline size_t DecodeRangeIndex(size_t encoded) {
  return encoded & kPlanIndexMask;
}

/** =============== WeightChunkPrefetcher Class =============== */
WeightChunkPrefetcher::~WeightChunkPrefetcher() { StopWorker(); }

std::string WeightChunkPrefetcher::GetPrefetchModeString() const {
  return std::string(PrefetchModeName(prefetch_mode_));
}


// ============================================================================
// Configuration Methods
// ============================================================================
void WeightChunkPrefetcher::ConfigureIOEngine(
    const std::string& io_engine_mode_str,
    unsigned iouring_ring_depth, size_t iouring_subread_bytes,
    size_t pread_min_block_size, size_t pread_max_threads) {

  if (io_engine_mode_str == "io_uring") {
    io_engine_type_ = IOEngineType::IO_URING;
    iouring_ring_depth_ = iouring_ring_depth;
    iouring_subread_bytes_ = iouring_subread_bytes;
    
    std::cout << "[WeightChunkPrefetcher] Configured IO Engine - io_uring"
              << "\n    iouring_ring_depth=" << iouring_ring_depth_
              << "\n    iouring_subread_bytes=" << iouring_subread_bytes_ << std::endl;

  } else if (io_engine_mode_str == "parallel_pread") {
    io_engine_type_ = IOEngineType::PARALLEL_PREAD;
    pread_min_block_size_ = pread_min_block_size;
    pread_max_threads_ = pread_max_threads;
    
    std::cout << "[WeightChunkPrefetcher] Configured IO Engine - parallel_pread"
              << "\n    pread_min_block_size=" << pread_min_block_size_
              << "\n    pread_max_threads=" << pread_max_threads_ << std::endl;
              
  } else {
    std::cout << "[ERROR] WeightChunkPrefetcher: unknown IO engine mode '" 
              << io_engine_mode_str << "'; exiting." << std::endl;
    exit(1);
  }
}

void WeightChunkPrefetcher::ConfigureIOBuffers(void* buffer0, size_t size0, void* buffer1,
                                               size_t size1) {
  // Note: This is only required for io_uring mode
  std::lock_guard<std::mutex> lock(state_mutex_);
  io_registered_buffers_[0] = buffer0;
  io_registered_buffer_sizes_[0] = size0;
  io_registered_buffers_[1] = buffer1;
  io_registered_buffer_sizes_[1] = size1;
  io_buffers_configured_ = buffer0 != nullptr && buffer1 != nullptr && size0 > 0 && size1 > 0;
}

void WeightChunkPrefetcher::SetWorkerThreadAffinity(const std::vector<int>& cores) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  io_worker_cores_ = cores;
}

void WeightChunkPrefetcher::ApplyWorkerAffinity() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (io_worker_cores_.empty()) {
    return;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  for (int core : io_worker_cores_) {
    if (core >= 0) {
      CPU_SET(static_cast<unsigned long>(core), &set);
    }
  }
  
  if (io_submission_thread_.joinable()) {
    pthread_setaffinity_np(io_submission_thread_.native_handle(), sizeof(set), &set);
  }
  
  if (io_engine_type_ == IOEngineType::IO_URING && io_completion_thread_.joinable()) {
    pthread_setaffinity_np(io_completion_thread_.native_handle(), sizeof(set), &set);
  }
}

// ============================================================================
// Worker Management Methods
// ============================================================================

void WeightChunkPrefetcher::StartWorker() {
  bool expected = false;
  if (!io_worker_running_.compare_exchange_strong(expected, true)) {
    return;  // already running
  }

  io_worker_stop_requested_.store(false, std::memory_order_relaxed);

  // Dispatch based on configured IO engine type
  if (io_engine_type_ == IOEngineType::IO_URING) {
    StartIoUringWorker();
  } else if (io_engine_type_ == IOEngineType::PARALLEL_PREAD) {
    StartParallelPreadWorker();
  } else {
    io_worker_running_.store(false, std::memory_order_relaxed);
    std::cout << "[ERROR] WeightChunkPrefetcher: IO engine type not configured; call ConfigureIOEngine() first" << std::endl;
    return;
  }

  ApplyWorkerAffinity();
}

void WeightChunkPrefetcher::StopWorker() {
  if (!io_worker_running_.exchange(false)) {
    return;
  }

  io_worker_stop_requested_.store(true, std::memory_order_relaxed);
  
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    io_job_queue_.clear();
  }
  
  io_submission_cv_.notify_all();
  if (io_engine_type_ == IOEngineType::IO_URING) {
    io_completion_cv_.notify_all();
  }

  if (io_submission_thread_.joinable()) io_submission_thread_.join();
  if (io_completion_thread_.joinable()) io_completion_thread_.join();

  io_worker_stop_requested_.store(false, std::memory_order_relaxed);

  if (io_engine_type_ == IOEngineType::IO_URING && io_engine_) {
    io_engine_->Shutdown();
  }

  ResetChunkStates();
}


// * IO Engine Implementations * //

// ============================================================================
// io_uring Implementation
// ============================================================================

void WeightChunkPrefetcher::StartIoUringWorker() {
  void* buffer0 = nullptr;
  void* buffer1 = nullptr;
  size_t size0 = 0;
  size_t size1 = 0;
  bool configured = false;
  
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    buffer0 = io_registered_buffers_[0];
    buffer1 = io_registered_buffers_[1];
    size0 = io_registered_buffer_sizes_[0];
    size1 = io_registered_buffer_sizes_[1];
    configured = io_buffers_configured_;
  }

  if (!configured) {
    io_worker_running_.store(false, std::memory_order_relaxed);
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "WeightChunkPrefetcher: IO buffers not configured");
    return;
  }

  if (!io_engine_) {
    io_engine_ = std::make_unique<WeightChunkIOEngine>();
  }

  if (io_engine_ && !io_engine_->IsReady()) {
    std::cout << "[INFO] Initializing io_uring engine..." << std::endl;
    if (!io_engine_->Initialize(iouring_ring_depth_, iouring_subread_bytes_, 
                                buffer0, size0, buffer1, size1)) {
      std::cout << "[ERROR] Failed to initialize io_uring engine" << std::endl;
      io_worker_running_.store(false, std::memory_order_relaxed);
      return;
    }
  }

  if (io_engine_ && io_engine_->IsReady()) {
    std::cout << "[INFO] Starting io_uring submission and completion threads" << std::endl;
    io_submission_thread_ = std::thread([this]() { RunIoUringSubmissionLoop(); });
    io_completion_thread_ = std::thread([this]() { RunIoUringCompletionLoop(); });
  } else {
    io_worker_running_.store(false, std::memory_order_relaxed);
  }
}

void WeightChunkPrefetcher::RunIoUringSubmissionLoop() {
  while (true) {
    PrefetchJob job;
    
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      io_submission_cv_.wait(lock, [this]() {
        return io_worker_stop_requested_.load(std::memory_order_relaxed) || !io_job_queue_.empty();
      });

      if (io_worker_stop_requested_.load(std::memory_order_relaxed) && io_job_queue_.empty()) {
        break;
      }
      
      if (io_job_queue_.empty()) continue;
      
      job = std::move(io_job_queue_.front());
      io_job_queue_.pop_front();
    }

    const WeightChunkGroupInfo* group = job.chunk_group;
    if (!group || job.buffer_index < 0) {
      MarkJobCompleted(job, false);
      continue;
    }

    const int plan_index = PrefetchModeToIndex(job.mode);
    if (plan_index < 0) {
      MarkJobCompleted(job, false);
      continue;
    }

    WeightChunkIOEngine::IORequest request;
    request.range_index = EncodeRangeId(plan_index, group->group_index);
    request.aligned_offset = group->start_aligned_offset;
    request.aligned_size = group->total_aligned_size;
    request.buffer_base = job.buffer_base;
    request.direct_io_fd = job.direct_io_fd;
    request.buffer_index = job.buffer_index;

    bool submitted = false;
    {
      std::lock_guard<std::mutex> lock(engine_mutex_);
      submitted = io_engine_->Submit(request);
      if (submitted) {
        io_completion_cv_.notify_one();
      }
    }
    
    if (!submitted) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      io_job_queue_.push_front(std::move(job));
    }
  }
}

void WeightChunkPrefetcher::RunIoUringCompletionLoop() {
  while (true) {
    std::vector<WeightChunkIOEngine::Completion> completions;
    
    bool has_pending = false;
    bool queue_empty = false;
    
    {
      std::unique_lock<std::mutex> lock(engine_mutex_);
      
      io_completion_cv_.wait_for(lock, std::chrono::milliseconds(5), [this]() {
        return io_worker_stop_requested_.load(std::memory_order_relaxed) || 
               (io_engine_ && io_engine_->HasPending());
      });
      
      if (io_worker_stop_requested_.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> q_lock(queue_mutex_);
        if (io_job_queue_.empty() && (!io_engine_ || !io_engine_->HasPending())) {
          break;
        }
      }
      
      has_pending = io_engine_ && io_engine_->HasPending();
    }
    
    if (!has_pending) continue;
    
    if (io_engine_) {
      io_engine_->DrainCompletions(&completions, true);
    }

    for (const auto& completion : completions) {
      const int plan_index = DecodePlanIndex(completion.range_index);
      if (plan_index < 0 || plan_index >= 2 || !has_plan_[plan_index]) {
        continue;
      }
      
      const size_t range_index = DecodeRangeIndex(completion.range_index);
      const auto& plan = prefetch_plans_[plan_index];
      if (range_index >= plan.chunk_groups.size()) {
        continue;
      }
      
      const auto mode = IndexToPrefetchMode(plan_index);
      if (!mode.has_value()) {
        continue;
      }
      
      PrefetchJob job_snapshot;
      job_snapshot.chunk_group = &plan.chunk_groups[range_index];
      job_snapshot.mode = *mode;
      MarkJobCompleted(job_snapshot, completion.success);
    }
  }
}

// ============================================================================
// parallel_pread Implementation
// ============================================================================

void WeightChunkPrefetcher::StartParallelPreadWorker() {
  std::cout << "[INFO] Starting parallel pread worker thread" << std::endl;
  io_submission_thread_ = std::thread([this]() { RunParallelPreadWorkerLoop(); });
}

void WeightChunkPrefetcher::RunParallelPreadWorkerLoop() {
  while (true) {
    PrefetchJob job;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      io_submission_cv_.wait(lock, [this]() {
        return io_worker_stop_requested_.load(std::memory_order_relaxed) || !io_job_queue_.empty();
      });

      if (io_worker_stop_requested_.load(std::memory_order_relaxed) && io_job_queue_.empty()) {
        break;
      }

      job = std::move(io_job_queue_.front());
      io_job_queue_.pop_front();
    }

    const WeightChunkGroupInfo* group = job.chunk_group;
    const bool success = group && ExecuteParallelPread(
        job.direct_io_fd, job.buffer_base, group->total_aligned_size,
        static_cast<off_t>(group->start_aligned_offset));

    MarkJobCompleted(job, success);
  }
}

bool WeightChunkPrefetcher::ExecuteParallelPread(int fd, void* buffer, size_t size, off_t offset) {
  if (fd < 0 || buffer == nullptr || size == 0) {
    std::cerr << "[parallel_pread] invalid arguments: fd=" << fd
              << " buffer=" << buffer
              << " size=" << size
              << " offset=" << offset << std::endl;
    return false;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    const int err = errno;
    std::cerr << "[parallel_pread] fstat failed: errno=" << err
              << " (" << strerror(err) << ")" << std::endl;
  } else {
    if (offset + static_cast<off_t>(size) > st.st_size) {
      std::cerr << "[parallel_pread] WARN: request out of range: offset=" << offset
                << " size=" << size
                << " file_size=" << st.st_size << std::endl;
    }
  }

  if (size < pread_min_block_size_ * 2) {
    const ssize_t bytes_read = pread(fd, buffer, size, offset);
    if (bytes_read < 0) {
      const int err = errno;
      std::cerr << "[parallel_pread] pread (single) failed: offset=" << offset
                << " size=" << size
                << " errno=" << err
                << " (" << strerror(err) << ")" << std::endl;
    } else if (static_cast<size_t>(bytes_read) != size) {
      std::cerr << "[parallel_pread] pread (single) short read: offset=" << offset
                << " size=" << size
                << " bytes_read=" << bytes_read << std::endl;
    }
    return bytes_read > 0 && static_cast<size_t>(bytes_read) == size;
  }

  const size_t num_threads = std::min(pread_max_threads_, size / pread_min_block_size_);

  // For DIRECT_IO compatibility: enforce 4096-byte alignment for all sub-blocks.
  constexpr size_t kDirectIOAlignment = 4096;

  // Round per-thread size up to 4096 to ensure each block starts at 4096-aligned offset
  size_t base_block_size = size / num_threads;
  if (base_block_size % kDirectIOAlignment != 0) {
    base_block_size = ((base_block_size + kDirectIOAlignment - 1) / kDirectIOAlignment) * kDirectIOAlignment;
  }

  // Total size covered by aligned blocks (may be >= requested size)
  const size_t covered_size = base_block_size * num_threads;

  // Tail adjustment: we restrict actual read size of last block so that
  // 1) we don't read past the requested [offset, offset+size)
  // 2) we keep file offsets 4096-aligned for DIRECT_IO

  std::vector<std::future<bool>> futures;
  futures.reserve(num_threads);

  for (size_t i = 0; i < num_threads; ++i) {
    const size_t block_offset = i * base_block_size;

    // If this block would start beyond requested range, skip it
    if (block_offset >= size) {
      continue;
    }

    size_t current_block_size = base_block_size;
    // Clamp the last block to not exceed requested range
    if (block_offset + current_block_size > size) {
      current_block_size = size - block_offset;
      // If tail is not aligned and fd is DIRECT_IO, kernel may still reject it,
      // but offset alignment is preserved. This is acceptable for now because
      // plan's aligned ranges should already be 4096-multiple sized.
    }

    uint8_t* block_buffer = static_cast<uint8_t*>(buffer) + block_offset;
    const off_t block_file_offset = offset + block_offset;

    futures.emplace_back(std::async(std::launch::async, [=]() -> bool {
      const ssize_t bytes_read = pread(fd, block_buffer, current_block_size, block_file_offset);
      if (bytes_read < 0) {
        const int err = errno;
        std::cerr << "[parallel_pread] pread (block) failed: block_index=" << i
                  << " file_offset=" << block_file_offset
                  << " size=" << current_block_size
                  << " errno=" << err
                  << " (" << strerror(err) << ")" << std::endl;
      } else if (static_cast<size_t>(bytes_read) != current_block_size) {
        std::cerr << "[parallel_pread] pread (block) short read: block_index=" << i
                  << " file_offset=" << block_file_offset
                  << " size=" << current_block_size
                  << " bytes_read=" << bytes_read << std::endl;
      }
      return bytes_read > 0 && static_cast<size_t>(bytes_read) == current_block_size;
    }));
  }

  bool all_success = true;
  for (auto& future : futures) {
    if (!future.get()) {
      all_success = false;
    }
  }

  return all_success;
}

// ============================================================================
// Prefetch Plan Management Methods
// ============================================================================

void WeightChunkPrefetcher::SetPrefetchPlan(
  PrefetchMode mode, std::unordered_map<size_t, size_t>&& offset_to_index,
  std::vector<WeightChunkInfo>&& chunks,
  std::vector<WeightChunkGroupInfo>&& chunk_groups) {

  const int idx = PrefetchModeToIndex(mode);

  if (idx < 0) {
    return;
  }

  auto& plan = prefetch_plans_[idx];
  plan.offset_to_index = std::move(offset_to_index);
  plan.chunks = std::move(chunks);
  plan.chunk_groups = std::move(chunk_groups);

  std::sort(plan.chunk_groups.begin(), plan.chunk_groups.end(),
            [](const WeightChunkGroupInfo& lhs, const WeightChunkGroupInfo& rhs) {
              return lhs.group_index < rhs.group_index;
            });

  for (size_t group_index = 0; group_index < plan.chunk_groups.size(); ++group_index) {
    auto& group = plan.chunk_groups[group_index];
    if (group_index > kPlanIndexMask) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: group index %zu exceeds encoding capacity (%zu). Results may be "
          "undefined.",
          group_index, kPlanIndexMask);
    }
    if (group.group_index != group_index) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: adjusting group_index from %zu to %zu to maintain contiguity.",
          group.group_index, group_index);
      group.group_index = group_index;
    }
    
    // Build reverse index within group: chunk_index -> relative_offset
    group.chunk_to_relative_offset.clear();
    for (size_t i = 0; i < group.chunk_indices.size(); ++i) {
      if (i < group.chunk_relative_offsets.size()) {
        group.chunk_to_relative_offset[group.chunk_indices[i]] = group.chunk_relative_offsets[i];
      }
    }
  }

  has_plan_[idx] = true;
  index_to_group_io_states_[idx].clear();
  
  // Pre-allocate all IO states for lock-free runtime access
  PreallocateIOStates(idx);
  
  ResetRuntimeState();
}



bool WeightChunkPrefetcher::HasPrefetchPlan(PrefetchMode mode) const {
  const int idx = PrefetchModeToIndex(mode);
  return idx >= 0 && has_plan_[idx];
}

const WeightChunkPrefetcher::PrefetchPlan* WeightChunkPrefetcher::GetPrefetchPlan( PrefetchMode mode) const {
  const int idx = PrefetchModeToIndex(mode);
  if (idx < 0 || !has_plan_[idx]) {
    return nullptr;
  }
  return &prefetch_plans_[idx];
}

void WeightChunkPrefetcher::BuildIndexToChunksFromPlans() {
  size_t max_index = 0;
  bool seen_any = false;
  for (int i = 0; i < 2; ++i) {
    if (!has_plan_[i]) {
      continue;
    }
    for (const auto& chunk : prefetch_plans_[i].chunks) {
      if (!seen_any || chunk.chunk_index > max_index) {
        max_index = chunk.chunk_index;
      }
      seen_any = true;
    }
  }

  if (!seen_any) {
    index_to_chunks_.clear();
    return;
  }

  WeightChunkInfo tmp_chunk;
  tmp_chunk.chunk_index = SIZE_MAX;
  tmp_chunk.origin_offset = 0;
  tmp_chunk.origin_size = 0;
  tmp_chunk.aligned_offset = 0;
  tmp_chunk.aligned_size = 0;
  tmp_chunk.offset_adjust = 0;
  tmp_chunk.weights_id = 0;

  index_to_chunks_.assign(max_index + 1, tmp_chunk);

  for (int i = 0; i < 2; ++i) {
    if (!has_plan_[i]) {
      continue;
    }
    
    auto& plan = prefetch_plans_[i];
    
    // First pass: copy chunk data
    for (const auto& chunk : plan.chunks) {
      const size_t idx = chunk.chunk_index;
      auto& destination = index_to_chunks_[idx];
      if (destination.chunk_index == SIZE_MAX) {
        destination = chunk;
      } else if (destination.origin_offset != chunk.origin_offset) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_WARNING,
            "WeightChunkPrefetcher::BuildIndexToChunksFromPlans: conflict for "
            "chunk_index=%zu: existing origin_offset=%zu, new origin_offset=%zu."
            " Keeping existing.",
            idx, destination.origin_offset, chunk.origin_offset);
      }
    }
    
    // Second pass: set group pointers
    for (const auto& group : plan.chunk_groups) {
      for (const size_t chunk_index : group.chunk_indices) {
        if (chunk_index < index_to_chunks_.size()) {
          index_to_chunks_[chunk_index].group_per_mode[i] = &group;
        }
      }
    }
  }
}


// ============================================================================
// State Management Methods
// ============================================================================

std::shared_ptr<WeightChunkPrefetcher::ChunkIOState> WeightChunkPrefetcher::GetChunkIOState(size_t chunk_index) {
  // Lock-free read for pre-allocated states (hot path)
  // Safe because index_to_chunk_io_states_ is not modified after PreallocateIOStates()
  auto it = index_to_chunk_io_states_.find(chunk_index);
  if (it != index_to_chunk_io_states_.end()) {
    return it->second;
  }
  
  // Cold path: state not pre-allocated, need lock for insertion
  std::lock_guard<std::mutex> lock(state_mutex_);
  // Double-check after acquiring lock (another thread might have inserted)
  it = index_to_chunk_io_states_.find(chunk_index);
  if (it != index_to_chunk_io_states_.end()) {
    return it->second;
  }
  
  // Create on-demand (should rarely happen if pre-allocation works)
  auto& state = index_to_chunk_io_states_[chunk_index];
  state = std::make_shared<ChunkIOState>();
  return state;
}

std::shared_ptr<WeightChunkPrefetcher::ChunkGroupIOState> WeightChunkPrefetcher::GetChunkGroupIOState(
    PrefetchMode mode, size_t group_index) {
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || plan_idx >= static_cast<int>(index_to_group_io_states_.size())) {
    static auto dummy = std::make_shared<ChunkGroupIOState>();
    return dummy;
  }
  
  // Lock-free read for pre-allocated states (hot path)
  // Safe because index_to_group_io_states_ is not modified after PreallocateIOStates()
  auto& states_map = index_to_group_io_states_[plan_idx];
  auto it = states_map.find(group_index);
  if (it != states_map.end()) {
    return it->second;
  }
  
  // Cold path: state not pre-allocated, need lock for insertion
  std::lock_guard<std::mutex> lock(state_mutex_);
  // Double-check after acquiring lock
  it = states_map.find(group_index);
  if (it != states_map.end()) {
    return it->second;
  }
  
  // Create on-demand
  auto& state = states_map[group_index];
  state = std::make_shared<ChunkGroupIOState>();
  return state;
}


void WeightChunkPrefetcher::PreallocateIOStates(int plan_idx) {
  if (plan_idx < 0 || plan_idx >= kPrefetchPlanCount) {
    return;
  }
  
  const auto& plan = prefetch_plans_[plan_idx];
  
  // Pre-allocate chunk IO states
  for (const auto& chunk : plan.chunks) {
    if (index_to_chunk_io_states_.find(chunk.chunk_index) == index_to_chunk_io_states_.end()) {
      index_to_chunk_io_states_[chunk.chunk_index] = std::make_shared<ChunkIOState>();
    }
  }
  
  // Pre-allocate group IO states
  for (const auto& group : plan.chunk_groups) {
    if (index_to_group_io_states_[plan_idx].find(group.group_index) == 
        index_to_group_io_states_[plan_idx].end()) {
      index_to_group_io_states_[plan_idx][group.group_index] = std::make_shared<ChunkGroupIOState>();
    }
  }
}

void WeightChunkPrefetcher::ResetRuntimeState() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    io_job_queue_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (auto& states : index_to_group_io_states_) {
      states.clear();
    }
  }
  ResetChunkStates();
}

void WeightChunkPrefetcher::ResetChunkStates() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  for (auto& [_, state] : index_to_chunk_io_states_) {
    if (!state) continue;
    {
      std::lock_guard<std::mutex> s_lock(state->mutex);
      state->in_flight = false;
      state->ready = false;
      state->success = false;
    }
    state->cv.notify_all();
  }
}

// ============================================================================
// Job Completion Methods
// ============================================================================
void WeightChunkPrefetcher::MarkJobCompleted(const PrefetchJob& job, bool success) {
  const WeightChunkGroupInfo* group = job.chunk_group;
  if (!group) {
    return;
  }

  // Update chunk states atomically (lock-free)
  for (const size_t chunk_index : group->chunk_indices) {
    auto state = GetChunkIOState(chunk_index);
    if (!state) {
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->success = success;
    }
    state->ready.store(true, std::memory_order_release);
    state->in_flight.store(false, std::memory_order_release);
    state->cv.notify_all();
  }

  // Update group state atomically (lock-free)
  auto group_state = GetChunkGroupIOState(job.mode, group->group_index);
  if (group_state) {
    group_state->in_flight.store(false, std::memory_order_release);
  }

  if (!success && !group->chunk_indices.empty()) {
    const auto* info = GetChunkInfoByIndex(group->chunk_indices.front());
    const size_t origin_offset = info ? info->origin_offset : 0;
    const size_t chunk_index = info ? info->chunk_index : group->chunk_indices.front();
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
          "WeightChunkPrefetcher: prefetch failed (group_index=%zu, chunk_index=%zu, "
          "origin_offset=%zu)",
          group->group_index, chunk_index, origin_offset);
  }
}

// ============================================================================
// Controller Methods
// ============================================================================

const WeightChunkInfo* WeightChunkPrefetcher::LookupChunkInfo(PrefetchMode mode,
                                                                 size_t offset) const {
  const int idx = PrefetchModeToIndex(mode);
  return LookupChunkInfoByOffset(idx, offset);
}

const WeightChunkInfo* WeightChunkPrefetcher::LookupChunkInfoByOffset(int idx, size_t offset) const {
  if (idx < 0 || !has_plan_[idx]) {
    return nullptr;
  }

  const auto& plan = prefetch_plans_[idx];
  const auto it = plan.offset_to_index.find(offset);
  if (it == plan.offset_to_index.end()) {
    return nullptr;
  }

  const size_t index = it->second;
  if (index >= index_to_chunks_.size()) {
    return nullptr;
  }

  const auto& candidate = index_to_chunks_[index];
  if (candidate.chunk_index == std::numeric_limits<size_t>::max()) {
    return nullptr;
  }
  return &candidate;
}

const WeightChunkInfo* WeightChunkPrefetcher::GetChunkInfoByIndex(size_t chunk_index) const {
  if (chunk_index >= index_to_chunks_.size()) {
    return nullptr;
  }

  const auto& info = index_to_chunks_[chunk_index];
  if (info.chunk_index == std::numeric_limits<size_t>::max()) {
    return nullptr;
  }

  return &info;
}



const WeightChunkGroupInfo* WeightChunkPrefetcher::GetNextChunkGroup(
    PrefetchMode mode, size_t current_group_index, PrefetchMode* next_mode) const {

  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return nullptr;
  }

  const auto& plan = prefetch_plans_[plan_idx];
  if (plan.chunk_groups.empty()) {
    return nullptr;
  }

  const size_t group_count = plan.chunk_groups.size();
  const size_t next_index = current_group_index + 1;

  // Check if we need to transition from PREFILL to DECODE
  if (mode == PrefetchMode::PREFILL && next_index >= group_count) {
    // Decode index is always 1 (no need for PrefetchModeToIndex call)
    constexpr int decode_idx = 1;
    if (has_plan_[decode_idx]) {
      const auto& decode_plan = prefetch_plans_[decode_idx];
      if (!decode_plan.chunk_groups.empty()) {
        if (next_mode) {
          *next_mode = PrefetchMode::DECODE;
        }
        return &decode_plan.chunk_groups.front();
      }
    }
    // If no decode plan, wrap around to first group
    if (next_mode) {
      *next_mode = mode;
    }
    return &plan.chunk_groups.front();
  }

  // Normal case: next group within same mode
  const size_t target_index = next_index % group_count;
  
  // Fast path: groups are contiguous and sorted by group_index
  if (target_index < plan.chunk_groups.size() &&
      plan.chunk_groups[target_index].group_index == target_index) {
    if (next_mode) {
      *next_mode = mode;
    }
    return &plan.chunk_groups[target_index];
  }

  // Slow path: binary search for non-contiguous groups
  auto it = std::lower_bound(
      plan.chunk_groups.begin(), plan.chunk_groups.end(), target_index,
      [](const WeightChunkGroupInfo& group, size_t idx) {
        return group.group_index < idx;
      });
  
  if (it != plan.chunk_groups.end() && it->group_index == target_index) {
    if (next_mode) {
      *next_mode = mode;
    }
    return &*it;
  }

  // Fallback: wrap to first group (should rarely happen)
  if (next_mode) {
    *next_mode = mode;
  }
  return &plan.chunk_groups.front();
}


bool WeightChunkPrefetcher::Submit(const PrefetchRequest& request) {
  const WeightChunkGroupInfo* group = request.chunk_group;
  if (!request.buffer_base || request.direct_io_fd < 0 || request.buffer_index < 0 || 
      !group || group->chunk_indices.empty()) {
    return false;
  }

  PrefetchMode mode = request.mode == PrefetchMode::UNINITIALIZED ? prefetch_mode_ : request.mode;
  const int plan_idx = PrefetchModeToIndex(mode);
  if (plan_idx < 0 || !has_plan_[plan_idx]) {
    return false;
  }

  // Fast path: atomic check if already in flight
  auto group_state = GetChunkGroupIOState(mode, group->group_index);
  bool expected = false;
  if (!group_state->in_flight.compare_exchange_strong(expected, true, 
                                                       std::memory_order_acq_rel)) {
    return true;  // Already in flight
  }

  // Mark all chunks as in_flight using atomics (no lock needed for flags)
  for (const size_t chunk_index : group->chunk_indices) {
    auto state = GetChunkIOState(chunk_index);
    state->in_flight.store(true, std::memory_order_release);
    state->ready.store(false, std::memory_order_release);
    // success is non-atomic, only written under lock in MarkJobCompleted
  }

  // Enqueue job (only place we need lock)
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    PrefetchJob job;
    job.chunk_group = group;
    job.buffer_base = request.buffer_base;
    job.direct_io_fd = request.direct_io_fd;
    job.buffer_index = request.buffer_index;
    job.mode = mode;
    io_job_queue_.push_back(std::move(job));
  }

  io_submission_cv_.notify_one();
  return true;
}

bool WeightChunkPrefetcher::WaitReady(const WeightChunkInfo* chunk_info) {
  if (!chunk_info) {
    return false;
  }

  auto state = GetChunkIOState(chunk_info->chunk_index);
  std::unique_lock<std::mutex> lock(state->mutex);
  state->cv.wait(lock, [&]() { 
    return state->ready.load(std::memory_order_acquire) || 
           !state->in_flight.load(std::memory_order_acquire); 
  });
  
  // Return success status without resetting flags
  // Flags will be reset by next Submit() call
  return state->success;
}

}  // namespace streaming
}  // namespace flash_slim
