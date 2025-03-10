#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <random>

// included by nclee
#include <iostream>
#include <sys/mman.h>
#include <linux/perf_event.h>  // For perf_event_attr and PERF_* constants
#include <sys/syscall.h>       // For syscall and __NR_perf_event_open
#include <unistd.h>           // For syscall wrapper and pid_t
#include <sys/ioctl.h>
#include <unordered_map>
#include <stdexcept>
#include <sys/time.h>
#include <sys/resource.h>
#ifndef __NR_perf_event_open
#define __NR_perf_event_open 241  // Syscall number for aarch64
#endif


// AI EDGE TORCH
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/match.h"
#include "ai_edge_torch/generative/examples/cpp/utils.h"
#include "src/sentencepiece_processor.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

// ----------------------
// absl::FLAGS definition
// ----------------------
ABSL_FLAG(std::string, tflite_model, "",
          "Two-signature tflite model for text generation using ODML tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to the SentencePiece model file.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt for the model.");
ABSL_FLAG(int, max_decode_steps, -1,
          "Number of tokens to generate. Defaults to the KV cache limit.");
ABSL_FLAG(std::string, start_token, "",
          "Optional start token appended to the beginning of the input prompt.");
ABSL_FLAG(std::string, stop_token, "",
          "Optional stop token that stops the decoding loop if encountered.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "",
          "Path for XNNPACK weight caching, e.g., /tmp/model.xnnpack_cache.");
ABSL_FLAG(std::string, lora_path, "", "Optional path to a LoRA artifact.");

namespace
{

    using ai_edge_torch::examples::AlignedAllocator;
    using ai_edge_torch::examples::LoRA;

    // Performance metrics structure to store all relevant timing data
    struct PerfStats {
        // Wall clock time
        double wall_time_ms;
        
        // CPU time (from rusage)
        double user_time_sec;
        double system_time_sec;
        double cpu_time_sec;  // user + system
        
        // I/O time (from multiple sources)
        double io_wait_time_ms;
        double io_bytes_read;
        double io_bytes_written;
        
        // Per-core metrics (if available)
        std::vector<double> core_user_times;
        std::vector<double> core_system_times;
        std::vector<double> core_cpu_times;  // Add this missing field
        
        // New fields for timespec-based CPU time verification
        double process_cpu_time_sec;  // CPU time using clock_gettime(CLOCK_PROCESS_CPUTIME_ID)
        
        PerfStats() : wall_time_ms(0), user_time_sec(0), system_time_sec(0), 
                    cpu_time_sec(0), io_wait_time_ms(0), io_bytes_read(0), io_bytes_written(0),
                    process_cpu_time_sec(0) {}
    };

    // Helper function to convert timeval to seconds
    double toSeconds(const struct timeval& tv) {
        return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
    }

    // Helper function to convert timespec to seconds
    double timespecToSeconds(const struct timespec& ts) {
        return ts.tv_sec + (ts.tv_nsec / 1.0e9);
    }

    // Function to detect which cores the process is actually running on
    std::vector<int> detect_active_cores() {
        std::vector<int> cores;
        
        // Try to read process affinity mask
        cpu_set_t mask;
        CPU_ZERO(&mask);
        
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            for (int i = 0; i < CPU_SETSIZE; i++) {
                if (CPU_ISSET(i, &mask)) {
                    cores.push_back(i);
                }
            }
        }
        
        return cores;
    }

    // Function to measure I/O statistics from /proc filesystem
    struct IOStats {
        uint64_t bytes_read;
        uint64_t bytes_written;
        uint64_t read_ops;
        uint64_t write_ops;
    };

    IOStats get_io_stats() {
        IOStats stats = {0, 0, 0, 0};
        std::ifstream io_stat("/proc/self/io");
        if (!io_stat.is_open()) {
            return stats;
        }
        
        std::string line;
        while (std::getline(io_stat, line)) {
            if (line.find("read_bytes:") != std::string::npos) {
                stats.bytes_read = std::stoull(line.substr(line.find(":") + 1));
            }
            else if (line.find("write_bytes:") != std::string::npos) {
                stats.bytes_written = std::stoull(line.substr(line.find(":") + 1));
            }
            else if (line.find("syscr:") != std::string::npos) {
                stats.read_ops = std::stoull(line.substr(line.find(":") + 1));
            }
            else if (line.find("syscw:") != std::string::npos) {
                stats.write_ops = std::stoull(line.substr(line.find(":") + 1));
            }
        }
        
        return stats;
    }

    // Function to get CPU time for a specific core (if possible)
    // Note: This uses /proc/stat to get per-CPU statistics
    std::pair<double, double> get_core_cpu_time(int core_id) {
        std::ifstream stat_file("/proc/stat");
        if (!stat_file.is_open()) {
            return {0.0, 0.0};
        }
        
        std::string line;
        std::string cpu_prefix = "cpu" + std::to_string(core_id);
        
        while (std::getline(stat_file, line)) {
            if (line.find(cpu_prefix) == 0) {
                std::istringstream iss(line);
                std::string cpu_label;
                unsigned long user, nice, system, idle, iowait, irq, softirq, steal;
                
                iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
                
                // user + nice = user time, system + irq + softirq = system time
                double user_time = user + nice;
                double system_time = system + irq + softirq;
                
                return {user_time, system_time};
            }
        }
        
        return {0.0, 0.0};
    }

    // Performance monitoring class with improved measurements
    class PerformanceMonitor {
        private:
            // Add timespec storage for start times
            std::unordered_map<std::string, struct timespec> phase_start_process_time;
            
        
            // For wall clock time
            std::unordered_map<std::string, std::chrono::steady_clock::time_point> phase_start_times;
            
            // For CPU time via rusage
            std::unordered_map<std::string, rusage> phase_start_rusage;
            
            // For I/O stats
            std::unordered_map<std::string, IOStats> phase_start_io;
            
            // For per-core CPU times from /proc/stat
            std::unordered_map<std::string, std::vector<std::pair<double, double>>> phase_start_core_times;

            // For per-core monitoring
            struct CoreEventFds {
                std::vector<int> user_time_fds;
                std::vector<int> system_time_fds;
                std::vector<int> io_wait_fds;
                std::vector<int> cpu_cycles_fds;      // Add this missing field
                std::vector<int> cpu_instructions_fds; // Add this missing field
                std::vector<int> cpu_ref_cycles_fds;   // Add this missing field
            };
            
            std::unordered_map<std::string, CoreEventFds> phase_core_fds;
            std::vector<int> monitored_cores;
            
            // For per-core CPU time using timespec (new addition)
            struct CoreTimespec {
                std::vector<struct timespec> start_times;
            };
            std::unordered_map<std::string, CoreTimespec> phase_core_timespec;

            // Helper for perf_event_open syscall
            static long perf_event_open(struct perf_event_attr* hw_event, pid_t pid,
                                    int cpu, int group_fd, unsigned long flags) {
                return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
            }
            
            // Setup user time counter for a specific core
            int setup_user_time_counter(int core_id) {
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(struct perf_event_attr));
                pe.type = PERF_TYPE_SOFTWARE;
                pe.size = sizeof(struct perf_event_attr);
                pe.config = PERF_COUNT_SW_TASK_CLOCK;
                pe.exclude_kernel = 1;
                pe.exclude_hv = 1;
                pe.disabled = 1;
                pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                
                // Use getpid() to track only our process on the specified core
                int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
                if (fd == -1) {
                    std::cerr << "Warning: Failed to open user time perf event for core " 
                                << core_id << ": " << strerror(errno) << std::endl;
                }
                return fd;
            }
            
            // Setup system time counter for a specific core
            int setup_system_time_counter(int core_id) {
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(struct perf_event_attr));
                pe.type = PERF_TYPE_SOFTWARE;
                pe.size = sizeof(struct perf_event_attr);
                pe.config = PERF_COUNT_SW_TASK_CLOCK;
                pe.exclude_user = 1;
                pe.exclude_hv = 1;
                pe.disabled = 1;
                pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                
                // Use getpid() to track only our process on the specified core
                int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
                if (fd == -1) {
                    std::cerr << "Warning: Failed to open system time perf event for core " 
                                << core_id << ": " << strerror(errno) << std::endl;
                }
                return fd;
            }
            
            // Setup I/O wait counter
            int setup_io_wait_counter(int core_id) {
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(struct perf_event_attr));
                pe.type = PERF_TYPE_SOFTWARE;
                pe.size = sizeof(struct perf_event_attr);
                
                // Block I/O delay tracking
                pe.config = PERF_COUNT_SW_CPU_MIGRATIONS;  // As a proxy for I/O waits
                pe.disabled = 1;
                pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                
                int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
                if (fd == -1) {
                    std::cerr << "Warning: Failed to open I/O wait perf event for core " 
                                << core_id << ": " << strerror(errno) << std::endl;
                }
                return fd;
            }

            // Setup CPU cycles counter for a specific core
            int setup_cpu_cycles_counter(int core_id) {
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(struct perf_event_attr));
                pe.type = PERF_TYPE_HARDWARE;
                pe.size = sizeof(struct perf_event_attr);
                pe.config = PERF_COUNT_HW_CPU_CYCLES;
                pe.disabled = 1;
                pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                
                int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
                if (fd == -1) {
                    std::cerr << "Warning: Failed to open CPU cycles perf event for core " 
                            << core_id << ": " << strerror(errno) << std::endl;
                }
                return fd;
            }

            // Setup CPU instructions counter for a specific core
            int setup_cpu_instructions_counter(int core_id) {
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(struct perf_event_attr));
                pe.type = PERF_TYPE_HARDWARE;
                pe.size = sizeof(struct perf_event_attr);
                pe.config = PERF_COUNT_HW_INSTRUCTIONS;
                pe.disabled = 1;
                pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                
                int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
                if (fd == -1) {
                    std::cerr << "Warning: Failed to open CPU instructions perf event for core " 
                            << core_id << ": " << strerror(errno) << std::endl;
                }
                return fd;
            }

            // Setup CPU reference cycles counter for a specific core
            int setup_cpu_ref_cycles_counter(int core_id) {
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(struct perf_event_attr));
                pe.type = PERF_TYPE_HARDWARE;
                pe.size = sizeof(struct perf_event_attr);
                pe.config = PERF_COUNT_HW_REF_CPU_CYCLES;
                pe.disabled = 1;
                pe.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
                
                int fd = perf_event_open(&pe, getpid(), core_id, -1, 0);
                if (fd == -1) {
                    std::cerr << "Warning: Failed to open CPU reference cycles perf event for core " 
                            << core_id << ": " << strerror(errno) << std::endl;
                }
                return fd;
            }
            
            // Get system I/O wait percentage
            double get_system_io_wait() {
                std::ifstream stat_file("/proc/stat");
                if (!stat_file.is_open()) {
                    return 0.0;
                }
                
                std::string line;
                std::getline(stat_file, line);
                
                std::istringstream iss(line);
                std::string cpu_label;
                unsigned long user, nice, system, idle, iowait, irq, softirq, steal;
                
                iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
                
                unsigned long total_time = user + nice + system + idle + iowait + irq + softirq + steal;
                
                return (total_time > 0) ? (iowait * 100.0) / total_time : 0.0;
            }
        
        public:
            // Constructor that takes a list of core IDs to monitor
            PerformanceMonitor(const std::vector<int>& cores = {}) : monitored_cores(cores) {
                // If no cores specified, detect active cores
                if (monitored_cores.empty()) {
                    monitored_cores = detect_active_cores();
                    
                    // If still empty, default to core 0
                    if (monitored_cores.empty()) {
                        monitored_cores.push_back(0);
                    }
                }
                
                std::cout << "Performance monitor tracking cores: ";
                for (int core : monitored_cores) {
                    std::cout << core << " ";
                }
                std::cout << std::endl;
            }
            
            // Start monitoring a phase
            void start_phase(const std::string& phase_name) {
                // Record wall clock start time
                phase_start_times[phase_name] = std::chrono::steady_clock::now();
                
                // Record CPU time via rusage
                rusage start_rusage;
                getrusage(RUSAGE_SELF, &start_rusage);
                phase_start_rusage[phase_name] = start_rusage;
                
                // Record I/O stats
                phase_start_io[phase_name] = get_io_stats();
                
                // Add timespec measurements for CPU time
                struct timespec process_ts;
                if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_ts) == 0) {
                    phase_start_process_time[phase_name] = process_ts;
                } else {
                    std::cerr << "Warning: Failed to get CLOCK_PROCESS_CPUTIME_ID" << std::endl;
                }

                // Record per-core CPU times from /proc/stat
                std::vector<std::pair<double, double>> core_start_times;
                for (int core_id : monitored_cores) {
                    core_start_times.push_back(get_core_cpu_time(core_id));
                }
                phase_start_core_times[phase_name] = core_start_times;
                
                // NEW: Setup per-core timespec measurements
                CoreTimespec core_timespec;
                core_timespec.start_times.resize(monitored_cores.size());
                
                // For per-core CPU time measurement, we'll use perf_event_open with CPU counters
                // This is more reliable for per-core measurements than clock_gettime
                for (size_t i = 0; i < monitored_cores.size(); ++i) {
                    // Initialize with zeros in case we can't get actual measurements
                    struct timespec ts = {0, 0};
                    core_timespec.start_times[i] = ts;
                    
                    // We'll use the start time from perf_event counters later
                    // This timespec is primarily for backup and compatibility
                    clock_gettime(CLOCK_MONOTONIC, &ts);
                    core_timespec.start_times[i] = ts;
                }
                phase_core_timespec[phase_name] = core_timespec;

                // Setup per-core monitoring
                CoreEventFds core_fds;
                core_fds.user_time_fds.resize(monitored_cores.size(), -1);
                core_fds.system_time_fds.resize(monitored_cores.size(), -1);
                core_fds.io_wait_fds.resize(monitored_cores.size(), -1);
                core_fds.cpu_cycles_fds.resize(monitored_cores.size(), -1);
                core_fds.cpu_instructions_fds.resize(monitored_cores.size(), -1);
                core_fds.cpu_ref_cycles_fds.resize(monitored_cores.size(), -1);
                
                for (size_t i = 0; i < monitored_cores.size(); ++i) {
                    int core_id = monitored_cores[i];
                    
                    // Setup user time counters
                    core_fds.user_time_fds[i] = setup_user_time_counter(core_id);
                    if (core_fds.user_time_fds[i] != -1) {
                        ioctl(core_fds.user_time_fds[i], PERF_EVENT_IOC_RESET, 0);
                        ioctl(core_fds.user_time_fds[i], PERF_EVENT_IOC_ENABLE, 0);
                    }
                    
                    // Setup system time counters
                    core_fds.system_time_fds[i] = setup_system_time_counter(core_id);
                    if (core_fds.system_time_fds[i] != -1) {
                        ioctl(core_fds.system_time_fds[i], PERF_EVENT_IOC_RESET, 0);
                        ioctl(core_fds.system_time_fds[i], PERF_EVENT_IOC_ENABLE, 0);
                    }
                    
                    // Setup I/O wait counters
                    core_fds.io_wait_fds[i] = setup_io_wait_counter(core_id);
                    if (core_fds.io_wait_fds[i] != -1) {
                        ioctl(core_fds.io_wait_fds[i], PERF_EVENT_IOC_RESET, 0);
                        ioctl(core_fds.io_wait_fds[i], PERF_EVENT_IOC_ENABLE, 0);
                    }
                    
                    // Setup CPU cycles counter
                    // core_fds.cpu_cycles_fds[i] = setup_cpu_cycles_counter(core_id);
                    // if (core_fds.cpu_cycles_fds[i] != -1) {
                    //     ioctl(core_fds.cpu_cycles_fds[i], PERF_EVENT_IOC_RESET, 0);
                    //     ioctl(core_fds.cpu_cycles_fds[i], PERF_EVENT_IOC_ENABLE, 0);
                    // }
                    
                    // Setup CPU instructions counter
                    core_fds.cpu_instructions_fds[i] = setup_cpu_instructions_counter(core_id);
                    if (core_fds.cpu_instructions_fds[i] != -1) {
                        ioctl(core_fds.cpu_instructions_fds[i], PERF_EVENT_IOC_RESET, 0);
                        ioctl(core_fds.cpu_instructions_fds[i], PERF_EVENT_IOC_ENABLE, 0);
                    }
                    
                    // Setup CPU reference cycles counter
                    // core_fds.cpu_ref_cycles_fds[i] = setup_cpu_ref_cycles_counter(core_id);
                    // if (core_fds.cpu_ref_cycles_fds[i] != -1) {
                    //     ioctl(core_fds.cpu_ref_cycles_fds[i], PERF_EVENT_IOC_RESET, 0);
                    //     ioctl(core_fds.cpu_ref_cycles_fds[i], PERF_EVENT_IOC_ENABLE, 0);
                    // }
                }
                
                phase_core_fds[phase_name] = core_fds;
            }
            
            // End monitoring a phase and return statistics
            PerfStats end_phase(const std::string& phase_name) {
                PerfStats stats;
                
                // Check if the phase exists in all required maps
                auto time_it = phase_start_times.find(phase_name);
                auto rusage_it = phase_start_rusage.find(phase_name);
                auto io_it = phase_start_io.find(phase_name);
                auto core_fds_it = phase_core_fds.find(phase_name);
                auto core_times_it = phase_start_core_times.find(phase_name);
                auto core_timespec_it = phase_core_timespec.find(phase_name);
                auto process_time_it = phase_start_process_time.find(phase_name);
                
                // Handle missing phase records gracefully
                if (time_it == phase_start_times.end()) {
                    std::cerr << "Warning: Phase '" << phase_name << "' not found in time records. Skipping wall clock time measurement." << std::endl;
                } else {
                    // Calculate wall clock time
                    auto end_time = std::chrono::steady_clock::now();
                    stats.wall_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - time_it->second).count();
                    
                    // Clean up
                    phase_start_times.erase(time_it);
                }
                
                if (rusage_it == phase_start_rusage.end()) {
                    std::cerr << "Warning: Phase '" << phase_name << "' not found in rusage records. Skipping CPU time measurement." << std::endl;
                } else {
                    // Calculate CPU time from rusage
                    rusage end_rusage;
                    getrusage(RUSAGE_SELF, &end_rusage);
                    
                    stats.user_time_sec = toSeconds(end_rusage.ru_utime) - 
                                            toSeconds(rusage_it->second.ru_utime);
                    
                    stats.system_time_sec = toSeconds(end_rusage.ru_stime) - 
                                            toSeconds(rusage_it->second.ru_stime);
                    
                    stats.cpu_time_sec = stats.user_time_sec + stats.system_time_sec;
                    
                    // Clean up
                    phase_start_rusage.erase(rusage_it);
                }
                
                if (io_it == phase_start_io.end()) {
                    std::cerr << "Warning: Phase '" << phase_name << "' not found in I/O records. Skipping I/O measurement." << std::endl;
                } else {
                    // Calculate I/O stats from /proc/self/io
                    IOStats end_io = get_io_stats();
                    stats.io_bytes_read = end_io.bytes_read - io_it->second.bytes_read;
                    stats.io_bytes_written = end_io.bytes_written - io_it->second.bytes_written;
                    
                    // Estimate I/O wait time based on I/O volume and throughput
                    uint64_t total_io_bytes = stats.io_bytes_read + stats.io_bytes_written;
                    uint64_t total_io_ops = (end_io.read_ops - io_it->second.read_ops) + 
                                            (end_io.write_ops - io_it->second.write_ops);
                    
                    // A simple heuristic: if there was I/O activity, estimate wait time
                    if (total_io_bytes > 0) {
                        double io_throughput = 100.0 * 1024 * 1024; // Assume 100 MB/s
                        stats.io_wait_time_ms = (total_io_bytes / io_throughput) * 1000.0;
                        
                        // Don't let I/O wait exceed wall time
                        if (stats.wall_time_ms > 0) {
                            stats.io_wait_time_ms = std::min(stats.io_wait_time_ms, stats.wall_time_ms * 0.9);
                        }
                    }
                    
                    // Clean up
                    phase_start_io.erase(io_it);
                }
                
                if (process_time_it != phase_start_process_time.end()) {
                    struct timespec end_process_ts;
                    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_process_ts) == 0) {
                        stats.process_cpu_time_sec = timespecToSeconds(end_process_ts) - 
                                                timespecToSeconds(process_time_it->second);
                    }
                    // Cleanup
                    phase_start_process_time.erase(process_time_it);
                }
                
                // Handle per-core metrics
                if (core_times_it == phase_start_core_times.end()) {
                    std::cerr << "Warning: Phase '" << phase_name << "' not found in core times records. Skipping per-core measurements." << std::endl;
                } else {
                    // Get per-core CPU times from /proc/stat for comparison
                    std::vector<std::pair<double, double>> core_end_times;
                    for (int core_id : monitored_cores) {
                        core_end_times.push_back(get_core_cpu_time(core_id));
                    }
                    
                    // Calculate per-core CPU utilization from /proc/stat
                    for (size_t i = 0; i < monitored_cores.size(); ++i) {
                        double user_delta = core_end_times[i].first - core_times_it->second[i].first;
                        double system_delta = core_end_times[i].second - core_times_it->second[i].second;
                        
                        // Store in our stats structure (these are in jiffies, need to convert to seconds)
                        // On most Linux systems, there are 100 jiffies per second
                        const double JIFFIES_PER_SEC = 100.0;
                        stats.core_user_times.push_back(user_delta / JIFFIES_PER_SEC);
                        stats.core_system_times.push_back(system_delta / JIFFIES_PER_SEC);
                    }
                    
                    // Clean up
                    phase_start_core_times.erase(core_times_it);
                }
                
                // Initialize per-core CPU times
                stats.core_cpu_times.resize(monitored_cores.size(), 0.0);
                
                if (core_fds_it == phase_core_fds.end()) {
                    std::cerr << "Warning: Phase '" << phase_name << "' not found in core fds records. Skipping perf event measurements." << std::endl;
                } else {
                    // Read per-core metrics from perf events
                    auto& core_fds = core_fds_it->second;
                    
                    // Structure to read perf event results
                    struct read_format {
                        uint64_t value;
                        uint64_t time_enabled;
                        uint64_t time_running;
                    };
                    
                    for (size_t i = 0; i < monitored_cores.size(); ++i) {
                        double core_user_time = 0.0;
                        double core_system_time = 0.0;
                        
                        // Read and disable user time counter
                        if (core_fds.user_time_fds[i] != -1) {
                            struct read_format rf;
                            if (read(core_fds.user_time_fds[i], &rf, sizeof(rf)) == sizeof(rf)) {
                                ioctl(core_fds.user_time_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                                // Convert from ns to sec
                                core_user_time = rf.value / 1000000000.0;
                                
                                // Update our stats with the more accurate perf event data
                                if (i < stats.core_user_times.size()) {
                                    stats.core_user_times[i] = core_user_time;
                                }
                            }
                            close(core_fds.user_time_fds[i]);
                        }
                        
                        // Read and disable system time counter
                        if (core_fds.system_time_fds[i] != -1) {
                            struct read_format rf;
                            if (read(core_fds.system_time_fds[i], &rf, sizeof(rf)) == sizeof(rf)) {
                                ioctl(core_fds.system_time_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                                // Convert from ns to sec
                                core_system_time = rf.value / 1000000000.0;
                                
                                // Update our stats with the more accurate perf event data
                                if (i < stats.core_system_times.size()) {
                                    stats.core_system_times[i] = core_system_time;
                                }
                            }
                            close(core_fds.system_time_fds[i]);
                        }
                        
                        // Update total CPU time for this core
                        if (i < stats.core_cpu_times.size()) {
                            stats.core_cpu_times[i] = stats.core_user_times[i] + stats.core_system_times[i];
                        }
                        
                        // Read and disable I/O wait counter
                        if (core_fds.io_wait_fds[i] != -1) {
                            uint64_t count;
                            if (read(core_fds.io_wait_fds[i], &count, sizeof(count)) == sizeof(count)) {
                                ioctl(core_fds.io_wait_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                                // Each CPU migration contributes to I/O wait time estimate
                                if (count > 0) {
                                    stats.io_wait_time_ms += count * 10.0;  // Estimate 10ms per migration
                                }
                            }
                            close(core_fds.io_wait_fds[i]);
                        }
                        
                        // Read CPU cycles counter
                        if (core_fds.cpu_cycles_fds[i] != -1) {
                            struct read_format rf;
                            if (read(core_fds.cpu_cycles_fds[i], &rf, sizeof(rf)) == sizeof(rf)) {
                                ioctl(core_fds.cpu_cycles_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                                
                                // Scaling factor calculation
                                if (rf.time_enabled > 0) {
                                    double scaling_factor = (double)rf.time_running / rf.time_enabled;
                                    
                                    if (scaling_factor < 1.0 && i < stats.core_cpu_times.size()) {
                                        stats.core_cpu_times[i] = stats.core_cpu_times[i] * scaling_factor;
                                    }
                                }
                            }
                            close(core_fds.cpu_cycles_fds[i]);
                        }
                        
                        // Read CPU instructions counter
                        if (core_fds.cpu_instructions_fds[i] != -1) {
                            struct read_format rf;
                            if (read(core_fds.cpu_instructions_fds[i], &rf, sizeof(rf)) == sizeof(rf)) {
                                ioctl(core_fds.cpu_instructions_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                            }
                            close(core_fds.cpu_instructions_fds[i]);
                        }
                        
                        // Read CPU reference cycles counter
                        if (core_fds.cpu_ref_cycles_fds[i] != -1) {
                            struct read_format rf;
                            if (read(core_fds.cpu_ref_cycles_fds[i], &rf, sizeof(rf)) == sizeof(rf)) {
                                ioctl(core_fds.cpu_ref_cycles_fds[i], PERF_EVENT_IOC_DISABLE, 0);
                            }
                            close(core_fds.cpu_ref_cycles_fds[i]);
                        }
                    }
                    
                    // Clean up
                    phase_core_fds.erase(core_fds_it);
                }
                
                if (core_timespec_it != phase_core_timespec.end()) {
                    // Clean up
                    phase_core_timespec.erase(core_timespec_it);
                }
                
                return stats;
            }
        };

    // Class to collect and report performance metrics
    class PerformanceMetrics {
        public:
            void RecordStats(const std::string& phase, const PerfStats& stats) {
                if (phase_stats.find(phase) == phase_stats.end()) {
                    phase_stats[phase] = std::vector<PerfStats>();
                }
                phase_stats[phase].push_back(stats);
            }
        
            void PrintStats() const {
                for (const auto& [phase, stats_vec] : phase_stats) {
                    if (stats_vec.empty()) continue;
        
                    std::cout << "\n=== Performance Statistics for Phase: " << phase << " ===\n";
                    
                    if (stats_vec.size() == 1) {
                        const auto& stats = stats_vec[0];
                        PrintSinglePhaseStat(stats);
                    } else {
                        double avg_wall_time = 0;
                        double avg_user_time = 0;
                        double avg_system_time = 0;
                        double avg_cpu_time = 0;
                        double avg_io_wait_time = 0;
                        double avg_io_bytes_read = 0;
                        double avg_io_bytes_written = 0;
                        
                        for (const auto& stats : stats_vec) {
                            avg_wall_time += stats.wall_time_ms;
                            avg_user_time += stats.user_time_sec;
                            avg_system_time += stats.system_time_sec;
                            avg_cpu_time += stats.cpu_time_sec;
                            avg_io_wait_time += stats.io_wait_time_ms;
                            avg_io_bytes_read += stats.io_bytes_read;
                            avg_io_bytes_written += stats.io_bytes_written;
                        }
                        
                        size_t count = stats_vec.size();
                        avg_wall_time /= count;
                        avg_user_time /= count;
                        avg_system_time /= count;
                        avg_cpu_time /= count;
                        avg_io_wait_time /= count;
                        avg_io_bytes_read /= count;
                        avg_io_bytes_written /= count;
        
                        std::cout << "Number of measurements: " << count << "\n"
                                << "Average wall clock time: " << avg_wall_time << " ms\n"
                                << "Average user time: " << avg_user_time << " sec\n"
                                << "Average system time: " << avg_system_time << " sec\n"
                                << "Average CPU time (user+system): " << avg_cpu_time << " sec\n"
                                << "Average I/O wait time: " << avg_io_wait_time << " ms\n"
                                << "Average I/O bytes read: " << avg_io_bytes_read / (1024.0 * 1024.0) << " MB\n"
                                << "Average I/O bytes written: " << avg_io_bytes_written / (1024.0 * 1024.0) << " MB\n"
                                << "CPU utilization: " << (avg_cpu_time * 1000 * 100) / avg_wall_time << "%\n";
                        
                        // Per-step details (if there aren't too many)
                        if (stats_vec.size() <= 10) {
                            std::cout << "\nPer-step details:\n";
                            for (size_t i = 0; i < stats_vec.size(); i++) {
                                std::cout << "Step " << i << ":\n";
                                PrintSinglePhaseStat(stats_vec[i], "  ");
                            }
                        }
                    }
                }
            }
        
        private:
            void PrintSinglePhaseStat(const PerfStats& stats, const std::string& prefix = "") const {
                std::cout << prefix << "Wall clock time: " << stats.wall_time_ms << " ms\n"
                        << prefix << "User time: " << stats.user_time_sec << " sec\n"
                        << prefix << "System time: " << stats.system_time_sec << " sec\n"
                        << prefix << "Total CPU time (user+system): " << stats.cpu_time_sec << " sec\n"
                        << prefix << "Process CPU time (timespec): " << stats.process_cpu_time_sec << " sec\n"
                        << prefix << "I/O wait time: " << stats.io_wait_time_ms << " ms\n"
                        << prefix << "I/O bytes read: " << stats.io_bytes_read / (1024.0 * 1024.0) << " MB\n"
                        << prefix << "I/O bytes written: " << stats.io_bytes_written / (1024.0 * 1024.0) << " MB\n"
                        << prefix << "CPU utilization: " << (stats.cpu_time_sec * 1000 * 100) / stats.wall_time_ms << "%\n";
                        
                // Print per-core stats if available
                if (!stats.core_user_times.empty()) {
                    std::cout << prefix << "Per-core statistics:\n";
                    for (size_t i = 0; i < stats.core_user_times.size(); i++) {
                        std::cout << prefix << "  Core " << i << ": "
                                << "User=" << stats.core_user_times[i] << "s, "
                                << "System=" << stats.core_system_times[i] << "s, "
                                << "Total=" << (stats.core_user_times[i] + stats.core_system_times[i]) << "s\n";
                    }
                }
            }
        
            std::unordered_map<std::string, std::vector<PerfStats>> phase_stats;
        };

    // Helper to calculate wall-to-cpu time ratio
    inline double GetParallelEfficiency(const PerfStats& stats) {
        if (stats.wall_time_ms <= 0) return 0.0;
        return (stats.cpu_time_sec * 1000) / stats.wall_time_ms;
    }

    // --------------------------------------------------------------------------
    // A scoped timer that prints the elapsed time when going out of scope
    // --------------------------------------------------------------------------
    class ScopeTimer
    {
    public:
        explicit ScopeTimer(const std::string &name)
            : name_(name),
              start_(std::chrono::high_resolution_clock::now()) {}

        ~ScopeTimer()
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
            std::cout << "\n[INFO] " << name_ << " took " << duration_ms << " ms\n";
        }

    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };

    // --------------------------------------------------------------------------
    // Class for measuring decoding metrics (time to first token, average times, etc.)
    // --------------------------------------------------------------------------
    class DecodingMetrics
    {
    public:
        // Called before decoding loop starts
        void StartDecoding()
        {
            decode_start_ = std::chrono::high_resolution_clock::now();
        }

        // Record times for each token
        //   - token_start: time point before inference/sampling starts for a token
        //   - inference_time_ms: how many ms were spent in model inference
        //   - sampling_time_ms : how many ms were spent in sampling the next token
        void RecordTimes(const std::chrono::high_resolution_clock::time_point &token_start,
                         double inference_time_ms, double sampling_time_ms)
        {
            auto token_end = std::chrono::high_resolution_clock::now();
            double decoding_time_ms =
                std::chrono::duration<double, std::milli>(token_end - token_start).count();

            // If this is the first token, record time to first token
            if (!first_token_recorded_)
            {
                first_token_recorded_ = true;
                time_to_first_token_ms_ =
                    std::chrono::duration<double, std::milli>(token_end - decode_start_).count();
            }

            // Track inference time
            total_inference_time_ms_ += inference_time_ms;
            // Track sampling time
            total_sampling_time_ms_ += sampling_time_ms;
            // Track total decoding time
            total_decoding_time_ms_ += decoding_time_ms;

            // Track total tokens
            ++token_count_;
        }

        // Print out final decoding metrics
        void PrintMetrics() const
        {
            double avg_inference_time_ms = 0.0;
            double avg_sampling_time_ms = 0.0;
            double avg_decoding_time_ms = 0.0;
            double avg_inference_speed = 0.0;
            double avg_sampling_speed = 0.0;
            double avg_decoding_speed = 0.0;

            if (token_count_ > 0)
            {
                avg_inference_time_ms = total_inference_time_ms_ / token_count_;
                avg_sampling_time_ms = total_sampling_time_ms_ / token_count_;
                avg_decoding_time_ms = (total_sampling_time_ms_ + total_inference_time_ms_) / token_count_;

                avg_inference_speed = token_count_ / (total_inference_time_ms_ / 1000);
                avg_sampling_speed = token_count_ / (total_sampling_time_ms_ / 1000);
                avg_decoding_speed = token_count_ / (total_decoding_time_ms_ / 1000);
            }

            std::cout << "\n\n================================\n";
            std::cout << "[INFO] Decoding stage completed\n";
            std::cout << "[METRICS] Total Number of Generated Tokens : " << token_count_ << " tokens\n\n";

            std::cout << "[METRICS] Total Inference Latency          : " << total_inference_time_ms_ << " ms\n";
            std::cout << "[METRICS] Total Sampling Latency           : " << total_sampling_time_ms_ << " ms\n";
            std::cout << "[METRICS] Total Decoding Latency           : " << total_decoding_time_ms_ << " ms\n\n";

            std::cout << "[METRICS] Time To First Token              : " << time_to_first_token_ms_ << " ms\n";
            std::cout << "[METRICS] Average Inference Latency        : " << avg_inference_time_ms << " ms/tokens"
                      << "(" << avg_inference_speed << " token/s )\n";
            std::cout << "[METRICS] Average Sampling Latency         : " << avg_sampling_time_ms << " ms/tokens"
                      << "(" << avg_sampling_speed << " token/s )\n";
            std::cout << "[METRICS] Average Decoding Latency         : " << avg_decoding_time_ms << " ms/tokens"
                      << "(" << avg_decoding_speed << " token/s )\n";
        }

    private:
        // Decode start time
        std::chrono::high_resolution_clock::time_point decode_start_;

        // Time to first token
        double time_to_first_token_ms_ = 0.0;
        bool first_token_recorded_ = false;

        // Accumulators
        double total_inference_time_ms_ = 0.0;
        double total_sampling_time_ms_ = 0.0;
        double total_decoding_time_ms_ = 0.0;
        int token_count_ = 0;
    };

    // --------------------------------------------------------------------------
    // A class that provides various sampling methods (Greedy, Top-K, Top-P, etc.)
    // --------------------------------------------------------------------------
    class Sampler
    {
    public:
        // ------------------------
        // Greedy Sampler
        // ------------------------
        static int GreedySampler(const TfLiteTensor *logits)
        {
            float max_value = -std::numeric_limits<float>::infinity();
            int max_index = 0;
            int vocab_size = logits->dims->data[2];

            for (int i = 0; i < vocab_size; ++i)
            {
                if (logits->data.f[i] > max_value)
                {
                    max_value = logits->data.f[i];
                    max_index = i;
                }
            }
            return max_index;
        }

        // ------------------------
        // Top-K Sampler
        // ------------------------
        static int TopKSampler(const TfLiteTensor *logits, int k)
        {
            int vocab_size = logits->dims->data[2];
            std::vector<std::pair<float, int>> sorted_logits;
            sorted_logits.reserve(vocab_size);

            for (int i = 0; i < vocab_size; ++i)
            {
                sorted_logits.emplace_back(logits->data.f[i], i);
            }

            // Partial sort to get the top k elements
            if (k < vocab_size)
            {
                std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + k, sorted_logits.end(),
                                  std::greater<std::pair<float, int>>());
                sorted_logits.resize(k);
            }
            else
            {
                // If k >= vocab_size, no need to cut
                std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<std::pair<float, int>>());
            }

            // Compute normalized probabilities
            float sum_probs = 0.0f;
            for (auto &pair : sorted_logits)
            {
                sum_probs += std::exp(pair.first);
            }
            std::vector<float> probabilities;
            probabilities.reserve(sorted_logits.size());
            for (auto &pair : sorted_logits)
            {
                probabilities.push_back(std::exp(pair.first) / sum_probs);
            }

            // Multinomial sampling
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

            return sorted_logits[dist(gen)].second;
        }

        // ------------------------
        // Top-P (Nucleus) Sampler
        // ------------------------
        static int TopPSampler(const TfLiteTensor *logits, float p)
        {
            int vocab_size = logits->dims->data[2];
            std::vector<std::pair<float, int>> sorted_logits;
            sorted_logits.reserve(vocab_size);

            for (int i = 0; i < vocab_size; ++i)
            {
                sorted_logits.emplace_back(logits->data.f[i], i);
            }

            // Sort descending by logit value
            std::sort(sorted_logits.begin(), sorted_logits.end(),
                      std::greater<std::pair<float, int>>());

            // Apply softmax to get probabilities
            std::vector<float> probabilities(vocab_size);
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; ++i)
            {
                float val = std::exp(sorted_logits[i].first);
                probabilities[i] = val;
                sum_exp += val;
            }
            for (int i = 0; i < vocab_size; ++i)
            {
                probabilities[i] /= sum_exp;
            }

            // Find the cutoff index where cumulative probability exceeds p
            float cumulative_prob = 0.0f;
            int cutoff_index = vocab_size - 1;
            for (int i = 0; i < vocab_size; ++i)
            {
                cumulative_prob += probabilities[i];
                if (cumulative_prob > p)
                {
                    cutoff_index = i;
                    break;
                }
            }

            // Resize vectors to [0..cutoff_index]
            float new_sum = 0.0f;
            for (int i = 0; i <= cutoff_index; ++i)
            {
                new_sum += probabilities[i];
            }
            for (int i = 0; i <= cutoff_index; ++i)
            {
                probabilities[i] /= new_sum;
            }

            probabilities.resize(cutoff_index + 1);
            sorted_logits.resize(cutoff_index + 1);

            // Multinomial sampling
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
            return sorted_logits[dist(gen)].second;
        }

        // ------------------------
        // Temperature + Top-K + Top-P Sampler
        // ------------------------
        static int TemperatureTopKTopPSampler(const TfLiteTensor *logits,
                                              float temperature, int k, float p)
        {
            int vocab_size = logits->dims->data[2];
            std::vector<std::pair<float, int>> sorted_logits;
            sorted_logits.reserve(vocab_size);

            // 1) Apply Temperature
            std::vector<float> scaled_logits(vocab_size);
            for (int i = 0; i < vocab_size; ++i)
            {
                scaled_logits[i] = logits->data.f[i] / temperature;
            }

            // 2) Softmax over scaled logits
            float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; ++i)
            {
                scaled_logits[i] = std::exp(scaled_logits[i] - max_logit);
                sum_exp += scaled_logits[i];
            }
            for (int i = 0; i < vocab_size; ++i)
            {
                scaled_logits[i] /= sum_exp;
                // Keep index-value pairs for sorting
                sorted_logits.emplace_back(scaled_logits[i], i);
            }

            // 3) Sort descending by probability
            std::sort(sorted_logits.begin(), sorted_logits.end(),
                      std::greater<std::pair<float, int>>());

            // 4) Top-K filter
            int top_k = std::min(k, vocab_size);
            sorted_logits.resize(top_k);

            // 5) Top-P filter within top-k
            float cumulative_prob = 0.0f;
            int cutoff_index = top_k - 1;
            for (int i = 0; i < top_k; ++i)
            {
                cumulative_prob += sorted_logits[i].first;
                if (cumulative_prob > p)
                {
                    cutoff_index = i;
                    break;
                }
            }
            sorted_logits.resize(cutoff_index + 1);

            // 6) Renormalize final probabilities
            float new_sum = 0.0f;
            for (auto &pair : sorted_logits)
            {
                new_sum += pair.first;
            }

            std::vector<float> final_probs;
            final_probs.reserve(sorted_logits.size());
            for (auto &pair : sorted_logits)
            {
                final_probs.push_back(pair.first / new_sum);
            }

            // 7) Multinomial sampling
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
            return sorted_logits[dist(gen)].second;
        }
    };

    // --------------------------------------------------------------------------
    // Utility for applying XNNPACK weight caching
    // --------------------------------------------------------------------------
    void ApplyXNNPACKWeightCaching(tflite::Interpreter *interpreter)
    {
        auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
        std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
        delegate_options.weight_cache_file_path = weight_cache_path.c_str();
        delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);
        delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
        delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;

        MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                          tflite::Interpreter::TfLiteDelegatePtr(
                              TfLiteXNNPackDelegateCreate(&delegate_options),
                              [](TfLiteDelegate *delegate)
                              { TfLiteXNNPackDelegateDelete(delegate); })) == kTfLiteOk);
    }

    // --------------------------------------------------------------------------
    // Loads the TFLite model
    // --------------------------------------------------------------------------
    std::unique_ptr<tflite::FlatBufferModel> LoadModel()
    {
        std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(absl::GetFlag(FLAGS_tflite_model).c_str());
        MINIMAL_CHECK(model != nullptr);
        return model;
    }

    // --------------------------------------------------------------------------
    // Builds a TFLite interpreter from the model and applies XNNPACK if requested
    // --------------------------------------------------------------------------
    std::unique_ptr<tflite::Interpreter>
    BuildInterpreter(tflite::FlatBufferModel *model, int num_threads)
    {
        tflite::ops::builtin::BuiltinOpResolver resolver;
        // Register GenAI custom ops
        tflite::ops::custom::GenAIOpsRegisterer(&resolver);

        tflite::InterpreterBuilder builder(*model, resolver);
        MINIMAL_CHECK(builder.SetNumThreads(num_threads) == kTfLiteOk);

        std::unique_ptr<tflite::Interpreter> interpreter;
        builder(&interpreter);
        MINIMAL_CHECK(interpreter != nullptr);

        if (!absl::GetFlag(FLAGS_weight_cache_path).empty())
        {
            ApplyXNNPACKWeightCaching(interpreter.get());
        }
        return interpreter;
    }

    // --------------------------------------------------------------------------
    // Constructs KV cache input structures for decode, based on the decode signature
    // --------------------------------------------------------------------------
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>
    BuildKVCache(tflite::Interpreter *interpreter)
    {
        tflite::SignatureRunner *runner = interpreter->GetSignatureRunner("decode");
        if (runner == nullptr)
        {
            return {};
        }

        // Expect runner->input_size() = tokens, input_pos, plus 2*(num_layers)
        size_t num_layers = (runner->input_size() - 2) / 2;
        if (num_layers == 0)
        {
            return {};
        }

        std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
        for (int i = 0; i < num_layers; ++i)
        {
            std::string k_cache_name = "kv_cache_k_" + std::to_string(i);
            std::string v_cache_name = "kv_cache_v_" + std::to_string(i);

            TfLiteTensor *tensor = runner->input_tensor(k_cache_name.c_str());
            size_t count = tensor->bytes / sizeof(float);

            kv_cache.emplace(k_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
            kv_cache.emplace(v_cache_name,
                             std::vector<float, AlignedAllocator<float>>(count, 0.0f));
        }
        return kv_cache;
    }

    // --------------------------------------------------------------------------
    // Sets custom memory allocations for the KV cache on the given runner
    // --------------------------------------------------------------------------
    void PrepareRunner(tflite::SignatureRunner *runner,
                       std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache)
    {
        for (auto &[name, cache] : kv_cache)
        {
            TfLiteCustomAllocation allocation{
                .data = static_cast<void *>(cache.data()),
                .bytes = cache.size() * sizeof(float)};

            MINIMAL_CHECK(runner->SetCustomAllocationForInputTensor(name.c_str(), allocation) == kTfLiteOk);
            MINIMAL_CHECK(runner->SetCustomAllocationForOutputTensor(name.c_str(), allocation) == kTfLiteOk);
        }
        MINIMAL_CHECK(runner->AllocateTensors() == kTfLiteOk);
    }

    // --------------------------------------------------------------------------
    // Finds the appropriate "prefill" runner for the given number of tokens.
    // If LoRA is used, it defers to LoRA's specialized runner selection.
    // --------------------------------------------------------------------------
    tflite::SignatureRunner *GetPrefillRunner(
        tflite::Interpreter *interpreter,
        std::size_t num_input_tokens,
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
        const ai_edge_torch::examples::LoRA *lora)
    {
        tflite::SignatureRunner *runner = nullptr;
        int best_seq_size = -1;
        int delta = std::numeric_limits<int>::max();

        for (const std::string *key : interpreter->signature_keys())
        {
            if (!absl::StrContains(*key, "prefill") || absl::StrContains(*key, "lora"))
            {
                continue;
            }
            TfLiteTensor *input_pos =
                interpreter->GetSignatureRunner(key->c_str())->input_tensor("input_pos");
            int seq_size = input_pos->dims->data[0];

            // Choose the runner where seq_size >= num_input_tokens and
            // (seq_size - num_input_tokens) is minimized
            if (num_input_tokens <= static_cast<size_t>(seq_size) &&
                seq_size - static_cast<int>(num_input_tokens) < delta)
            {
                if (lora == nullptr)
                {
                    runner = interpreter->GetSignatureRunner(key->c_str());
                }
                best_seq_size = seq_size;
                delta = seq_size - static_cast<int>(num_input_tokens);
            }
        }

        // If LoRA is enabled, use the LoRA-specific prefill runner
        if (lora != nullptr)
        {
            runner = lora->GetPrefillRunner(interpreter, best_seq_size);
        }
        MINIMAL_CHECK(runner != nullptr);

        // Prepare KV memory allocations
        PrepareRunner(runner, kv_cache);
        return runner;
    }

    // --------------------------------------------------------------------------
    // Retrieves the decode runner (LoRA-based if needed) and prepares it
    // --------------------------------------------------------------------------
    tflite::SignatureRunner *GetDecodeRunner(
        tflite::Interpreter *interpreter,
        std::map<std::string, std::vector<float, AlignedAllocator<float>>> &kv_cache,
        ai_edge_torch::examples::LoRA *lora)
    {
        tflite::SignatureRunner *runner =
            (lora == nullptr)
                ? interpreter->GetSignatureRunner("decode")
                : lora->GetDecodeRunner(interpreter);
        MINIMAL_CHECK(runner != nullptr);

        PrepareRunner(runner, kv_cache);
        return runner;
    }

    // --------------------------------------------------------------------------
    // Loads the SentencePiece model from file
    // --------------------------------------------------------------------------
    std::unique_ptr<sentencepiece::SentencePieceProcessor> LoadSentencePieceProcessor()
    {
        std::ifstream input(absl::GetFlag(FLAGS_sentencepiece_model), std::ios::binary);
        std::string serialized_proto((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());

        auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
        MINIMAL_CHECK(processor->LoadFromSerializedProto(serialized_proto).ok());
        return processor;
    }

    // RUSAGE
    struct RUsageRecord {
        rusage start;
        rusage end;
    };

    void PrintRUsage(rusage usage_start, rusage usage_end, const std::string phase_name) {
        double user_time_start = toSeconds(usage_start.ru_utime);
        double user_time_end   = toSeconds(usage_end.ru_utime);
        double sys_time_start  = toSeconds(usage_start.ru_stime);
        double sys_time_end    = toSeconds(usage_end.ru_stime);
        double cpu_time_sec = (user_time_end - user_time_start)
                                + (sys_time_end - sys_time_start);
	    double user_time_sec = (user_time_end - user_time_start);
	    double sys_time_sec = (sys_time_end - sys_time_start);
        std::cout << phase_name << " took \n- "
            << cpu_time_sec << " [sec] CPU time\n- "
	       << user_time_sec << " [sec] User time\n- "
	       << sys_time_sec << " [sec] System time" << std::endl;
    }

    void PrintRUsageRecords(const std::vector<RUsageRecord>& records) {
        for (size_t i = 0; i < records.size(); i++) {
            PrintRUsage(records[i].start, records[i].end, "Decode " + std::to_string(i));
        }
    }

    void uploadTensorsForAllSubgraphs(tflite::Interpreter* interpreter) {
        if (!interpreter) {
            std::cerr << "Invalid interpreter pointer\n";
            return;
        }
    
        // Get the number of subgraphs
        size_t num_subgraphs = interpreter->subgraphs_size();
        std::cout << "Processing " << num_subgraphs << " subgraphs\n";
    
        // Keep track of total tensors touched across all subgraphs
        size_t total_tensors_touched = 0;
    
        // Process each subgraph
        for (size_t subgraph_idx = 0; subgraph_idx < num_subgraphs; ++subgraph_idx) {
            const tflite::Subgraph& subgraph = (subgraph_idx == 0) ?
                interpreter->primary_subgraph() :
                *interpreter->subgraph(subgraph_idx);
    
            const std::vector<int>& execution_plan = subgraph.execution_plan();
            std::unordered_set<int> seen_tensors;
    
            std::cout << "Touching tensors for subgraph " << subgraph_idx << "\n";
    
            // Process each node in the execution plan
            for (int node_idx : execution_plan) {
                const auto* node_and_reg = subgraph.node_and_registration(node_idx);
                const TfLiteNode* node = &node_and_reg->first;
    
                // Helper lambda to process tensors
                auto processTensors = [&](const TfLiteIntArray* tensor_array) {
                    if (!tensor_array) return;
                    for (int i = 0; i < tensor_array->size; ++i) {
                        int tensor_idx = tensor_array->data[i];
                        if (tensor_idx < 0 || seen_tensors.count(tensor_idx)) continue;
    
                        TfLiteTensor* tensor = interpreter->tensor(tensor_idx);
                        if (tensor && tensor->data.raw) {
                            size_t size = tensor->bytes;
                            for (size_t offset = 0; offset < size; offset += 4096) {
                                volatile char dummy = *reinterpret_cast<char*>(tensor->data.raw + std::min(offset, size - 1));
                                (void)dummy;
                            }
                        }
                        seen_tensors.insert(tensor_idx);
                    }
                };
    
                // Process all tensor types
                processTensors(node->inputs);
                processTensors(node->outputs);
                processTensors(node->temporaries);
            }
    
            total_tensors_touched += seen_tensors.size();
            std::cout << "Touched " << seen_tensors.size() << " tensors in subgraph " << subgraph_idx << "\n";
        }
    
        std::cout << "Total tensors touched across all subgraphs: " << total_tensors_touched << "\n";
    }

} // end anonymous namespace

// =======================================================================
// main() entry
// =======================================================================
int main(int argc, char *argv[])
{
    // 0. Parse flags
    absl::ParseCommandLine(argc, argv);
    std::cout << "[INFO] Preparing Required Components\n";

    // Global variables
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor;
    std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
    std::unique_ptr<ai_edge_torch::examples::LoRA> lora = nullptr;
    std::vector<int> prompt_tokens;
    std::string prompt, start_token, stop_token;
    int stop_token_id = -1;

    // 0-1. Perf monitor initialziation
    // Check which cores we're actually running on
    std::vector<int> active_cores = detect_active_cores();
    std::cout << "Process is running on cores: ";
    for (int core : active_cores) {
        std::cout << core << " ";
    }
    std::cout << std::endl;

    // Just monitor the cores we're allowed to run on (should be only core 0 with taskset)
    PerformanceMonitor perf_monitor(active_cores);
    PerformanceMetrics metrics;
    PerfStats stats;

    // Add some code to get I/O stats from /proc for better I/O measurement
    double proc_io_wait_start = 0.0;

    // 0-2. Variable for CPU time only
    rusage usage_start, usage_end;

    // 1. Load Model
    {
        ScopeTimer timer("Model Loading");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Model_Loading");
        model = LoadModel();
        stats = perf_monitor.end_phase("Model_Loading");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Model Loading");
    metrics.RecordStats("Model_Loading", stats);

    // 2. Build Interpreter
    {
        ScopeTimer timer("Interpreter Building");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Build_Interperter");
        interpreter = BuildInterpreter(model.get(), absl::GetFlag(FLAGS_num_threads));
        stats = perf_monitor.end_phase("Build_Interperter");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Interpreter Building");
    metrics.RecordStats("Build_Interpreter", stats);

    // Tensor upload before prefill
    {
        ScopeTimer timer("Tensor Uploading");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Upload_Tensor");

        // Uploading Here
        // uploadTensorsForAllSubgraphs(interpreter.get());

        stats = perf_monitor.end_phase("Upload_Tensor");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Tensor Uploading");
    metrics.RecordStats("Upload_Tensor", stats);


    // 3. Load SentencePiece
    {
        ScopeTimer timer("SentencePiece Loading");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Load_SentencePiece");
        sp_processor = LoadSentencePieceProcessor();
        stats = perf_monitor.end_phase("Load_SentencePiece");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Sentence Piece Loading");
    metrics.RecordStats("Load_SentencePiece", stats);

    // 4. Build KV Cache
    {
        ScopeTimer timer("KV Cache Building");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Build_KVCache");
        kv_cache = BuildKVCache(interpreter.get());
        MINIMAL_CHECK(!kv_cache.empty());
        stats = perf_monitor.end_phase("Build_KVCache");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "KV Cache Building");
    metrics.RecordStats("Build_KVCache", stats);

    // 5. Optionally load LoRA
    // {
    //     ScopeTimer timer("LoRA Loading");
    //     if (!absl::GetFlag(FLAGS_lora_path).empty())
    //     {
    //         lora = ai_edge_torch::examples::LoRA::FromFile(absl::GetFlag(FLAGS_lora_path));
    //         MINIMAL_CHECK(lora != nullptr);
    //     }
    // }

    // 6. Prepare Input Prompt
    {
        ScopeTimer timer("Input Prompt Preparation");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Prepare_Prompt");
        prompt = absl::GetFlag(FLAGS_prompt);
        MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

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
        stats = perf_monitor.end_phase("Prepare_Prompt");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Input Prompt Preparation");
    metrics.RecordStats("Prepare_Prompt", stats);

    // 7. Prepare Signature Runners
    tflite::SignatureRunner *prefill_runner = nullptr;
    tflite::SignatureRunner *decode_runner = nullptr;
    {
        ScopeTimer timer("Signature Runners Preparation");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Prepare_Runners");
        std::size_t effective_prefill_token_size =
            (prompt_tokens.size() > 0) ? (prompt_tokens.size() - 1) : 0;
            // std::cout << "HELLO";
        prefill_runner = GetPrefillRunner(
            interpreter.get(), effective_prefill_token_size, kv_cache, nullptr);
        // std::cout << "HELLO2";
        MINIMAL_CHECK(prefill_runner != nullptr);
        // std::cout << "HELLO1";

        decode_runner = GetDecodeRunner(interpreter.get(), kv_cache, nullptr);
        // std::cout << "HELLO4";
        MINIMAL_CHECK(decode_runner != nullptr);
        // std::cout << "HELLO3";
        
        stats = perf_monitor.end_phase("Prepare_Runners");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Signature Runner Preparation");
    metrics.RecordStats("Prepare_Runners", stats);
    
    // 8. Access Tensors
    TfLiteTensor *prefill_input = prefill_runner->input_tensor("tokens");
    TfLiteTensor *prefill_input_pos = prefill_runner->input_tensor("input_pos");
    TfLiteTensor *decode_input = decode_runner->input_tensor("tokens");
    TfLiteTensor *decode_input_pos = decode_runner->input_tensor("input_pos");
    TfLiteTensor *kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");

    int max_seq_size = prefill_input->dims->data[1];
    int kv_cache_max_size = kv_cache_k_0->dims->data[1];
    
    // 9. Prefill Stage
    {
        ScopeTimer timer("Prefill Stage");
        getrusage(RUSAGE_SELF, &usage_start);
        perf_monitor.start_phase("Prefill");
        int prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);
        std::cout << prefill_seq_size;
        // Zero out the input tensors
        std::memset(prefill_input->data.i32, 0, prefill_input->bytes);
        std::memset(prefill_input_pos->data.i32, 0, prefill_input_pos->bytes);
        
        // Prefill uses all but the last token from the prompt
        for (int i = 0; i < prefill_seq_size - 1; ++i)
        {
            prefill_input->data.i32[i] = prompt_tokens[i];
            prefill_input_pos->data.i32[i] = i;
        }

        
        // Execute the prefill runner
        MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk);
        stats = perf_monitor.end_phase("Prefill");
        getrusage(RUSAGE_SELF, &usage_end);
    }
    PrintRUsage(usage_start, usage_end, "Prefill Stage");
    metrics.RecordStats("Prefill", stats);

    // 10. Decoding Stage with separate metrics for inference and sampling
    std::cout << "\nPrompt:\n"
              << prompt << "\n\nOutput Text:\n";

    // Metrics object
    DecodingMetrics decoding_metrics;
    decoding_metrics.StartDecoding();
    std::vector<PerfStats> decode_stats_vec;
    std::vector<RUsageRecord> rusageRecords;
    struct RUsageRecord decode_record;
    //rusage decode_start, decode_end;
    {
        // ScopeTimer timer("Decoding Stage");

        // Determine how many tokens to generate
        int max_decode_steps = (absl::GetFlag(FLAGS_max_decode_steps) == -1)
                                   ? kv_cache_max_size
                                   : absl::GetFlag(FLAGS_max_decode_steps);

        int prefill_seq_size = std::min<int>(prompt_tokens.size(), max_seq_size);
        int decode_steps = std::min<int>(max_decode_steps, kv_cache_max_size - prefill_seq_size);
        MINIMAL_CHECK(decode_steps > 0);

        int next_token = prompt_tokens[prefill_seq_size - 1];
        int next_position = prefill_seq_size - 1;

        // Decoding loop
        for (int i = 0; i < decode_steps; ++i)
        {
            // Start time for this token
            auto token_start = std::chrono::high_resolution_clock::now();
            getrusage(RUSAGE_SELF, &decode_record.start);
            perf_monitor.start_phase("Decode_Token_" + std::to_string(i));

            // -----------------------
            // 1) Model Inference
            // -----------------------
            auto inference_start = std::chrono::high_resolution_clock::now();

            decode_input->data.i32[0] = next_token;
            decode_input_pos->data.i32[0] = next_position;
            MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);

            auto inference_end = std::chrono::high_resolution_clock::now();
            double inference_time_ms =
                std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

            // -----------------------
            // 2) Token Sampling
            // -----------------------
            auto sampling_start = std::chrono::high_resolution_clock::now();
            next_token = Sampler::TemperatureTopKTopPSampler(
                decode_runner->output_tensor("logits"), 0.9f, 85, 0.9f);
            auto sampling_end = std::chrono::high_resolution_clock::now();
            double sampling_time_ms =
                std::chrono::duration<double, std::milli>(sampling_end - sampling_start).count();

            next_position++;

            // Check stop token
            if (next_token == stop_token_id)
            {
                break;
            }

            // Decode the single token to text
            std::vector<int> single_token_vec = {next_token};
            std::string single_decoded_text;
            MINIMAL_CHECK(sp_processor->Decode(single_token_vec, &single_decoded_text).ok());
            std::cout << single_decoded_text << std::flush;

            // End perf recording
            PerfStats token_stats = perf_monitor.end_phase("Decode_Token_" + std::to_string(i));
            decode_stats_vec.push_back(token_stats);
            metrics.RecordStats("Decode_Token", token_stats);
            // Record metrics for this token
            decoding_metrics.RecordTimes(token_start, inference_time_ms, sampling_time_ms);
            getrusage(RUSAGE_SELF, &decode_record.end);
            rusageRecords.push_back(decode_record);
        }
    }

    // 11. Print decoding metrics (inference vs. sampling)
    decoding_metrics.PrintMetrics();
    // 12. Print Perf results
    metrics.PrintStats();
    // 13. Print RUsage results
    PrintRUsageRecords(rusageRecords);

    return 0;
}
