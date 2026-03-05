#!/usr/bin/env bash
# run.sh - Refactored version with better readability
#
# The JSON file should contain hyperparameters for each prompt:
# {
#   "tokens": 512,
#   "prompt": "Your prompt text here",
#   "temperature": 0.8,
#   "top_k": 50,
#   "top_p": 0.95,
#   "repetition_penalty": 1.1,
#   "enable_repetition_penalty": true
# }

set -euo pipefail

# =========================================================================== #
# 0. Load Environment and Helpers                                             #
# =========================================================================== #
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found. Please run from project root directory." >&2
    exit 1
fi

source .env
source ./scripts/utils.sh

# =========================================================================== #
# 1. Configuration                                                            #
# =========================================================================== #
banner "Script Configuration"

# --- Model and Binary Settings ---
# You can switch models by uncommenting the desired lines.
# MODEL_DIR="${MODEL_PATH}/Llama3.2-1B"
# MODEL_NAME="llama3.2_q8_ekv1280"

MODEL_DIR="${MODEL_PATH}/Llama3.2-3B"
MODEL_NAME="llama3.2_q8_ekv1024"

# MODEL_DIR="${MODEL_PATH}/Gemma3-1B"
# MODEL_NAME="gemma3_q4_ekv2048"
# MODEL_NAME="gemma3_q8_ekv2048"

# MODEL_DIR="${MODEL_PATH}/Gemma2-2B"
# MODEL_NAME="gemma2_q8_ekv1024"

# MODEL_DIR="${MODEL_PATH}/Qwen2.5-1.5B"
# MODEL_NAME="qwen2.5-1.5b_q8_ekv1280"
# MODEL_NAME="qwen2.5-1.5b_q8_ekv4096"

# MODEL_DIR="${MODEL_PATH}/Qwen2.5-3B"
# MODEL_NAME="qwen2.5-3b_q8_ekv1280"

# MODEL_DIR="${MODEL_PATH}/SmolLM-135M"
# MODEL_NAME="smollm_q8_ekv1280"

BIN="bin/text_generator_main"

# --- Execution Settings ---
TARGET="cpu"           # Default target: cpu | gpu
LOG_ENABLED=false      # Default logging: false
CORE_LIST="all"          # Default core list for taskset
NUM_THREADS=1          # Default number of threads
PROMPT_FILE="${PROMPT_PATH}/sample_prompt_8_1.json" # Default prompt file
MAX_TOK_LEN=16         # Default max tokens to generate
NUM_REPEATS=1          # Default number of iterations
MEMORY_LIMITS=()       # Array of memory limits for cgroup testing
ENABLE_CGROUP=false    # Default cgroup enable state
BPF_PHASE_LOGGING=false # Default BPF phase logging

# --- Logging Settings ---
# Base directory for logs. The final path will be e.g. <LOG_DIR_BASE>/<model_target_mem>
LOG_DIR_BASE="benchmark/llm_infer_results"

log "--- Using the following configuration ---"
log "Model            : ${MODEL_DIR}/${MODEL_NAME}.tflite"
log "Binary           : ${BIN}"
log "Default prompt   : ${PROMPT_FILE}"
log "Log base dir     : ${LOG_DIR_BASE}"
log "-------------------------------------------"

# =========================================================================== #
# 2. Validate Prerequisites                                                   #
# =========================================================================== #
[[ -x "$BIN" ]] || error "Binary not found: $BIN"
[[ -f "$PROMPT_FILE" ]] || error "Prompt file not found: $PROMPT_FILE"
[[ "$PROMPT_FILE" =~ \.(json)$ ]] || error "Prompt file must be .json format"

# =========================================================================== #
# 3. Usage and CLI Parsing                                                    #
# =========================================================================== #
usage() {
    banner "Usage Information"
    cat <<'EOF'
    Usage: run.sh [OPTIONS]

    This script runs LLM inference using either a GPU or CPU binary,
    based on the specified model and prompt input from JSON files.
    Supports memory-constrained benchmarking with cgroups.

    Options:
    -g, --target <gpu|cpu>   Select binary to run (default: cpu)
    -l, --log                Enable logging to ./result_* directory (default: off)
    -c, --core <CORES>       Specify CPU core(s) to bind the process using 'taskset'
                             For example: "0-3" or "0,2". Default is to use all available cores.
    -t, --threads <N>        Set the number of threads to use for inference (default: 1)
    -f, --file <PATH>        Path to the input prompt file (JSON format)
                             Default: ./${PROMPT_PATH}/sample_prompt_8_1.json
    --tl, --max_tokens <N>   Set the maximum number of tokens to generate (default: 16)
    -r, --repeat <N>         Repeat all prompts N times (default: 1)
    -m, --memory <LIMIT>     Enable memory-constrained benchmarking with specified limit
                             Can be used multiple times: -m 512M -m 1G -m 2G
                             Supports suffixes: K, M, G (e.g., 512M, 1G, 2G)
    -h, --help               Show this help message and exit

    JSON Format:
        Single prompt: {
          "tokens": 512,
          "prompt": "Hello, how are you?",
          "temperature": 0.8,
          "top_k": 50,
          "top_p": 0.95,
          "repetition_penalty": 1.1,
          "enable_repetition_penalty": true
        }
        
        Multiple prompts: [
          {"tokens": 512, "prompt": "...", "temperature": 0.8, ...},
          {"tokens": 256, "prompt": "...", "temperature": 0.7, ...}
        ]

    Examples:
        ./run.sh --target gpu --log --core 0-3 --threads 4
        ./run.sh -g cpu -f prompts.json --tl 32
        ./run.sh --target cpu --file my_prompts.json
        ./run.sh --memory 512M --memory 1G --repeat 3 --log
EOF
    exit 0
}

parse_command_line(){
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -g | --target)
        TARGET="$2"
        shift 2
        ;;
    -l | --log)
        LOG_ENABLED=true
        shift
        ;;
    -c | --core)
        CORE_LIST="$2"
        shift 2
        ;;
    -t | --threads)
        NUM_THREADS="$2"
        shift 2
        ;;
    -f | --file)
        PROMPT_FILE="$2"
        shift 2
        ;;
    --tl | --max_tokens)
        MAX_TOK_LEN="$2"
        shift 2
        ;;
    -r | --repeat)
        NUM_REPEATS="$2"
        shift 2
        ;;
    -m | --memory)
        MEMORY_LIMITS+=("$2")
        ENABLE_CGROUP=true
        shift 2
        ;;
    -h | --help) 
        usage 
        ;;
    *)
        error "Unknown option: $1"
        usage
        ;;
    esac
done

# Validate target
[[ "${TARGET}" =~ ^(gpu|cpu)$ ]] || error "Invalid target argument: ${TARGET}"

# Validate memory limits if specified
if [[ ${#MEMORY_LIMITS[@]} -gt 0 ]]; then
    log "Memory-constrained benchmarking enabled with limits: ${MEMORY_LIMITS[*]}"
fi

# Set default memory limits if cgroup is enabled but no limits specified
if [[ "$ENABLE_CGROUP" == true && ${#MEMORY_LIMITS[@]} -eq 0 ]]; then
    MEMORY_LIMITS=("2G")
    log "No memory limits specified, using default: ${MEMORY_LIMITS[*]}"
fi

}


# =========================================================================== #
# 5. Core Functions                                                           #
# =========================================================================== #


# Function to run a single prompt
run_with_single_prompt() {
    local TOKENS="$1"
    local PROMPT="$2"  
    local LOG_FILE="$3"
    local TEMPERATURE="$4"
    local TOP_K="$5"
    local TOP_P="$6"
    local REPETITION_PENALTY="$7"
    local ENABLE_REPETITION_PENALTY="$8"
    local MEMORY_LIMIT="${9:-}"
    local CSV_FILE="${LOG_FILE%.log}.csv"
    local BPF_PHASE_PYTHON_PATH="./tools/ebpf/profile_phase.py"
    local BPF_LOG_FILE_PATH="bpf_profile_phase_results_${NUM_THREADS}threads_prefill_${TOKENS}.log"

    banner "LLM inference start (${TARGET^^})"
    log "Model                           : ${MODEL_NAME}"
    log "Tokens requested                : ${TOKENS}"
    # log "Prompt           : ${PROMPT}"
    log "Cores                           : ${CORE_LIST}"
    log "Threads                         : ${NUM_THREADS}"
    log "Max tokens                      : ${MAX_TOK_LEN}"
    log "Temperature                     : ${TEMPERATURE}"
    log "Top-k                           : ${TOP_K}"
    log "Top-p                           : ${TOP_P}"
    log "Repetition penalty              : ${REPETITION_PENALTY}"
    log "Enable rep. penalty             : ${ENABLE_REPETITION_PENALTY}"
    [[ -n "$MEMORY_LIMIT" ]] && \
    log "Memory limit                    : ${MEMORY_LIMIT}"
    log "Target Processor                : ${TARGET^^}"
    if [[ "$LOG_ENABLED" == "true" ]]; then
        log "Log file                        : ${LOG_FILE}"
        log "Op-level profiling csv results  : ${CSV_FILE}"
    fi
    if [[ "$BPF_PHASE_LOGGING" == "true" ]]; then
        log "BPF phase profiling script      : ${BPF_PHASE_PYTHON_PATH}"
        log "BPF phase profiling log file    : ${BPF_LOG_FILE_PATH}"
    fi

    clear_caches

    # Build command as array (safe quoting)
    # Select io_engine (parallel_pread or io_uring) and following params
    # local io_engine="io_uring" 
    local io_ring_depth=16
    local io_subread_bytes=$((128*1024))

    local io_engine="parallel_pread" 
    local io_min_block_size=$((512*1024))
    local io_max_threads=4

    if [[ "$LOG_ENABLED" == "true" ]]; then
        local CMD=(
            "${BIN}"
            --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
            --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
            --max_decode_steps "${MAX_TOK_LEN}"
            --start_token "<bos>"
            --stop_token "<end_of_turn>"
            --num_threads "${NUM_THREADS}"
            --prompt "${PROMPT}"
            --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
            --temperature "${TEMPERATURE}"
            --top_k "${TOP_K}"
            --top_p "${TOP_P}"
            --repetition_penalty "${REPETITION_PENALTY}"
            --csv_profile_output_path "$CSV_FILE"
            --io_engine "${io_engine}"
            --io_ring_depth "${io_ring_depth}"
            --io_subread_bytes "${io_subread_bytes}"
            --io_min_block_size "${io_min_block_size}"
            --io_max_threads "${io_max_threads}"
        )
    else
        local CMD=(
            "${BIN}"
            --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
            --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
            --max_decode_steps "${MAX_TOK_LEN}"
            --start_token "<bos>"
            --stop_token "<end_of_turn>"
            --num_threads "${NUM_THREADS}"
            --prompt "${PROMPT}"
            --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
            --temperature "${TEMPERATURE}"
            --top_k "${TOP_K}"
            --top_p "${TOP_P}"
            --repetition_penalty "${REPETITION_PENALTY}"
            --io_engine "${io_engine}"
            --io_ring_depth "${io_ring_depth}"
            --io_subread_bytes "${io_subread_bytes}"
            --io_min_block_size "${io_min_block_size}"
            --io_max_threads "${io_max_threads}"
        )
    fi
    

    # Add repetition penalty flag if enabled
    if [[ "${ENABLE_REPETITION_PENALTY}" == "true" ]]; then
        CMD+=(--enable_repetition_penalty)
    fi
    
    # Add taskset if specified
    [[ "${CORE_LIST}" != "all" ]] && CMD=(taskset -c "${CORE_LIST}" "${CMD[@]}")

    if [[ -n "$BPF_PHASE_LOGGING" && "$BPF_PHASE_LOGGING" == "true" ]]; then
        # Start BPF profiler in the background with proper session management
        banner "--- BPF PROFILER START ---"
        run_bpf "$BPF_PHASE_PYTHON_PATH" "$BIN" "$BPF_LOG_FILE_PATH" &
        BG_PID=$! # Get background process PID (run_bpf)
        log "Started BPF profiler with PID: $BG_PID, Wait for initialization... 10 seconds"
        sleep 10 # Give some time for BPF to initialize
    fi

    banner "--- C++ Binary Execution START ---"
    banner "Command: ${CMD[*]}"

    # Activate cgroups if memory limit is specified
    if [[ -n "$MEMORY_LIMIT" ]]; then    
        # Use systemd-run for cgroup v2
        if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
            log "cgroup v2 detected: systemd-run"
            CMD=(
                systemd-run --quiet --scope
                -p MemoryMax="$MEMORY_LIMIT" -p MemoryHigh="$MEMORY_LIMIT" -p MemorySwapMax=0
                -- "${CMD[@]}"
            )
        else
            # Use cgexec for cgroup v1
            log "cgroup v1 detected: cgexec"
            
            local cg="/sys/fs/cgroup/memory/llmbench"
            if [[ ! -d "$cg" ]]; then
                log "Creating cgroup: $cg"
                sudo mkdir -p "$cg"
            fi
            echo 0 | sudo tee "$cg/memory.force_empty" >/dev/null || true
            echo "$(($(numfmt --from=iec "$MEMORY_LIMIT")))" | sudo tee "$cg/memory.limit_in_bytes" >/dev/null
            
            # Execute with cgroup
            CMD=(cgexec -g memory:llmbench "${CMD[@]}")
            
        fi
    fi

    # execute CMD
    sudo "${CMD[@]}"

    banner "--- C++ Binary Execution END ---"
    
    banner "--- BPF PROFILER END ---"
    
    if [[ "$BPF_PHASE_LOGGING" == "true" ]]; then
        cleanup_bpf "$BPF_PHASE_PYTHON_PATH" "$BG_PID"
        log "BPF profiling log saved to ${BPF_LOG_FILE_PATH}"
    fi

    # Post-process CSV file to fix any formatting issues
    if [[ "$LOG_ENABLED" == "true" ]]; then
        local tmp_fixed_csv_file="${CSV_FILE}.fixed.csv"

        echo "[INFO] Post-processing CSV file..."
        python3 ${ROOT_PATH}/tools/benchmark/fix_profile_report.py "$CSV_FILE" "$tmp_fixed_csv_file"
        if [ $? -eq 0 ]; then
                mv "$tmp_fixed_csv_file" "$CSV_FILE"
                echo "[INFO] CSV file overwritten with fixed version: $CSV_FILE"
            else
                echo "[ERROR] Failed to fix CSV file. Keeping original."
                rm -f "$tmp_fixed_csv_file"
            fi
        echo ""
        log "Log saved to ${LOG_FILE}"
        log "Op-level profiling csv results saved to ${CSV_FILE}"
    fi
}


# Function to process multiple prompts from array
process_prompts_and_run() {
    local parse_result="$1"
    local memory_limit="${2:-}"
    local iteration="${3:-1}"
    local log_dir="$4"

    local prompt_count
    prompt_count=$(echo "$parse_result" | head -n1 | awk '{print $2}')
    
    log "Found $prompt_count prompts in JSON file. Processing all..."
    
    local prompt_index=0
    local current_tokens current_temperature current_top_k current_top_p
    local current_repetition_penalty current_enable_repetition_penalty
    local current_prompt="" in_prompt=false
    
    while IFS= read -r line; do
        if [[ "$line" =~ ^ITEM ]]; then
            # Parse: ITEM index tokens temperature top_k top_p repetition_penalty enable_repetition_penalty
            prompt_index=$(echo "$line" | awk '{print $2}')
            current_tokens=$(echo "$line" | awk '{print $3}')
            current_temperature=$(echo "$line" | awk '{print $4}')
            current_top_k=$(echo "$line" | awk '{print $5}')
            current_top_p=$(echo "$line" | awk '{print $6}')
            current_repetition_penalty=$(echo "$line" | awk '{print $7}')
            current_enable_repetition_penalty=$(echo "$line" | awk '{print $8}')
            current_prompt=""
            in_prompt=false
            
        elif [[ "$line" == "PROMPT_START" ]]; then
            in_prompt=true
            current_prompt=""
            
        elif [[ "$line" == "PROMPT_END" ]]; then
            in_prompt=false
            
            # Process this prompt
            local prompt_index_display=$((prompt_index + 1))
            log "Processing prompt $prompt_index_display/$prompt_count (${current_tokens} tokens)..." >&2
            
            local timestamp log_file
            timestamp=$(date +'%y%m%d_%H%M%S')
            if [[ $NUM_REPEATS -gt 1 ]]; then
                log_file="${log_dir}/run_${current_tokens}_${prompt_index_display}_${iteration}_${timestamp}.log"
            else
                log_file="${log_dir}/run_${current_tokens}_${prompt_index_display}_${timestamp}.log"
            fi

            log "Executing LLM Inference"
            execute_with_log "$log_file" run_with_single_prompt \
                "$current_tokens" "$current_prompt" "$log_file" \
                "$current_temperature" "$current_top_k" "$current_top_p" \
                "$current_repetition_penalty" "$current_enable_repetition_penalty" \
                "$memory_limit"
                
        elif [[ "$in_prompt" == "true" ]]; then
            if [[ -z "$current_prompt" ]]; then
                current_prompt="$line"
            else
                current_prompt="$current_prompt"$'\n'"$line"
            fi
        fi
    done <<< "$parse_result"
}


# Function to execute benchmarks with or without memory constraints
run_benchmarks() {
    local memory_limit="${1:-}"
    local current_log_dir
    
    # Setup log directory
    local model_id="${MODEL_NAME}"
    local target_id="${TARGET}"
    local mem_id
    
    if [[ -n "$memory_limit" ]]; then
        mem_id="_${memory_limit}"
        log "Memory Limit: $memory_limit"
    else
        mem_id=""
        log "Normal run (no memory limit)"
    fi
    current_log_dir="${LOG_DIR_BASE}/${model_id}_${target_id}${mem_id}"
    
    if [[ "$LOG_ENABLED" == "true" ]]; then
        if [[ -d "${current_log_dir}" ]]; then
            log "Log directory exists: ${current_log_dir} (appending)"
        else
            ensure_dir "${current_log_dir}"
            log "Created log directory: ${current_log_dir}"
        fi
    fi
    
    # Execute for each repeat
    for ((iter = 1; iter <= NUM_REPEATS; iter++)); do
        if [[ $NUM_REPEATS -gt 1 ]]; then
            banner "Iteration $iter / $NUM_REPEATS"
        fi
        
        # Parse JSON file
        local parse_result=$(parse_json_file "${PROMPT_FILE}")
        
        # Process based on JSON structure
        if [[ "$parse_result" =~ ^ARRAY ]]; then
            process_prompts_and_run "$parse_result" "$memory_limit" "$iter" "$current_log_dir"
        else
            error "Invalid JSON parse result format"
        fi
    done
}

# =========================================================================== #
# 6. Main Execution                                                           #
# =========================================================================== #


# --- Main Execution Logic ---
main() {
    parse_command_line "$@"

    # Main execution logic
    if [[ "$ENABLE_CGROUP" == "true" ]]; then
        # Memory-constrained benchmarking
        for memory_limit in "${MEMORY_LIMITS[@]}"; do
            run_benchmarks "$memory_limit"
        done
    else
        # Normal benchmarking
        run_benchmarks
    fi

    log "All benchmarks completed successfully!"
}

# Run main function
main "$@"
