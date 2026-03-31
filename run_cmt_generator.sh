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

# Usage example
#  ./run_cmt_generator.sh --core 3-6 --threads 4 -f ./prompt/sample_prompt_128_1.json

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
MODEL_DIR="${MODEL_PATH}/Llama2-7B"
MODEL_NAME="llama2_7b_q8_ekv1280"

# MODEL_DIR="${MODEL_PATH}/Llama3.2-1B"
# MODEL_NAME="llama3.2_q8_ekv1280"

# MODEL_DIR="${MODEL_PATH}/Llama3.2-3B"
# MODEL_NAME="llama3.2_q8_ekv1024"

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

# MODEL_DIR="${MODEL_PATH}/Qwen2.5-14B"
# MODEL_NAME="qwen2_5_14b_q8_ekv1280"

# MODEL_DIR="${MODEL_PATH}/SmolLM-135M"
# MODEL_NAME="smollm_q8_ekv1280"

BIN="bin/cmt_generator"

# --- Execution Settings ---
TARGET="cpu"           # Default target: cpu | gpu
LOG_ENABLED=false      # Default logging: false
CORE_LIST="all"        # Default core list for taskset
NUM_THREADS=1          # Default number of threads
PROMPT_FILE="${PROMPT_PATH}/sample_prompt_8_1.json" # Default prompt file



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
EOF
    exit 0
}

parse_command_line(){
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
    -h | --help) 
        usage 
        ;;
    *)
        error "Unknown option: $1"
        usage
        ;;
    esac
done

}

# =========================================================================== #
# 4. Core Functions                                                           #
# =========================================================================== #


# Function to run a single prompt
run_with_single_prompt() {
    local TOKENS="$1"
    local PROMPT="$2"  
    local LOG_FILE_PATH="$3"
    local TEMPERATURE="$4"
    local TOP_K="$5"
    local TOP_P="$6"
    local REPETITION_PENALTY="$7"
    local ENABLE_REPETITION_PENALTY="$8"
    local CSV_LOG_FILE_PATH="${LOG_FILE_PATH%.log}.csv"
    local BPF_OPS_PYTHON_PATH="./tools/ebpf/profile_ops.py"
    local BPF_LOG_FILE_PATH="bpf_profile_ops_results_${NUM_THREADS}threads_prefill_${TOKENS}.log"

    banner "LLM inference start (${TARGET^^})"
    log "Model                           : ${MODEL_NAME}"
    log "Tokens requested                : ${TOKENS}"
    # log "Prompt           : ${PROMPT}"
    log "Cores                           : ${CORE_LIST}"
    log "Threads                         : ${NUM_THREADS}"
    log "Temperature                     : ${TEMPERATURE}"
    log "Top-k                           : ${TOP_K}"
    log "Top-p                           : ${TOP_P}"
    log "Repetition penalty              : ${REPETITION_PENALTY}"
    log "Enable rep. penalty             : ${ENABLE_REPETITION_PENALTY}"
    log "Target Processor                : ${TARGET^^}"
    if [[ "$LOG_ENABLED" == "true" ]]; then
        log "Log file                        : ${LOG_FILE_PATH}"
        log "Op-level profiling csv results  : ${CSV_LOG_FILE_PATH}"
        log "BPF profiling log               : ${BPF_LOG_FILE_PATH}"
        log "BPF profiling script            : ${BPF_OPS_PYTHON_PATH}"
    fi

    clear_caches

    # Build command as array (safe quoting)

    if [[ "$LOG_ENABLED" == "true" ]]; then
        local CMD=(
            "${BIN}"
            --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
            --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
            --start_token "<bos>"
            --stop_token "<end_of_turn>"
            --num_threads "${NUM_THREADS}"
            --prompt "${PROMPT}"
            --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
            --temperature "${TEMPERATURE}"
            --top_k "${TOP_K}"
            --top_p "${TOP_P}"
            --repetition_penalty "${REPETITION_PENALTY}"
            --csv_profile_output_path "$CSV_LOG_FILE_PATH"
            --model_dump_file_path "${MODEL_DIR}/${MODEL_NAME}_dump"
            --op_tensor_byte_stats
            --dump_tensor_details
            --profile_steps 10
        )
    else
        local CMD=(
            "${BIN}"
            --tflite_model "${MODEL_DIR}/${MODEL_NAME}.tflite"
            --sentencepiece_model "${MODEL_DIR}/tokenizer.model"
            --start_token "<bos>"
            --stop_token "<end_of_turn>"
            --num_threads "${NUM_THREADS}"
            --prompt "${PROMPT}"
            --weight_cache_path "${MODEL_DIR}/${MODEL_NAME}.xnnpack_cache"
            --temperature "${TEMPERATURE}"
            --top_k "${TOP_K}"
            --top_p "${TOP_P}"
            --repetition_penalty "${REPETITION_PENALTY}"
            --model_dump_file_path "${MODEL_DIR}/${MODEL_NAME}_dump"
            --op_tensor_byte_stats
            --dump_tensor_details
            --profile_steps 10
        )
    fi
    

    # Add repetition penalty flag if enabled
    if [[ "${ENABLE_REPETITION_PENALTY}" == "true" ]]; then
        CMD+=(--enable_repetition_penalty)
    fi
    
    # Add taskset if specified
    [[ "${CORE_LIST}" != "all" ]] && CMD=(taskset -c "${CORE_LIST}" "${CMD[@]}")

    # Start BPF profiler in the background with proper session management
    banner "--- BPF PROFILER START ---"
    run_bpf "$BPF_OPS_PYTHON_PATH" "$BIN" "$BPF_LOG_FILE_PATH" &
    BG_PID=$! # Get background process PID (run_bpf)
    sleep 3 # Give some time for BPF to initialize

    banner "--- C++ Binary Execution START ---"
    banner "Command: ${CMD[*]}"
    sudo "${CMD[@]}"
    banner "--- C++ Binary Execution END ---"

    banner "--- BPF PROFILER END ---"
    cleanup_bpf "$BPF_OPS_PYTHON_PATH" "$BG_PID"
    log "BPF profiling log saved to ${BPF_LOG_FILE_PATH}"

    if [[ "$LOG_ENABLED" == "true" ]]; then
        log "Log saved to ${LOG_FILE_PATH}"
    fi
    
    # Generate analysis report
    # python3 tools/model_dump/tensor_visualization.py \
    #         "${MODEL_DIR}/${MODEL_NAME}_dump.log" \
    #          "${MODEL_DIR}/${MODEL_NAME}_analysis_report.txt" \
    #          "${MODEL_DIR}/${MODEL_NAME}_analysis_data.json"

    
    # run prefetch_planner
    # python3 ./tools/model_prefetch_planner/prefetch_planner.py \
    #     --cmt weight_chunks_metadata_table.json \
    #     --output prefetch_plan_simple_${NUM_THREADS}_${TOKENS}.json \
    #     --profile-pattern ${BPF_LOG_FILE_PATH} \
    #     --strategy simple
    
    # python3 ./tools/model_prefetch_planner/prefetch_planner.py \
    #     --cmt weight_chunks_metadata_table.json \
    #     --output prefetch_plan_rechunk_${NUM_THREADS}_${TOKENS}.json \
    #     --profile-pattern ${BPF_LOG_FILE_PATH} \
    #     --strategy rechunk

    python3 ./tools/model_prefetch_planner/prefetch_planner.py \
        --cmt weight_chunks_metadata_table.json \
        --output prefetch_plan_fixedpair_${NUM_THREADS}_${TOKENS}.json \
        --profile-pattern ${BPF_LOG_FILE_PATH} \
        --strategy fixedpair

    # python3 ./tools/model_prefetch_planner/prefetch_planner.py \
    #     --cmt weight_chunks_metadata_table.json \
    #     --output prefetch_plan_sizeonly_${NUM_THREADS}_${TOKENS}.json \
    #     --profile-pattern ${BPF_LOG_FILE_PATH} \
    #     --strategy sizeonly

    # python3 ./tools/model_prefetch_planner/prefetch_planner.py \
    #     --cmt weight_chunks_metadata_table.json \
    #     --output prefetch_plan_smart_${NUM_THREADS}_${TOKENS}.json \
    #     --profile-pattern ${BPF_LOG_FILE_PATH} \
    #     --strategy smart
}

# =========================================================================== #
# 5. Main Execution                                                           #
# =========================================================================== #

main() {
    parse_command_line "$@"

    # Setup log directory
    local model_id="${MODEL_NAME}"
    local target_id="${TARGET}"
    local log_dir="${LOG_DIR_BASE}/${model_id}_${target_id}_cmt_generator"
    
    if [[ "$LOG_ENABLED" == "true" ]]; then
        if [[ -d "${log_dir}" ]]; then
            log "Log directory exists: ${log_dir} (appending)"
        else
            ensure_dir "${log_dir}"
            log "Created log directory: ${log_dir}"
        fi
    fi
    
    # Parse JSON file
    local parse_result=$(parse_json_file "${PROMPT_FILE}")
    local prompt_count=$(echo "$parse_result" | head -n1 | awk '{print $2}')
    
    log "Found $prompt_count prompts in JSON file. Processing all..."
    
    local prompt_index=0
    local current_prompt="" 
    local in_prompt=false
    local current_tokens current_temperature current_top_k current_top_p \
        current_repetition_penalty current_enable_repetition_penalty
    
    # Loop through each line of the parse result
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
            log_file="${log_dir}/run_${current_tokens}_${prompt_index_display}_${timestamp}.log"

            log "Executing LLM Inference"
            execute_with_log "$log_file" run_with_single_prompt "$current_tokens" "$current_prompt" "$log_file" \
                "$current_temperature" "$current_top_k" "$current_top_p" \
                "$current_repetition_penalty" "$current_enable_repetition_penalty" 
                
        elif [[ "$in_prompt" == "true" ]]; then
            if [[ -z "$current_prompt" ]]; then
                current_prompt="$line"
            else
                current_prompt="$current_prompt"$'\n'"$line"
            fi
        fi
    done <<< "$parse_result"

    log "All benchmarks completed successfully!"
}

# Run main function
main "$@"


