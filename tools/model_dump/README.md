# model_dump/ Guide

This directory provides utilities for dumping and analyzing LiteRT/TFLite model structure and weight-cache-related metadata.

## Main files

- `dump_llm_nodes.cc`: core C++ model dump implementation
- `build_model_parser.sh`: builds `dump_llm_nodes` (or requested binary name) via Bazel and copies it locally
- `parse_model.sh`: batch runner over selected model directory (`*.tflite`)
- `tensor_visualization.py`: post-processes dump log into analysis outputs
- `weight_cache_check.bt`: optional tracing helper

## Build

From `tools/model_dump/`:

```bash
./build_model_parser.sh
```

This builds `tools/model_dump:dump_llm_nodes` and copies the result to `tools/model_dump/dump_llm_nodes`.

## Batch run

`parse_model.sh` is a convenience runner. It:

1. builds the dump binary,
2. iterates over models in `destination_dir`,
3. executes dump,
4. runs `tensor_visualization.py`.

Run:

```bash
./parse_model.sh
```

Before running, adjust `destination_dir` inside `parse_model.sh` for your model location.

## Direct run example

```bash
./dump_llm_nodes \
  --tflite_model /path/to/model.tflite \
  --weight_cache_path /path/to/model.xnnpack_cache \
  --dump_file_path /path/to/model_dump.log \
  --num_threads 1 \
  --dump_tensor_details \
  --op_tensor_byte_stats
```

## Outputs

Typical outputs per model:

- `<model>_dump.log`
- `<model>_analysis_report.txt`
- `<model>_analysis_data.json`
