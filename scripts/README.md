# scripts/ Guide

This directory contains helper scripts for build/setup/model operations.

## Scripts

- `install_prerequisites.sh`
  - Installs development dependencies (apt/dpkg based)
  - Intended for Debian/Ubuntu-like Linux environments

- `build-benchmark_util.sh`
  - Builds LiteRT `benchmark_model` and copies it to `tools/bin/benchmark_model`

- `convert-llama3.2-3b.sh`
  - Local workflow script for model conversion/export
  - Environment-specific path placeholders must be edited before use

- `util-set_swap.sh`
  - Recreates `/swap.img` with requested size (default: 8GB)
  - Example: `./scripts/util-set_swap.sh 16`

- `util-upload_models.sh`
  - Upload helper using `sshpass` + `rsync`
  - Requires editing remote/local path and credentials in script

- `utils.sh`
  - Shared shell utility functions used by top-level runners/builders
  - Logging helpers, cache clear, build config setup, BPF process helpers

## Typical usage from repository root

```bash
./scripts/install_prerequisites.sh
./scripts/build-benchmark_util.sh
./scripts/util-set_swap.sh 8
```

## Notes

- Most scripts expect `.env` to be configured at repository root.
- Some scripts require `sudo`.
- Utility scripts with placeholders (IP/user/password/path) are templates and should be customized locally.
