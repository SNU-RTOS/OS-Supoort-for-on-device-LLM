# prompt/ Guide

This directory stores JSON prompt inputs consumed by `run.sh` and `run_cmt_generator.sh`.

## Supported JSON formats

The parser (`tools/prompt/parse_json_prompt.py`) supports both:

1. Single prompt object
2. Array of prompt objects

### Single prompt example

```json
{
  "tokens": 128,
  "prompt": "Write a short summary.",
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "enable_repetition_penalty": false
}
```

### Multi prompt example

```json
[
  {
    "tokens": 128,
    "prompt": "Question 1",
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "enable_repetition_penalty": false
  },
  {
    "tokens": 256,
    "prompt": ["Line 1", "Line 2"],
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "enable_repetition_penalty": true
  }
]
```

Note: `prompt` can be a string or an array of strings.

## Usage

- Main inference:

```bash
./run.sh --file ./prompt/sample_prompt_8_1.json
```

- CMT generation:

```bash
./run_cmt_generator.sh --file ./prompt/sample_prompt_8_1.json
```

## Utilities

- Parse JSON prompt:

```bash
python3 ./tools/prompt/parse_json_prompt.py ./prompt/sample_prompt_8_1.json
```

- Backfill missing hyperparameters:

```bash
python3 ./tools/prompt/update_json_prompt_hyperparams.py
```
