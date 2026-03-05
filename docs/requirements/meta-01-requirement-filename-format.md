# Requirement Filename Format

This document defines the markdown filename convention for requirement specs.

## Pattern

`<domain>-<sequence>-<slug>.md`

- `domain`: lowercase category token (`meta`, `memory`, `runtime`, `planner`, etc.)
- `sequence`: two-digit incremental number (`01`, `02`, ...)
- `slug`: short lowercase kebab-case summary

Examples:

- `memory-01-decoder-memory-profiling.md`
- `runtime-02-prefetch-fallback-policy.md`
- `planner-03-overlap-scheduling-heuristics.md`

## Rules

1. Use lowercase letters, digits, and hyphens only.
2. Keep the slug concise and implementation-oriented.
3. Keep one requirement topic per file.
4. Do not rename existing files unless scope has changed substantially.
5. When splitting scope, create a new sequence number instead of overloading one file.

## Status tags (inside file content)

Use one status marker in the document header:

- `Draft`
- `Planned`
- `In Progress`
- `Done`
