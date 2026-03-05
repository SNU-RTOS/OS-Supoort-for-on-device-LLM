#!/usr/bin/env python3
"""
Parse a prefetch-plan JSON (like the one you pasted) and compute, per segment:
- overlap (ms) and overlap_pct (of that segment's IO time)
- stall (ms): IO not hidden by previous segment's compute
- slack (ms): previous segment compute left unused after hiding this segment's IO

Computed separately for PREFILL and DECODE.
- PREFILL is non-circular by default (last segment has no "next/prev" wrap).
- DECODE is circular by default, but can be disabled.

Outputs ONE CSV file containing rows for both modes + summary rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Segment:
    mode: str
    seg_id: int
    chunk_count: int
    total_aligned_size: int
    total_origin_size: int
    start_origin_offset: int
    compute_ms: float
    io_ms: float


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_segments(plan: Dict[str, Any], mode: str) -> List[Segment]:
    """
    Reads: data["prefetch_plan"][mode] which is a dict like {"0": {...}, "1": {...}, ...}
    Sorts by integer segment id.
    """
    pp = plan.get("prefetch_plan", {})
    mode_obj = pp.get(mode, {})
    if not isinstance(mode_obj, dict):
        raise ValueError(f'prefetch_plan["{mode}"] is not an object/dict')

    seg_ids = sorted((int(k) for k in mode_obj.keys()))
    segs: List[Segment] = []
    for sid in seg_ids:
        s = mode_obj[str(sid)]
        segs.append(
            Segment(
                mode=mode,
                seg_id=sid,
                chunk_count=int(s.get("chunk_count", 0)),
                total_aligned_size=int(s.get("total_aligned_size", 0)),
                total_origin_size=int(s.get("total_origin_size", 0)),
                start_origin_offset=int(s.get("start_origin_offset", 0)),
                compute_ms=float(s.get("total_avg_compute_time", 0.0)),
                io_ms=float(s.get("estimated_io_time_ms", 0.0)),
            )
        )
    return segs


def _compute_metrics(
    segs: List[Segment],
    circular: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    We model a simple 2-stage pipeline:
      - While computing segment (i-1), we prefetch IO for segment i.
    For segment i:
      overlap_i = min(compute_{i-1}, io_i)
      stall_i   = max(0, io_i - compute_{i-1})    # extra wait before compute_i can start
      slack_i   = max(0, compute_{i-1} - io_i)    # unused compute time after IO_i is fully hidden

    Edge handling:
      - If not circular: segment 0 has no previous compute to overlap its IO, so overlap=0, stall=io_0, slack=0.
        And segment last has no "next" dependency, but that does not affect per-segment i definition above.
      - If circular: segment 0 uses previous=last.
    """
    n = len(segs)
    rows: List[Dict[str, Any]] = []

    tot_io = 0.0
    tot_compute = 0.0
    tot_overlap = 0.0
    tot_stall = 0.0
    tot_slack = 0.0

    for i, seg in enumerate(segs):
        tot_io += seg.io_ms
        tot_compute += seg.compute_ms

        has_prev = (i > 0) or circular
        if has_prev and n > 0:
            prev_idx = (i - 1) % n
            prev_compute = segs[prev_idx].compute_ms
            overlap = min(prev_compute, seg.io_ms)
            stall = max(0.0, seg.io_ms - prev_compute)
            slack = max(0.0, prev_compute - seg.io_ms)
        else:
            # first segment in non-circular mode
            overlap = 0.0
            stall = seg.io_ms
            slack = 0.0

        tot_overlap += overlap
        tot_stall += stall
        tot_slack += slack

        overlap_pct = (overlap / seg.io_ms * 100.0) if seg.io_ms > 0 else 0.0

        rows.append(
            {
                "mode": seg.mode,
                "segment_id": seg.seg_id,
                "chunk_count": seg.chunk_count,
                "total_aligned_size": seg.total_aligned_size,
                "total_origin_size": seg.total_origin_size,
                "start_origin_offset": seg.start_origin_offset,
                "compute_ms": seg.compute_ms,
                "io_ms": seg.io_ms,
                "overlap_ms": overlap,
                "overlap_pct_of_io": overlap_pct,
                "stall_ms": stall,
                "slack_ms": slack,
                "circular": circular,
            }
        )

    summary = {
        "mode": segs[0].mode if segs else "",
        "segments": n,
        "circular": circular,
        "total_io_ms": tot_io,
        "total_compute_ms": tot_compute,
        "total_overlap_ms": tot_overlap,
        "total_stall_ms": tot_stall,
        "total_slack_ms": tot_slack,
        "total_overlap_pct_of_io": (tot_overlap / tot_io * 100.0) if tot_io > 0 else 0.0,
    }
    return rows, summary


def write_single_csv(
    out_path: Path,
    rows_by_mode: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
) -> None:
    # One file: write all segment rows, then a blank line, then summary rows.
    fieldnames = [
        "mode",
        "segment_id",
        "chunk_count",
        "total_aligned_size",
        "total_origin_size",
        "start_origin_offset",
        "compute_ms",
        "io_ms",
        "overlap_ms",
        "overlap_pct_of_io",
        "stall_ms",
        "slack_ms",
        "circular",
    ]

    summary_fieldnames = [
        "mode",
        "segments",
        "circular",
        "total_io_ms",
        "total_compute_ms",
        "total_overlap_ms",
        "total_stall_ms",
        "total_slack_ms",
        "total_overlap_pct_of_io",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_by_mode:
            w.writerow({k: r.get(k, "") for k in fieldnames})

        # separator
        f.write("\n")

        w2 = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w2.writeheader()
        for s in summaries:
            w2.writerow({k: s.get(k, "") for k in summary_fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path, help="Input prefetch-plan JSON file")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("prefetch_metrics.csv"),
        help="Output CSV path (single file)",
    )

    # Circularity controls
    ap.add_argument(
        "--decode-circular",
        action="store_true",
        default=True,
        help="Make DECODE circular (default: enabled)",
    )
    ap.add_argument(
        "--no-decode-circular",
        dest="decode_circular",
        action="store_false",
        help="Disable circular DECODE",
    )
    ap.add_argument(
        "--prefill-circular",
        action="store_true",
        default=False,
        help="Make PREFILL circular (default: disabled)",
    )

    args = ap.parse_args()

    data = _load_json(args.json_path)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    # PREFILL
    prefill_segs = _extract_segments(data, "PREFILL")
    prefill_rows, prefill_summary = _compute_metrics(prefill_segs, circular=bool(args.prefill_circular))
    all_rows.extend(prefill_rows)
    summaries.append(prefill_summary)

    # DECODE
    decode_segs = _extract_segments(data, "DECODE")
    decode_rows, decode_summary = _compute_metrics(decode_segs, circular=bool(args.decode_circular))
    all_rows.extend(decode_rows)
    summaries.append(decode_summary)

    write_single_csv(args.out, all_rows, summaries)
    print(f"Wrote: {args.out.resolve()}")
    return 0

"""
Usage
1. default: PREFILL non-circular, DECODE circular
python prefetch_plan_analyzer.py prefetch_plan.json -o analysis.csv
2. disable circular DECODE
python prefetch_plan_analyzer.py prefetch_plan.json -o analysis.csv --no-decode-circular
3. (optional) if you ever want to test circular PREFILL
python prefetch_plan_analyzer.py prefetch_plan.json -o analysis.csv --prefill-circular
"""
if __name__ == "__main__":
    raise SystemExit(main())
