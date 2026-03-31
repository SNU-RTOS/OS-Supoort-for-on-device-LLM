#!/usr/bin/env python3
# prefetch_planner.py
# Prototype version of Prefetch Planner
# Author: GeonHa Park
# Description:
#   Generate a Prefetch Plan based on Weight Chunk Metadata Table (CMT)
#   following the architecture defined in Flash-SLiM Prefetch Planner Design.

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Mapping

from planning.__planner_data_structures__ import PrefetchPlan, WeightChunkInfo
from planning.io_estimator import BandwidthIoTimeEstimator, DirectIoTimeEstimator
from planning.strategy_smart import SmartPlanningStrategy
from planning.strategy_sizeonly import SizeOnlyStrategy
from planning.strategy_fixedpair import FixedPairDecodePlanningStrategy
from planning.stratgey_rechunk import RechunkPlanningStrategy
from planning.strategy_simple import SimplePlanningStrategy
from planning.strategy_base import ChunkKey, PlanningContext, PlanningStrategy
from planning.common import _coerce_to_float, _coerce_to_int


class PrefetchPlanner:
    def __init__(self, cmt_path: str):
        self.cmt_path = cmt_path
        self.metadata: Dict[str, object] = {}
        self.weight_chunks: Dict[str, List[WeightChunkInfo]] = {}
        self.chunk_lookup: Dict[ChunkKey, Mapping[str, object]] = {}
        self.profile_stats: Dict[ChunkKey, float] = {}

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        base_dir = os.path.dirname(self.cmt_path) or os.getcwd()
        return os.path.join(base_dir, path)

    def _resolve_buffer_size(self) -> int:
        key = "weight_chunk_buffer_size"
        if key in self.metadata:
            return _coerce_to_int(self.metadata.get(key), 0)
        return 0

    def _resolve_default_compute_ms(self) -> float:
        return _coerce_to_float(self.metadata.get("default_compute_ms"), 0.0)

    def load_cmt(self) -> None:
        print(
            f"[PrefetchPlanner] Loading Weight Chunk Metadata Table from: {self.cmt_path}"
        )
        with open(self._resolve_path(self.cmt_path), "r", encoding="utf-8") as source:
            cmt_data = json.load(source)

        self.metadata = cmt_data.get("metadata", {}) or {}
        raw_chunks = cmt_data.get("weight_chunks", {}) or {}

        for mode, chunks in raw_chunks.items():
            typed_chunks: List[WeightChunkInfo] = []
            for chunk in chunks:
                chunk_payload = {
                    k: v for k, v in chunk.items() if k != "managed_buffer_index"
                }
                info = WeightChunkInfo(**chunk_payload)
                typed_chunks.append(info)
                self.chunk_lookup[(mode, info.chunk_index, info.origin_offset)] = (
                    chunk_payload
                )
            self.weight_chunks[mode] = typed_chunks

        print(
            f"[PrefetchPlanner] Loaded {sum(len(v) for v in self.weight_chunks.values())} chunks"
        )

    def load_profile_logs(self, pattern: str = "bpf_profile_ops_results_*") -> None:
        print(f"[PrefetchPlanner] Loading profiling logs with pattern: {pattern}")
        base_dir = os.path.dirname(self.cmt_path) or os.getcwd()
        glob_path = os.path.join(base_dir, pattern)
        profile_files = sorted(glob.glob(glob_path))
        if not profile_files:
            print(
                "[PrefetchPlanner] No profiling logs found; avg_compute_time will be left as null"
            )
            return

        aggregator: Dict[ChunkKey, List[float]] = defaultdict(list)
        ops_pattern = re.compile(
            r"^(?P<mode>[A-Z_]+)\[(?P<chunk>\d+),(?P<offset>\d+)\]"
        )

        line_count = 0
        for profile_file in profile_files:
            print(f"[PrefetchPlanner] Parsing profiling log: {profile_file}")
            with open(profile_file, "r", encoding="utf-8") as source:
                for line in source:
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("=")
                        or line.startswith("--")
                        or line.startswith("Ops")
                    ):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    match = ops_pattern.match(parts[0])
                    if not match:
                        continue
                    mode = match.group("mode")
                    chunk_idx = int(match.group("chunk"))
                    # origin_offset = int(match.group("offset"))
                    
                    try:
                        avg_ms = float(parts[2])
                        # line_count += 1
                        # print(f"{line_count} and {parts[2]}")
                    except ValueError:
                        continue
                    aggregator[(mode, chunk_idx)].append(avg_ms)
                    # aggregator[(mode, chunk_idx, origin_offset)].append(avg_ms)
                    # print(mode, chunk_idx, origin_offset, avg_ms)

        self.profile_stats = {
            key: statistics.mean(values) for key, values in aggregator.items()
        }
        print(
            f"[PrefetchPlanner] Loaded profiling stats for {len(self.profile_stats)} chunks"
        )

    def build_context(self) -> PlanningContext:
        return PlanningContext(
            metadata=dict(self.metadata),
            weight_chunks=self.weight_chunks,
            chunk_lookup=self.chunk_lookup,
            profile_stats=self.profile_stats,
        )

    def integrate_profiling_data(self, plan: PrefetchPlan) -> None:
        """Enrich plan entries with avg_compute_time from loaded profiling logs.

        This method is safe to call even when no profiling data was loaded.
        """
        # Determine if any plan entry is missing compute time.
        needs_enrichment = False
        for entries in plan.plan_entries.values():
            for entry in entries:
                if getattr(entry, "avg_compute_time", None) is None:
                    needs_enrichment = True
                    break
            if needs_enrichment:
                break

        if not needs_enrichment:
            print(
                "[PrefetchPlanner] Plan entries already contain avg_compute_time; skipping profiling enrichment"
            )
            return

        if not self.profile_stats:
            print(
                "[PrefetchPlanner] No profiling data available; leaving avg_compute_time unset for some entries"
            )
            return

        print("[PrefetchPlanner] Integrating profiling data into prefetch plan")
        for mode, entries in plan.plan_entries.items():
            for entry in entries:
                # If already set, keep existing value
                if getattr(entry, "avg_compute_time", None) is not None:
                    continue
                key = (mode, entry.chunk_index, entry.origin_offset)
                if key in self.profile_stats:
                    entry.avg_compute_time = self.profile_stats[key]

        print("[PrefetchPlanner] Profiling data integration complete")

    def save_prefetch_plan(self, plan: PrefetchPlan, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as sink:
            json.dump(plan.to_dict(), sink, indent=2)
        print(f"[PrefetchPlanner] Prefetch plan saved to {output_path}")

    def create_strategy(self, strategy_id: str) -> PlanningStrategy:
        normalized = strategy_id.lower()
        if normalized == "simple":
            return SimplePlanningStrategy()

        if normalized == "rechunk":
            buffer_size = self._resolve_buffer_size()
            if buffer_size <= 0:
                raise ValueError(
                    "Re-chunk strategy requires a positive max_buffer_size in metadata"
                )
            fallback_estimator = BandwidthIoTimeEstimator(bandwidth_bytes_per_sec=1e9)
            io_estimator = DirectIoTimeEstimator(fallback=fallback_estimator)
            return RechunkPlanningStrategy(
                max_buffer_size=buffer_size,
                io_estimator=io_estimator,
                default_compute_ms=self._resolve_default_compute_ms(),
            )
        
        if normalized == "fixedpair":
            buffer_size = self._resolve_buffer_size()
            if buffer_size <= 0:
                raise ValueError(
                    "Fixed-pair strategy requires a positive max_buffer_size in metadata"
                )
            fallback_estimator = BandwidthIoTimeEstimator(bandwidth_bytes_per_sec=1e9)
            io_estimator = DirectIoTimeEstimator(fallback=fallback_estimator)
            return FixedPairDecodePlanningStrategy(
                max_buffer_size=buffer_size,
                io_estimator=io_estimator,
                default_compute_ms=self._resolve_default_compute_ms(),
            )
        
        if normalized == "sizeonly":
            buffer_size = self._resolve_buffer_size()
            if buffer_size <= 0:
                raise ValueError(
                    "Size-only strategy requires a positive weight_chunk_buffer_size in metadata"
                )
            fallback_estimator = BandwidthIoTimeEstimator(bandwidth_bytes_per_sec=1e9)
            io_estimator = DirectIoTimeEstimator(fallback=fallback_estimator)
            return SizeOnlyStrategy(
                max_buffer_size=buffer_size,
                io_estimator=io_estimator,
                default_compute_ms=self._resolve_default_compute_ms(),
            )
            
        if normalized == "smart":
            buffer_size = self._resolve_buffer_size()
            if buffer_size <= 0:
                raise ValueError(
                    "Smart strategy requires a positive max_buffer_size in metadata"
                )
            fallback_estimator = BandwidthIoTimeEstimator(bandwidth_bytes_per_sec=1e9)
            io_estimator = DirectIoTimeEstimator(fallback=fallback_estimator)
            from planning.strategy_smart import SmartPlanningStrategy

            KiB = 1 << 10
            MiB = 1 << 20
            GiB = 1 << 30

            return SmartPlanningStrategy(
                max_buffer_size=buffer_size, # buffer_size / 1 GB: 1 << 30 / 1 MB: 1 << 20
                io_estimator=io_estimator,
                default_compute_ms=self._resolve_default_compute_ms(),
            )

        raise ValueError(f"Unknown planning strategy: {strategy_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Prefetch Plan from Weight Chunk Metadata Table",
    )
    parser.add_argument(
        "--cmt",
        default="weight_chunks_metadata_table.json",
        help="Path to weight chunks metadata file in JSON format",
    )
    parser.add_argument(
        "--output",
        default="prefetch_plan.json",
        help="Path to save the generated prefetch plan",
    )
    parser.add_argument(
        "--profile-pattern",
        default="bpf_profile_ops_results_*",
        help="Glob pattern (relative to the CMT directory) for profiling logs",
    )
    parser.add_argument(
        "--strategy",
        choices=["simple", "rechunk", "fixedpair", "sizeonly", "smart"],
        default="simple",
        help="Planning strategy to apply",
    )
    return parser.parse_args()


def run_prefetch_planner(args: argparse.Namespace) -> PrefetchPlan:
    planner = PrefetchPlanner(args.cmt)
    planner.load_cmt()
    planner.load_profile_logs(args.profile_pattern)

    context = planner.build_context()
    strategy = planner.create_strategy(args.strategy)

    print("[PrefetchPlanner] Building prefetch plan...")
    plan = strategy.build(context)
    planner.integrate_profiling_data(plan)
    planner.save_prefetch_plan(plan, args.output)
    return plan


def main() -> None:
    args = parse_args()
    run_prefetch_planner(args)


if __name__ == "__main__":
    main()
