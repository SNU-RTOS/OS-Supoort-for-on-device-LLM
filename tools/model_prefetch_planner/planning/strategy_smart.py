"""Local-improvement planning strategy with memory-budgeted segment adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .__planner_data_structures__ import (
    PrefetchPlan,
    PrefetchPlanEntry,
    WeightChunkInfo,
)
from .strategy_base import PlanningContext, PlanningStrategy
from .io_estimator import IoTimeEstimator

from .common import (
    sort_chunk_list,
    _compute_gap_bytes,
    sum_chunk_aligned_size,
    sum_chunk_compute_time,
    resolve_chunk_payload,
)


@dataclass
class SmartPlanningStrategy(PlanningStrategy):
    """
    Strategy outline:
      1) Initialize segments as one chunk per segment.
      2) Iterate boundaries i -> i+1. If compute(seg_i) < io(seg_{i+1}), try moving
         chunks from the head of seg_{i+1} into tail of seg_i.
      3) Accept a move only if:
         - peak memory constraint holds (pairwise consecutive sum, plus wrap if circular),
         - and local objective decreases.

    Objective ("stall" per your definition):
      gap(i, i+1) = abs(io_time(seg_{i+1}) - compute_time(seg_i))

    NOTE: This is a draft. You may want to tune:
      - objective weighting for wrap edge in decode
      - allowed move direction(s) (this only pulls i+1 -> i)
      - max passes / early stopping
    """

    max_buffer_size: int
    io_estimator: IoTimeEstimator
    default_compute_ms: float = 0.0

    # Whether to treat segment sequence as circular (decode-like)
    circular: bool = False

    # Hard limit to avoid long runtimes on large chunk counts
    max_passes: int = 50

    def build(self, context: PlanningContext) -> PrefetchPlan:
        self._validate_chunks(context)

        plan = PrefetchPlan(metadata=dict(context.metadata))
        for mode, chunks in context.weight_chunks.items():
            ordered_chunks = sort_chunk_list(chunks)
            for chunk in ordered_chunks:
                key = (mode, chunk.chunk_index)
                print(key)
                print(context.profile_stats.get(key, self.default_compute_ms))

            logical_groups, group_io_times = self._plan_mode(mode, ordered_chunks, context)

            plan.plan_entries[mode] = []
            for group_idx, group in enumerate(logical_groups):
                group_io_time = group_io_times[group_idx] if group_idx < len(group_io_times) else None

                prefetch_io_order = group_idx
                for chunk in group:
                    chunk_payload = resolve_chunk_payload(context.chunk_lookup, mode, chunk)

                    key = (mode, chunk.chunk_index)
                    print(key)
                    compute_ms = context.profile_stats.get(key, self.default_compute_ms)

                    entry = PrefetchPlanEntry(
                        mode=mode,
                        chunk_data=chunk_payload,
                        io_order=prefetch_io_order,
                        avg_compute_time=compute_ms,
                        estimated_io_time_ms=group_io_time,
                    )
                    plan.plan_entries[mode].append(entry)

        return plan

    # ----------------------------
    # Core planning logic (mode)
    # ----------------------------
    def _plan_mode(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        context: PlanningContext,
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        if not chunks:
            return [], []

        # 1) init: one chunk per segment
        segs: List[List[WeightChunkInfo]] = [[c] for c in chunks]

        # 2) sanity
        if not self._peak_memory_ok(mode, segs):
            raise ValueError(
                f"[{mode}] initial 1-chunk segments violate max_buffer_size={self.max_buffer_size}."
            )

        # Define a stall objective that matches your “stall difference” intent.
        # (If you already have _local_obj/_affected_boundaries, you can keep them,
        # but boundary-local is usually clearer for this greedy.)
        def boundary_stall(i: int) -> float:
            left = segs[i]
            right = segs[i + 1]
            left_compute = self._segment_compute_ms(mode, left, context)
            right_io = self._segment_io_ms(mode, right)
            return max(0.0, right_io - left_compute)

        for _ in range(self.max_passes):
            any_change = False

            i = 0
            while i < len(segs) - 2:
                # Keep moving HEAD from right -> left while it improves this boundary stall
                while True:
                    left = segs[i]
                    right = segs[i + 1]
                    if not right:
                        # Shouldn’t happen often, but handle defensively
                        del segs[i + 1]
                        any_change = True
                        break

                    prev = boundary_stall(i)

                    cand = right[0]  # <-- HEAD ONLY

                    # Optional contiguity: if head is not contiguous, stop this boundary
                    if not self._check_chunk_contiguity(left, right, cand):
                        break

                    # Tentative move
                    left.append(cand)
                    del right[0]

                    # Memory feasibility for current segs
                    if not self._peak_memory_ok(mode, segs):
                        # rollback and STOP boundary (head-only policy)
                        right.insert(0, cand)
                        left.pop()
                        break

                    # If right became empty, deleting it changes the next boundary structure
                    if len(right) == 0:
                        tmp = segs[: i + 1] + segs[i + 2 :]
                        if not self._peak_memory_ok(mode, tmp):
                            # rollback and STOP boundary
                            segs[i + 1].insert(0, cand)  # segs[i+1] is right (currently empty list)
                            left.pop()
                            break
                        # commit deletion
                        del segs[i + 1]
                        any_change = True
                        # After deletion, boundary (i, i+1) now refers to old (i, i+2),
                        # so continue trying at same i.
                        continue

                    new = boundary_stall(i)

                    if new < prev:
                        any_change = True
                        # Commit succeeded; continue at SAME boundary,
                        # trying the NEW head right[0].
                        continue
                    else:
                        # rollback and STOP boundary
                        right.insert(0, cand)
                        left.pop()
                        break

                i += 1

            if not any_change:
                break

        group_io_times = [self._segment_io_ms(mode, g) for g in segs]
        return segs, group_io_times



    def _plan_mode_old(
        self,
        mode: str,
        chunks: Sequence[WeightChunkInfo],
        context: PlanningContext,
    ) -> Tuple[List[List[WeightChunkInfo]], List[float]]:
        if not chunks:
            return [], []

        # 1) init: one chunk per segment
        segs: List[List[WeightChunkInfo]] = [[c] for c in chunks]

        # 2) sanity: memory constraint for baseline
        if not self._peak_memory_ok(mode, segs):
            raise ValueError(
                f"[{mode}] initial 1-chunk segments violate max_buffer_size={self.max_buffer_size}. "
                "Your peak memory model may be stricter than pairwise sums, or buffer is too small."
            )

        # 3) local improvement passes
        for _ in range(self.max_passes):
            any_change = False

            i = 0
            # We iterate boundaries i -> i+1 (and wrap separately if circular)
            length = len(segs)
            while i < len(segs) - 1:
                print(f"Segment {i}")
                left = segs[i]
                right = segs[i + 1]

                left_compute = self._segment_compute_ms(mode, left, context)
                right_io = self._segment_io_ms(mode, right)

                # Trigger: compute(left) < io(right)  => boundary likely "stalling"
                if left_compute < right_io and right:
                    # local objective over affected boundaries
                    pairs = self._affected_boundaries(i, len(segs))
                    prev_obj = self._local_obj(mode, segs, pairs, context)

                    # Try moving from head of right into tail of left, one chunk at a time
                    moved = False
                    j = 0
                    while j < len(right):
                        print(" Trying to move node", j, "from right to left")
                        cand = right[j]

                        # optional contiguity check (keeps origin-space contiguity)
                        if not self._check_chunk_contiguity(left, right, cand):
                            break

                        # Apply move: right[j] -> end(left)
                        left.append(cand)
                        del right[j]

                        # If right becomes empty, we will consider deleting it
                        # First check memory feasibility
                        if not self._peak_memory_ok(mode, segs):
                            # rollback
                            right.insert(j, cand)
                            left.pop()
                            # GPT
                            j += 1
                            continue # --> 더 증가시킬 필요가 없음 왜냐면 non-contiguous
                            # break

                        # Evaluate new objective (handle potential deletion)
                        if len(right) == 0:
                            tmp = segs[: i + 1] + segs[i + 2 :]
                            if not self._peak_memory_ok(mode, tmp):
                                right.insert(j, cand)
                                left.pop()
                                j += 1
                                continue
                            tmp_i = min(i, max(0, len(tmp) - 2))
                            tmp_pairs = self._affected_boundaries(tmp_i, len(tmp)) if len(tmp) >= 2 else []
                            new_obj = self._local_obj(mode, tmp, tmp_pairs, context) if tmp_pairs else 0.0
                        else:
                            pairs2 = self._affected_boundaries(i, len(segs))
                            new_obj = self._local_obj(mode, segs, pairs2, context)

                        if new_obj < prev_obj:
                            prev_obj = new_obj
                            moved = True
                            any_change = True

                            # delete empty segment if needed
                            if len(right) == 0:
                                del segs[i + 1]
                                break

                            # Keep trying to move the next head (still at index j)
                            continue
                        else:
                            # rollback and move on
                            right.insert(j, cand)
                            left.pop()
                            j += 1
                            # break # No point in trying further chunks; they won't be contiguous

                    if moved:
                        # Move to next boundary (or re-evaluate same i if segment count changed)
                        if i >= len(segs) - 1:
                            break
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1

            if not any_change:
                break

        group_io_times = [self._segment_io_ms(mode, g) for g in segs]
        return segs, group_io_times

    # ----------------------------
    # Objective + boundary helpers
    # ----------------------------

    def _gap_abs_ms(
        self,
        mode: str,
        prev_seg: List[WeightChunkInfo],
        next_seg: List[WeightChunkInfo],
        context: PlanningContext,
    ) -> float:
        # abs(io(next) - compute(prev))
        io_next = self._segment_io_ms(mode, next_seg)
        comp_prev = self._segment_compute_ms(mode, prev_seg, context)
        return abs(io_next - comp_prev)

    def _affected_boundaries(self, i: int, n: int) -> List[Tuple[int, int]]:
        """
        Boundaries affected by moving between seg[i] and seg[i+1]:
          (i-1, i), (i, i+1), (i+1, i+2)
        For non-circular we clamp to valid indices.
        """
        pairs: List[Tuple[int, int]] = []

        def add(a: int, b: int) -> None:
            if 0 <= a < n and 0 <= b < n and a != b:
                pairs.append((a, b))

        add(i - 1, i)
        add(i, i + 1)
        add(i + 1, i + 2)

        # de-dupe
        out: List[Tuple[int, int]] = []
        seen = set()
        for p in pairs:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _local_obj(
        self,
        mode: str,
        segs: List[List[WeightChunkInfo]],
        pairs: List[Tuple[int, int]],
        context: PlanningContext,
    ) -> float:
        # interior affected boundaries
        obj = sum(self._gap_abs_ms(mode, segs[a], segs[b], context) for (a, b) in pairs)

        # wrap-aware penalty (read seg0 vs compute seg_last)
        # This does NOT imply we can read them together; it's just an end-to-end "between decode" mismatch term.
        wrap_weight = 1.0
        if self.circular and len(segs) >= 2:
            obj += self.wrap_weight * self._gap_abs_ms(mode, segs[-1], segs[0], context)

        return obj

    # ----------------------------
    # Memory feasibility (pairwise peak)
    # ----------------------------

    def _is_circular_mode(self, mode: str) -> bool:
        if mode == "DECODE":
            return True
        else:
            return False

    def _peak_memory_ok(self, mode: str, segs: List[List[WeightChunkInfo]]) -> bool:
        if not segs:
            return True

        peak = 0
        for i in range(len(segs) - 1):
            peak = max(peak, sum_chunk_aligned_size(segs[i]) + sum_chunk_aligned_size(segs[i + 1]))

        if self._is_circular_mode(mode) and len(segs) >= 2:
            peak = max(peak, sum_chunk_aligned_size(segs[-1]) + sum_chunk_aligned_size(segs[0]))
        elif len(segs) == 1:
            peak = max(peak, sum_chunk_aligned_size(segs[0]))

        return peak <= self.max_buffer_size

    # ----------------------------
    # Segment time estimation
    # ----------------------------

    def _segment_compute_ms(
        self,
        mode: str,
        seg: List[WeightChunkInfo],
        context: PlanningContext,
    ) -> float:
        return float(
            sum_chunk_compute_time(
                mode,
                seg,
                context.profile_stats,
                default_compute_ms=self.default_compute_ms,
            )
        )

    def _segment_io_ms(self, mode: str, seg: List[WeightChunkInfo]) -> float:
        if not seg:
            return 0.0
        try:
            # Use gap_bytes=0 for group estimate (intra-group contiguous reads are already encoded in chunk layout).
            return float(self.io_estimator.estimate(mode, seg, gap_bytes=0))
        except Exception:
            return 0.0

    # ----------------------------
    # Contiguity (optional invariant)
    # ----------------------------

    def _check_chunk_contiguity(
        self,
        left_seg: List[WeightChunkInfo],
        right_seg: List[WeightChunkInfo],
        candidate: WeightChunkInfo,
    ) -> bool:
        """
        Optional: preserve origin-space contiguity in the left segment after appending.
        If you don't need this, you can always return True.
        """
        if not left_seg:
            return True
        last = left_seg[-1]
        expected = last.origin_offset + last.origin_size
        return candidate.origin_offset == expected

    # ----------------------------
    # Wrap-boundary improvement
    # ----------------------------

    def _improve_wrap_boundary(
        self,
        mode: str,
        segs: List[List[WeightChunkInfo]],
        context: PlanningContext,
    ) -> bool:
        """
        Try improving the wrap boundary (last -> first) using the same "pull" move:
        move chunks from head of seg[0] into tail of seg[-1] if beneficial.

        This helps decode plans where circular overlap matters.
        """
        if len(segs) < 2:
            return False

        last = segs[-1]
        first = segs[0]

        last_compute = self._segment_compute_ms(mode, last, context)
        first_io = self._segment_io_ms(mode, first)

        if not (last_compute < first_io and first):
            return False

        # objective around wrap: consider (n-2,n-1), (n-1,0), (0,1)
        n = len(segs)
        pairs = [(n - 2, n - 1), (n - 1, 0), (0, 1)]
        prev_obj = self._local_obj(mode, segs, pairs, context)

        j = 0
        while j < len(first):
            cand = first[j]

            # contiguity: last_seg tail + cand
            if not self._check_chunk_contiguity(last, first, cand):
                break

            # apply move: first[j] -> end(last)
            last.append(cand)
            del first[j]

            if not self._peak_memory_ok(mode, segs):
                first.insert(j, cand)
                last.pop()
                j += 1
                continue

            # if first becomes empty, deletion changes indices; handle conservatively
            if len(first) == 0:
                tmp = segs[:-1]  # remove empty first? careful: first is segs[0]
                # Actually rebuild: segs[1:] plus last (already includes moved)
                tmp = segs[1:-1] + [segs[-1]]
                if not self._peak_memory_ok(mode, tmp):
                    first.insert(j, cand)
                    last.pop()
                    j += 1
                    continue
                # compute objective on tmp is messy; accept only if strictly reduces wrap gap itself
                new_wrap = abs(self._segment_io_ms(mode, tmp[0]) - self._segment_compute_ms(mode, tmp[-1], context)) if len(tmp) >= 2 else 0.0
                old_wrap = abs(first_io - last_compute)
                if new_wrap < old_wrap:
                    # commit deletion on original segs
                    del segs[0]
                    return True
                # rollback
                segs.insert(0, first)  # restore structure (defensive)
                first.insert(j, cand)
                last.pop()
                j += 1
                continue

            new_obj = self._local_obj(mode, segs, pairs, context)
            if new_obj < prev_obj:
                prev_obj = new_obj
                # keep going (same j now points to next head)
                continue
            else:
                # rollback
                first.insert(j, cand)
                last.pop()
                j += 1

        return prev_obj < self._local_obj(mode, segs, pairs, context)

    # ----------------------------
    # Validation
    # ----------------------------

    def _validate_chunks(self, context: PlanningContext) -> None:
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        for mode, chunks in context.weight_chunks.items():
            for chunk in chunks:
                if chunk.aligned_size > self.max_buffer_size:
                    raise ValueError(
                        f"Chunk {mode}[{chunk.chunk_index}] size {chunk.aligned_size} exceeds buffer {self.max_buffer_size}"
                    )
