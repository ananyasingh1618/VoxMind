#!/usr/bin/env python3
"""System-level evaluation: p50/p95 latency, failure rate, and pipeline
completion rate, computed over real persisted execution data across every
user (a pipeline-health audit, not a user's private data - same reasoning
as the analytics dashboard's model-version/pipeline-health sections).

Two real pipelines have well-defined, distinct execution outcomes in this
codebase and are evaluated separately rather than merged into one
ambiguous "success" count:

  1. The Phase 5 real-time voice loop (`voice_turns`), whose `status`
     column already distinguishes completed / partial / failed /
     interrupted / pending (see models/voice_turn.py). Latency percentiles
     are computed only over `stage_latencies_ms.total_ms` from genuinely
     `completed` turns - a failed or interrupted turn's latency numbers are
     incomplete by definition and are excluded, not padded or estimated.
     `pending` turns (still in flight, or a stale/crashed process) are
     excluded from the completion-rate denominator entirely: they are not
     a resolved outcome yet, so counting them as neither success nor
     failure would misrepresent the real completion rate either way.

  2. Real LLM generation attempts (`llm_generations`), which distinguish
     three real states this script never blurs together: a real attempt
     that succeeded (`model_name` set, no `error_message`), a real attempt
     that failed (`model_name` set, `error_message` present), and an
     unavailable-provider call where no LLM was even configured
     (`model_name IS NULL`). Per the Phase 7 instruction, unavailable-
     provider calls are never counted as successful, and are excluded from
     the failure-rate denominator entirely (they were never an attempt).

Usage:
    python -m ml.evaluation.evaluate_system
"""
from __future__ import annotations

import argparse
import asyncio


async def run_evaluation(_args: argparse.Namespace) -> None:
    from sqlalchemy import select

    from voxmind.db.session import AsyncSessionLocal
    from voxmind.models.llm_generation import LlmGeneration
    from voxmind.models.voice_turn import VoiceTurn
    from voxmind.repositories.evaluation_run_repository import EvaluationRunRepository
    from voxmind.services.analytics_service import _latency_stats

    async with AsyncSessionLocal() as session:
        voice_turns = list((await session.execute(select(VoiceTurn))).scalars().all())
        generations = list((await session.execute(select(LlmGeneration))).scalars().all())

        voice_status_counts: dict[str, int] = {}
        for vt in voice_turns:
            voice_status_counts[vt.status] = voice_status_counts.get(vt.status, 0) + 1
        resolved_voice_turns = [vt for vt in voice_turns if vt.status != "pending"]
        completed_voice_turns = [vt for vt in voice_turns if vt.status == "completed"]
        completion_rate = (
            len(completed_voice_turns) / len(resolved_voice_turns) if resolved_voice_turns else None
        )
        total_latencies = [
            float(vt.stage_latencies_ms["total_ms"])
            for vt in completed_voice_turns
            if "total_ms" in (vt.stage_latencies_ms or {})
        ]
        voice_latency = _latency_stats(total_latencies)

        real_attempts = [g for g in generations if g.model_name is not None]
        unavailable = [g for g in generations if g.model_name is None]
        real_failures = [g for g in real_attempts if g.error_message]
        real_successes = [g for g in real_attempts if not g.error_message]
        failure_rate = len(real_failures) / len(real_attempts) if real_attempts else None
        llm_latency = _latency_stats([float(g.latency_ms) for g in real_successes])

        metrics = {
            "voice_turn_pipeline": {
                "status_distribution": voice_status_counts,
                "n_total": len(voice_turns),
                "n_resolved": len(resolved_voice_turns),
                "n_excluded_pending": len(voice_turns) - len(resolved_voice_turns),
                "completion_rate": completion_rate,
                "total_latency_ms": voice_latency.model_dump(),
            },
            "llm_generation": {
                "n_total_rows": len(generations),
                "n_real_attempts": len(real_attempts),
                "n_unavailable_provider": len(unavailable),
                "n_real_successes": len(real_successes),
                "n_real_failures": len(real_failures),
                "failure_rate": failure_rate,
                "success_latency_ms": llm_latency.model_dump(),
                "note": (
                    "failure_rate excludes unavailable-provider calls from the denominator - "
                    "they were never a real attempt, so counting them as failures (or successes) "
                    "would misrepresent both numbers."
                ),
            },
        }

        print("=== System evaluation ===")
        print(f"voice_turns: n={len(voice_turns)} status_distribution={voice_status_counts} "
              f"completion_rate={completion_rate}")
        print(f"  total_latency p50={voice_latency.p50_ms} p95={voice_latency.p95_ms} "
              f"(n={voice_latency.sample_count} completed turns with recorded total_ms)")
        print(f"llm_generations: n_real_attempts={len(real_attempts)} "
              f"n_unavailable_provider={len(unavailable)} failure_rate={failure_rate}")
        print(f"  success_latency p50={llm_latency.p50_ms} p95={llm_latency.p95_ms} "
              f"(n={llm_latency.sample_count} real successes)")

        status = "completed" if (voice_turns or generations) else "unavailable"
        notes = None
        if status == "unavailable":
            notes = "No voice_turns or llm_generations rows exist yet - nothing to evaluate."

        run = await EvaluationRunRepository(session).create(
            evaluation_type="system",
            dataset_version=None,
            model_version=None,
            configuration={"source": "voice_turns + llm_generations (all real rows, all users)"},
            sample_count=len(voice_turns) + len(generations),
            status=status,
            metrics=metrics if status == "completed" else {},
            errors=[],
            notes=notes,
        )
        await session.commit()
        print(f"\nPersisted EvaluationRun {run.id} (status={status}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
