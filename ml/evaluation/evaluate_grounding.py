#!/usr/bin/env python3
"""Evaluates real grounding/citation behavior over every real LLM generation
attempt persisted so far (across all users - this audits the pipeline, not
one user's private data, same reasoning as the analytics dashboard's
model-version section).

Two metrics are genuinely measurable from data this codebase already
computes and persists (see services/llm/grounding.py::validate_citations,
called by rag_service.py for every real generation attempt):

  - citation_validity_rate: of every citation an LLM actually returned,
    what fraction pointed to a chunk_id that was genuinely part of that
    query's retrieved set. This is computed by the application at
    generation time (never self-reported by the model) and is aggregated
    here, not recomputed with different logic.
  - grounding_status_distribution: the real distribution of
    grounded/partially_grounded/ungrounded/unavailable across every real
    attempt.

"citation_validity_rate" and what the Phase 7 spec calls "retrieval-source
alignment" are, in this codebase, the exact same underlying check - a
citation is only ever considered valid if its chunk_id belongs to the
chunks actually retrieved for that query (see grounding.py). There is no
separate, independently-computed alignment signal to report without
duplicating that same logic under a different name, so this script reports
them as one metric and says so explicitly, rather than fabricating a second
distinct-sounding number from the same underlying check.

unsupported_answer_rate is marked unavailable: measuring whether an
answer's actual *content* (not just its citation IDs) is supported by the
cited text requires either a natural-language-inference/entailment model or
human judgment, and this codebase has neither. Rather than approximate it
with citation-ID validity (which would silently misrepresent what was
measured), this script reports it explicitly as unavailable.

Usage:
    python -m ml.evaluation.evaluate_grounding
"""
from __future__ import annotations

import argparse
import asyncio


async def run_evaluation(_args: argparse.Namespace) -> None:
    from sqlalchemy import select

    from voxmind.db.session import AsyncSessionLocal
    from voxmind.models.llm_generation import LlmGeneration
    from voxmind.repositories.evaluation_run_repository import EvaluationRunRepository

    async with AsyncSessionLocal() as session:
        stmt = select(LlmGeneration).where(LlmGeneration.model_name.isnot(None))
        rows = list((await session.execute(stmt)).scalars().all())

        if not rows:
            run = await EvaluationRunRepository(session).create(
                evaluation_type="grounding",
                dataset_version=None,
                model_version=None,
                configuration={"source": "llm_generations (all real attempts)"},
                sample_count=0,
                status="unavailable",
                metrics={},
                errors=[],
                notes=(
                    "No real LLM generation attempts exist yet (llm_generations.model_name is "
                    "null for every row) - there is nothing to evaluate. Run at least one real "
                    "/ask call with an LLM provider configured, then re-run this evaluation."
                ),
            )
            await session.commit()
            print(f"UNAVAILABLE - {run.notes}")
            return

        status_counts: dict[str, int] = {}
        total_citations_returned = 0
        total_citations_valid = 0
        rows_with_citations = 0
        for row in rows:
            status_counts[row.grounding_status] = status_counts.get(row.grounding_status, 0) + 1
            details = row.grounding_details or {}
            returned = details.get("citations_returned", 0)
            if returned:
                rows_with_citations += 1
                total_citations_returned += returned
                total_citations_valid += details.get("citations_valid", 0)

        metrics: dict = {
            "n_generations": len(rows),
            "grounding_status_distribution": status_counts,
            "citation_validity_rate": (
                {
                    "value": total_citations_valid / total_citations_returned,
                    "citations_returned": total_citations_returned,
                    "citations_valid": total_citations_valid,
                    "generations_with_citations": rows_with_citations,
                    "note": (
                        "Identical underlying check to 'retrieval-source alignment' in this "
                        "codebase - see module docstring."
                    ),
                }
                if total_citations_returned > 0
                else {
                    "status": "unavailable",
                    "reason": "No real generation has returned any citation yet.",
                }
            ),
            "unsupported_answer_rate": {
                "status": "unavailable",
                "reason": (
                    "Requires an entailment/NLI model or human judgment of whether the answer's "
                    "content (not just its citation IDs) is supported by the cited text - neither "
                    "exists in this codebase. Not approximated by citation-ID validity."
                ),
            },
        }

        print("=== Grounding evaluation ===")
        print(f"n_generations={len(rows)} grounding_status_distribution={status_counts}")
        if total_citations_returned > 0:
            print(
                f"citation_validity_rate={total_citations_valid / total_citations_returned:.4f} "
                f"({total_citations_valid}/{total_citations_returned} citations, "
                f"{rows_with_citations} generation(s) with citations)"
            )
        else:
            print("citation_validity_rate=UNAVAILABLE (no real generation has returned a citation yet)")
        print("unsupported_answer_rate=UNAVAILABLE (no entailment model or human judgment available)")

        run = await EvaluationRunRepository(session).create(
            evaluation_type="grounding",
            dataset_version=None,
            model_version=None,
            configuration={"source": "llm_generations (all real attempts, all users)"},
            sample_count=len(rows),
            status="completed",
            metrics=metrics,
            errors=[],
            notes=None,
        )
        await session.commit()
        print(f"\nPersisted EvaluationRun {run.id} (status=completed).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
