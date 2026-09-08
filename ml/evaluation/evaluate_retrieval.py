#!/usr/bin/env python3
"""Evaluates the real retrieval pipeline (query -> embedding -> pgvector
search -> lexical search -> RRF fusion -> optional cross-encoder rerank)
against a small, hand-authored relevance-judgment dataset - see
ml/evaluation/retrieval_datasets/retrieval_eval_v1.json for the dataset
itself and exactly how its judgments were produced.

IMPORTANT (see docs/evaluation.md): Recall@K/Precision@K/MRR/nDCG require
genuine ground-truth relevance judgments. This script never treats "the
chunk the system returned" as automatically relevant - relevance for every
query is fixed in the dataset file, authored by reading real chunk content
before this evaluation ever ran.

This script performs real, end-to-end side effects against the configured
database: it creates (or reuses) a dedicated evaluation user and a fresh
evaluation conversation, uploads and processes the fixture documents
through the real KnowledgeService, and runs the real RetrievalService for
every query - the exact same code path production `/ask` calls use. It
does not fabricate or mock any part of the retrieval pipeline.

Usage:
    python -m ml.evaluation.evaluate_retrieval \\
        --dataset ml/evaluation/retrieval_datasets/retrieval_eval_v1.json \\
        --k 1 3 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from ml.evaluation.retrieval_metrics import QueryResult, compute_retrieval_metrics

EVAL_USER_EMAIL = "eval-retrieval@voxmind.internal"


async def run_evaluation(args: argparse.Namespace) -> None:
    from voxmind.core.config import get_settings
    from voxmind.core.security import hash_password
    from voxmind.db.session import AsyncSessionLocal
    from voxmind.repositories.conversation_repository import ConversationRepository
    from voxmind.repositories.evaluation_run_repository import EvaluationRunRepository
    from voxmind.repositories.message_repository import MessageRepository
    from voxmind.repositories.retrieval_result_repository import RetrievalResultRepository
    from voxmind.repositories.user_repository import UserRepository
    from voxmind.services.knowledge.embedding_provider import HuggingFaceEmbeddingProvider
    from voxmind.services.knowledge_service import KnowledgeService
    from voxmind.services.retrieval.reranker import CrossEncoderReranker
    from voxmind.services.retrieval_service import RetrievalService
    from voxmind.services.storage.factory import build_storage_backend
    from voxmind.workers.task_runner import build_task_runner

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text())
    dataset_dir = dataset_path.parent

    settings = get_settings()

    async with AsyncSessionLocal() as session:
        users = UserRepository(session)
        conversations = ConversationRepository(session)
        messages = MessageRepository(session)
        run_repo = EvaluationRunRepository(session)

        user = await users.get_by_email(EVAL_USER_EMAIL)
        if user is None:
            user = await users.create(email=EVAL_USER_EMAIL, hashed_password=hash_password(uuid.uuid4().hex))
            await session.commit()

        conversation = await conversations.create(user_id=user.id, title=f"retrieval-eval {dataset['version']}")
        await session.commit()

        storage = build_storage_backend(settings)
        task_runner = build_task_runner(settings.TASK_RUNNER)
        embedding_provider = HuggingFaceEmbeddingProvider(settings)
        reranker = CrossEncoderReranker(settings)
        knowledge = KnowledgeService(
            session, settings=settings, storage=storage, task_runner=task_runner,
            embedding_provider=embedding_provider,
        )
        retrieval = RetrievalService(
            session, settings=settings, task_runner=task_runner,
            embedding_provider=embedding_provider, reranker=reranker,
        )

        source_to_document_id: dict[str, uuid.UUID] = {}
        errors: list[dict] = []
        for doc_spec in dataset["documents"]:
            raw_bytes = (dataset_dir / doc_spec["path"]).read_bytes()
            document = await knowledge.upload(
                conversation_id=conversation.id, user_id=user.id,
                filename=f"{doc_spec['source']}.md", content_type="text/markdown", raw_bytes=raw_bytes,
            )
            document = await knowledge.process(
                conversation_id=conversation.id, user_id=user.id, document_id=document.id
            )
            if document.status != "completed":
                errors.append({"source": doc_spec["source"], "reason": document.error_message})
                continue
            source_to_document_id[doc_spec["source"]] = document.id
            print(f"ingested {doc_spec['source']}: {document.chunk_count} chunk(s)")

        if errors:
            print(f"error: {len(errors)} document(s) failed to ingest: {errors}", file=sys.stderr)
            sys.exit(1)

        document_id_to_source = {v: k for k, v in source_to_document_id.items()}

        query_results: list[QueryResult] = []
        per_query_trace: list[dict] = []
        for q in dataset["queries"]:
            relevant_set = {(j["source"], j["chunk_index"]) for j in q["relevant_chunks"]}
            user_turn = await messages.create(session_id=conversation.id, role="user", content=q["query"])
            await session.commit()
            _selected, retrieval_result_id = await retrieval.retrieve(
                conversation_id=conversation.id, query_message_id=user_turn.id, query=q["query"]
            )
            trace_row = await RetrievalResultRepository(session).get_by_id(retrieval_result_id)
            # `retrieval.retrieve()` just created this row in this same
            # transaction, a few lines above - a real None here would mean
            # the retrieval result vanished immediately after being
            # written, a genuine bug worth failing loudly on, not an
            # Optional case worth silently chaining around.
            assert trace_row is not None, f"retrieval_result {retrieval_result_id} was just created but is not found"
            ranked_items = sorted(trace_row.items, key=lambda item: item["hybrid_rank"])

            ranked_relevant: list[bool] = []
            for item in ranked_items:
                source = document_id_to_source.get(uuid.UUID(item["document_id"]))
                ranked_relevant.append((source, item["chunk_index"]) in relevant_set)

            query_results.append(
                QueryResult(query=q["query"], ranked_relevant=ranked_relevant, total_relevant=len(relevant_set))
            )
            per_query_trace.append(
                {
                    "query": q["query"],
                    "reranked": trace_row.reranked,
                    "ranked_sources": [
                        {"document_id": item["document_id"], "chunk_index": item["chunk_index"]}
                        for item in ranked_items
                    ],
                }
            )
            print(f"query={q['query']!r}: {sum(ranked_relevant)} relevant hit(s) in {len(ranked_relevant)} ranked candidates")

        metrics = compute_retrieval_metrics(query_results, k_values=args.k)

        print(f"\n=== Retrieval evaluation: {dataset['version']} ===")
        # `settings.RERANK_ENABLED` (not the last loop iteration's
        # `trace_row`, which mypy correctly can't assume is still bound/
        # non-None here if the dataset had zero queries) is the real
        # authoritative source for this - reranking is a pipeline-wide
        # setting, not something that varies per query, and this exact
        # value is already what's persisted in `run_repo.create(...)`'s
        # `model_version` string below.
        print(f"n_queries={metrics.n_queries} mrr={metrics.mrr:.4f} reranked={settings.RERANK_ENABLED}")
        for k in args.k:
            print(f"  recall@{k}={metrics.recall_at_k[k]:.4f} precision@{k}={metrics.precision_at_k[k]:.4f} ndcg@{k}={metrics.ndcg_at_k[k]:.4f}")

        run = await run_repo.create(
            evaluation_type="retrieval",
            dataset_version=dataset["version"],
            model_version=f"{embedding_provider.model_id}+rerank={settings.RERANK_ENABLED}",
            configuration={
                "k_values": args.k,
                "documents": [d["source"] for d in dataset["documents"]],
                "chunking_config": dataset["chunking_config"],
                "retrieval_top_k": settings.RETRIEVAL_TOP_K,
                "retrieval_candidate_k": settings.RETRIEVAL_CANDIDATE_K,
                "hybrid_rrf_k": settings.HYBRID_RRF_K,
                "rerank_enabled": settings.RERANK_ENABLED,
            },
            sample_count=metrics.n_queries,
            status="completed",
            metrics=metrics.to_dict(),
            errors=[],
            notes=(
                "Small, hand-authored illustrative retrieval set (see dataset description). "
                "Not a claim of production-scale statistical significance."
            ),
        )
        await session.commit()
        print(f"\nPersisted EvaluationRun {run.id} (status=completed).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="ml/evaluation/retrieval_datasets/retrieval_eval_v1.json")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
