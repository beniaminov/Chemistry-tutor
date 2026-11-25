#!/usr/bin/env python
"""Evaluate dense, BM25 and hybrid retrieval on the versioned control set."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.local_rag as rag  # noqa: E402


MODES = ("dense", "bm25", "hybrid", "hybrid_weighted")
MODE_LABELS = {
    "dense": "dense (E5)",
    "bm25": "BM25",
    "hybrid": "hybrid RRF 1:1",
    "hybrid_weighted": "hybrid RRF dense×4 (опыт)",
}
WEIGHTED_DENSE = 4.0
FORMULA_TYPES = {"formula", "stoichiometry", "typo"}
DEFAULT_QUESTIONS = ROOT / "tests" / "data" / "rag_control_questions.json"
DEFAULT_REPORT = ROOT / "docs" / "Результаты-сравнения-поиска.md"
DEFAULT_BASELINE = ROOT / "tests" / "data" / "retrieval_baseline.json"


def file_sha256(path: Path) -> str:
    return rag.content_hash(path)


def load_questions(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    if not 30 <= len(questions) <= 50:
        raise ValueError(f"Expected 30-50 questions, found {len(questions)}.")
    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question ids must be unique.")
    counts = Counter(question["type"] for question in questions)
    if counts["no_answer"] < 5:
        raise ValueError("The control set must contain at least 5 no-answer cases.")
    if counts["formula"] < 5:
        raise ValueError("The control set must contain at least 5 formula cases.")
    return data


def resolve_expected_chunks(question: dict, payloads: list[dict]) -> list[str]:
    pages = set(question["expected_pages"])
    if not pages:
        return []
    terms = [term.lower() for term in question["evidence_terms"]]
    if not terms:
        raise ValueError(
            f"{question['id']}: expected_pages given but evidence_terms is empty."
        )
    chunks = [
        payload["chunk_id"]
        for payload in payloads
        if payload["page"] in pages
        and any(term in payload["text"].lower() for term in terms)
    ]
    if not chunks:
        raise ValueError(
            f"{question['id']}: no evidence chunk found on pages {sorted(pages)} "
            f"for terms {question['evidence_terms']}."
        )
    return sorted(chunks)


def first_relevant_rank(
    ranking: list[dict],
    expected_pages: set[int],
    limit: int,
) -> int | None:
    for rank, payload in enumerate(ranking[:limit], start=1):
        if payload["page"] in expected_pages:
            return rank
    return None


def formula_hit(question: dict, ranking: list[dict]) -> tuple[bool, bool]:
    expected_pages = set(question["expected_pages"])
    patterns = question["required_terms"]
    if not patterns:
        return False, False
    top5 = ranking[:5]
    hit5 = False
    for payload in top5:
        if payload["page"] not in expected_pages:
            continue
        text = payload["text"]
        if all(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            hit5 = True
            break
    top1 = False
    if ranking and ranking[0]["page"] in expected_pages:
        top1 = all(
            re.search(pattern, ranking[0]["text"], re.IGNORECASE)
            for pattern in patterns
        )
    return hit5, top1


def evaluate(
    dataset: dict,
    payloads: list[dict],
    rankings: dict[str, dict[str, list[dict]]],
) -> dict:
    scored = [
        question
        for question in dataset["questions"]
        if question["retrieval_scored"] and question["expected_pages"]
    ]
    formula_questions = [
        question
        for question in dataset["questions"]
        if question["required_terms"]
        and question["retrieval_scored"]
        and question["expected_pages"]
    ]

    overall = {}
    per_type = {}
    failures = []
    for mode in MODES:
        hits = 0
        reciprocal_ranks = 0.0
        type_stats = defaultdict(lambda: {"count": 0, "hits": 0, "rr": 0.0})
        mode_failures = []
        for question in scored:
            ranking = rankings[question["id"]][mode]
            expected_pages = set(question["expected_pages"])
            rank5 = first_relevant_rank(ranking, expected_pages, 5)
            rank10 = first_relevant_rank(ranking, expected_pages, 10)
            if rank5 is not None:
                hits += 1
            if rank10 is not None:
                reciprocal_ranks += 1 / rank10
            stats = type_stats[question["type"]]
            stats["count"] += 1
            if rank5 is not None:
                stats["hits"] += 1
            if rank10 is not None:
                stats["rr"] += 1 / rank10
            if rank5 is None:
                mode_failures.append(
                    {
                        "id": question["id"],
                        "type": question["type"],
                        "pages": sorted(expected_pages),
                        "top5": [
                            {"page": item["page"], "chunk_id": item["chunk_id"][:12]}
                            for item in ranking[:5]
                        ],
                    }
                )
        total = len(scored)
        overall[mode] = {
            "questions": total,
            "hit_rate_at_5": hits / total,
            "mrr_at_10": reciprocal_ranks / total,
        }
        per_type[mode] = {
            question_type: {
                "count": stats["count"],
                "hit_rate_at_5": stats["hits"] / stats["count"],
                "mrr_at_10": stats["rr"] / stats["count"],
            }
            for question_type, stats in sorted(type_stats.items())
        }
        if mode == "hybrid":
            failures = mode_failures

    formula = {}
    for mode in MODES:
        hits5 = 0
        hits1 = 0
        for question in formula_questions:
            hit5, top1 = formula_hit(question, rankings[question["id"]][mode])
            hits5 += int(hit5)
            hits1 += int(top1)
        total = len(formula_questions)
        formula[mode] = {
            "questions": total,
            "formula_hit_at_5": hits5 / total,
            "formula_top1": hits1 / total,
        }

    controls = []
    for question in dataset["questions"]:
        if question["retrieval_scored"]:
            continue
        ranking = rankings[question["id"]]["hybrid"]
        controls.append(
            {
                "id": question["id"],
                "type": question["type"],
                "question": question["question"],
                "expected_behavior": question["expected_behavior"],
                "top1_page": ranking[0]["page"] if ranking else None,
                "top1_score": ranking[0].get("score") if ranking else None,
                "top3_pages": [item["page"] for item in ranking[:3]],
            }
        )

    return {
        "overall": overall,
        "per_type": per_type,
        "formula": formula,
        "failures": failures,
        "controls": controls,
        "scored_questions": len(scored),
        "formula_questions": len(formula_questions),
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_report(
    dataset: dict,
    results: dict,
    baseline: dict,
    dataset_path: Path,
    dataset_sha: str,
    source_sha: str,
    point_count: int,
    elapsed: float,
) -> str:
    lines = [
        "# Сравнение поиска на контрольном наборе (этап 0)",
        "",
        f"Дата прогона: {baseline['created']}.",
        "",
        "## Конфигурация",
        "",
        f"- Набор: `{dataset_path.relative_to(ROOT)}` (SHA-256 `{dataset_sha[:16]}…`, "
        f"{len(dataset['questions'])} вопросов, из них {results['scored_questions']} "
        "оцениваются по retrieval).",
        f"- Источник: `{rag.SOURCE_PDF.relative_to(ROOT)}` "
        f"(SHA-256 `{source_sha[:16]}…`).",
        f"- Коллекция Qdrant: `{rag.COLLECTION}`, точек: {point_count}.",
        f"- Embeddings: `{rag.EMBEDDING_MODEL}` rev. `{rag.EMBEDDING_REVISION[:12]}`, "
        f"{rag.EMBEDDING_SIZE} измерений, cosine.",
        f"- Поиск: dense top-30, BM25 top-30, RRF (k={rag.RRF_K}); метрики "
        "HitRate@5 и MRR@10 считаются по странице PDF.",
        f"- Время прогона: {elapsed:.1f} с.",
        "",
        "## Итоговые метрики",
        "",
        "| Метод | HitRate@5 | MRR@10 | Формулы: hit@5 | Формулы: top-1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        overall = results["overall"][mode]
        formula = results["formula"][mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {pct(overall['hit_rate_at_5'])} | "
            f"{overall['mrr_at_10']:.3f} | {pct(formula['formula_hit_at_5'])} | "
            f"{pct(formula['formula_top1'])} |"
        )

    lines.extend(
        [
            "",
            f"Формульные вопросы: {results['formula_questions']} "
            "(типы `formula`, `stoichiometry`, `typo` с непустым `required_terms`). "
            "Для зачёта формулы в top-5 нужен фрагмент с ожидаемой страницы, "
            "содержащий все regex-группы из набора.",
            "",
            "Строка `hybrid RRF dense×4` — диагностический опыт: вес dense-списка "
            "увеличен в 4 раза, чтобы показать вклад слабого BM25 в 1:1-фьюжн. "
            "Продакшн-путь пока использует 1:1; выбор веса — решение этапа 1.",
            "",
            "## Разбивка по типам (hybrid)",
            "",
            "| Тип | Вопросов | HitRate@5 | MRR@10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for question_type, stats in results["per_type"]["hybrid"].items():
        lines.append(
            f"| {question_type} | {stats['count']} | "
            f"{pct(stats['hit_rate_at_5'])} | {stats['mrr_at_10']:.3f} |"
        )

    lines.extend(["", "## Промахи hybrid в top-5", ""])
    if results["failures"]:
        lines.extend(
            [
                "| ID | Тип | Ожидаемые страницы | Top-5 (страницы) |",
                "|---|---|---|---|",
            ]
        )
        for failure in results["failures"]:
            top5 = ", ".join(str(item["page"]) for item in failure["top5"])
            lines.append(
                f"| {failure['id']} | {failure['type']} | "
                f"{', '.join(map(str, failure['pages']))} | {top5} |"
            )
    else:
        lines.append("Промахов нет.")

    lines.extend(
        [
            "",
            "## Контроль отказов и инъекций",
            "",
            "Эти вопросы не входят в HitRate/MRR: для `no_answer` нет эталонной "
            "страницы, для `injection` оценивается поведение генерации на этапе 3. "
            "Таблица показывает, что retrieval всё равно возвращает.",
            "",
            "| ID | Тип | Ожидаемое поведение | Top-1 страница | Top-3 страницы |",
            "|---|---|---|---:|---|",
        ]
    )
    for control in results["controls"]:
        lines.append(
            f"| {control['id']} | {control['type']} | "
            f"`{control['expected_behavior']}` | {control['top1_page']} | "
            f"{', '.join(map(str, control['top3_pages']))} |"
        )

    lines.extend(
        [
            "",
            "## Baseline и ограничения",
            "",
            f"- Baseline сохранён в `{DEFAULT_BASELINE.relative_to(ROOT)}`.",
            "- Оценка page-level: попадание любого фрагмента с ожидаемой страницы "
            "считается релевантным. Chunk ID разрешаются по `evidence_terms` и "
            "используются только для проверки, что опора существует в корпусе.",
            "- Формулы проверяются regex-вариантами с учётом типичных ошибок OCR; "
            "это не заменяет ручную сверку по изображению страницы.",
            "- Набор проверяет retrieval, а не качество ответа Qwen. Отказы, "
            "цитирование и устойчивость к инъекциям оцениваются на этапе 3.",
            "- Онтологические URI в payload пока пусты, поэтому вопросы типа "
            "`ontology` проверяют только текстовую опору; связь с RDF — этап 4.",
            "- Контрольный набор составлен по OCR-тексту и сверен с оглавлением "
            "и выборочными страницами PDF; перед изменением PDF его нужно "
            "перепроверить.",
            "",
            "## Воспроизведение",
            "",
            "```powershell",
            ".\\.venv-e5\\Scripts\\python.exe scripts\\eval_retrieval.py",
            "```",
            "",
            "Перезаписать baseline после осознанного изменения индекса или набора:",
            "",
            "```powershell",
            ".\\.venv-e5\\Scripts\\python.exe scripts\\eval_retrieval.py --write-baseline",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_with_baseline(results: dict, baseline: dict) -> list[str]:
    problems = []
    previous_overall = baseline.get("metrics", {}).get("overall", {})
    for mode in MODES:
        if mode not in previous_overall:
            continue
        for metric in ("hit_rate_at_5", "mrr_at_10"):
            current = results["overall"][mode][metric]
            previous = previous_overall[mode][metric]
            if current < previous - 0.05:
                problems.append(
                    f"{mode}.{metric}: {previous:.3f} -> {current:.3f} "
                    "(падение больше 0.05)"
                )
    return problems


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Replace the stored retrieval baseline with this run",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Fail if a metric drops by more than 0.05 versus the stored baseline",
    )
    args = parser.parse_args()

    dataset_path = args.questions.resolve()
    dataset = load_questions(dataset_path)
    dataset_sha = file_sha256(dataset_path)
    source_sha = file_sha256(rag.SOURCE_PDF)

    client = rag.QdrantClient(path=str(rag.QDRANT_PATH))
    try:
        if not client.collection_exists(rag.COLLECTION):
            raise RuntimeError("Local Qdrant collection is missing; run index first.")
        payloads = rag.all_payloads(client)
        if not payloads:
            raise RuntimeError("The local Qdrant collection is empty.")
        point_count = client.count(rag.COLLECTION, exact=True).count

        expected_chunks = {}
        for question in dataset["questions"]:
            expected_chunks[question["id"]] = resolve_expected_chunks(
                question, payloads
            )

        model = rag.load_embeddings()
        model.max_seq_length = 512
        started = time.perf_counter()
        rankings = {}
        for question in dataset["questions"]:
            dense = rag.dense_candidates(
                client, question["question"], model, payloads
            )
            lexical_order = rag.bm25_candidates(question["question"], payloads)
            rankings[question["id"]] = {
                "dense": [point.payload for point in dense],
                "bm25": [dict(payloads[index]) for index in lexical_order],
                "hybrid": rag.fuse_rrf(dense, payloads, lexical_order),
                "hybrid_weighted": rag.fuse_rrf(
                    dense,
                    payloads,
                    lexical_order,
                    dense_weight=WEIGHTED_DENSE,
                ),
            }
        elapsed = time.perf_counter() - started
    finally:
        client.close()

    results = evaluate(dataset, payloads, rankings)
    created = date.today().isoformat()
    baseline = {
        "schema_version": 1,
        "created": created,
        "question_set": {
            "path": dataset_path.relative_to(ROOT).as_posix(),
            "sha256": dataset_sha,
            "count": len(dataset["questions"]),
            "scored": results["scored_questions"],
            "formula_questions": results["formula_questions"],
            "types": dict(
                sorted(Counter(q["type"] for q in dataset["questions"]).items())
            ),
        },
        "source": {
            "path": rag.SOURCE_PDF.relative_to(ROOT).as_posix(),
            "sha256": source_sha,
            "document_id": rag.DOC_ID,
        },
        "index": {
            "collection": rag.COLLECTION,
            "points": point_count,
            "embedding_model": rag.EMBEDDING_MODEL,
            "embedding_revision": rag.EMBEDDING_REVISION,
            "embedding_dimension": rag.EMBEDDING_SIZE,
            "distance": "Cosine",
            "rrf_k": rag.RRF_K,
            "candidate_limit": 30,
        },
        "metrics": {
            "overall": results["overall"],
            "per_type": results["per_type"],
            "formula": results["formula"],
        },
        "expected_chunks": expected_chunks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    if args.check_baseline and args.baseline.exists():
        previous = json.loads(args.baseline.read_text(encoding="utf-8"))
        problems = compare_with_baseline(results, previous)
        if problems:
            print("Baseline check failed:")
            for problem in problems:
                print(f"- {problem}")
            raise SystemExit(1)
        print("Baseline check passed.")

    args.report.write_text(
        build_report(
            dataset,
            results,
            baseline,
            dataset_path,
            dataset_sha,
            source_sha,
            point_count,
            elapsed,
        ),
        encoding="utf-8",
    )
    if args.write_baseline:
        args.baseline.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Baseline written: {args.baseline}")

    print(f"Report written: {args.report}")
    for mode in MODES:
        overall = results["overall"][mode]
        formula = results["formula"][mode]
        print(
            f"{mode}: HitRate@5={pct(overall['hit_rate_at_5'])}, "
            f"MRR@10={overall['mrr_at_10']:.3f}, "
            f"formula@5={pct(formula['formula_hit_at_5'])}"
        )


if __name__ == "__main__":
    main()
