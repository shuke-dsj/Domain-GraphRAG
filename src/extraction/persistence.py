"""实体提取结果的 JSON 持久化"""


from __future__ import annotations
import json
from pathlib import Path
from src.extraction.models import ExtractionResult


def save_extraction_results(
    results: list[ExtractionResult],
    output_dir: str | Path,
):
    """将提取结果保存为 entities.json、relations.json、chunks.json"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entities = []
    all_relations = []
    all_chunks = []

    for r in results:
        for e in r.entities:
            all_entities.append(e.model_dump())
        for rel in r.relations:
            all_relations.append(rel.model_dump())
        all_chunks.append({
            "chunk_id": r.chunk_id,
            "text": r.chunk_text,
        })

    with open(output_dir / "entities.json", "w", encoding="utf-8") as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=2)

    with open(output_dir / "relations.json", "w", encoding="utf-8") as f:
        json.dump(all_relations, f, ensure_ascii=False, indent=2)

    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)


def load_extraction_results(
    input_dir: str | Path,
) -> list[dict]:
    """加载已保存的提取结果"""
    input_dir = Path(input_dir)
    results = {"entities": [], "relations": [], "chunks": []}
    for key in results:
        path = input_dir / f"{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                results[key] = json.load(f)
    return results
