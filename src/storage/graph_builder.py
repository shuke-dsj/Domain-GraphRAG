"""图谱构建器 —— 从实体/关系 JSON 写入 Neo4j"""


from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

from src.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphBuilder:
    """将提取的实体和关系批量写入 Neo4j 图谱"""

    # 写入批大小
    BATCH_SIZE = 200

    def __init__(self, client: Neo4jClient):
        self.client = client
        self.graph_name = client.graph_name

    # ── 从 JSON 文件构建 ──────────────────────────────────

    def build_from_json(
        self,
        entities_path: str | Path,
        relations_path: str | Path,
        chunks_path: str | Path = None,
        clear_first: bool = False,
    ):
        """从提取结果 JSON 文件构建 Neo4j 图谱"""
        if clear_first:
            self.client.clear_graph()

        self.client.init_schema()

        entities = self._load_json(entities_path)
        relations = self._load_json(relations_path)
        chunks = self._load_json(chunks_path) if chunks_path else []

        logger.info(f"Building graph: {len(entities)} entities, {len(relations)} relations, {len(chunks)} chunks")

        self._write_chunks(chunks)
        self._write_entities(entities)
        self._write_relations(relations)

        logger.info("Graph build complete")

    # ── 写入节点 ───────────────────────────────────────────

    def _write_chunks(self, chunks: list[dict]):
        cypher = """
        UNWIND $batch AS item
        MERGE (c:Chunk {chunk_id: item.chunk_id, graph: $graph})
        SET c.text = item.text,
            c.source_doc = item.source_doc,
            c.document_id = item.document_id,
            c.position = item.position,
            c.graph = $graph
        """
        for batch in self._batches(chunks):
            self.client.execute_write(cypher, {"batch": batch, "graph": self.graph_name})

    def _write_entities(self, entities: list[dict]):
        cypher = """
        UNWIND $batch AS item
        MERGE (e:Entity {name: item.name, graph: $graph})
        SET e.embedding = CASE WHEN coalesce(e.description, '') <> coalesce(item.description, '') THEN null ELSE e.embedding END,
            e.type = item.type,
            e.description = item.description,
            e.aliases = item.aliases,
            e.graph = $graph
        """
        for batch in self._batches(entities):
            self.client.execute_write(cypher, {"batch": batch, "graph": self.graph_name})

        # 建立实体 → Chunk 的 APPEARS_IN 边
        edge_cypher = """
        UNWIND $batch AS item
        MATCH (e:Entity {name: item.name, graph: $graph})
        MATCH (c:Chunk {chunk_id: item.chunk_id, graph: $graph})
        MERGE (e)-[:APPEARS_IN]->(c)
        """
        items_with_chunk = [e for e in entities if e.get("chunk_id")]
        for batch in self._batches(items_with_chunk):
            self.client.execute_write(edge_cypher, {"batch": batch, "graph": self.graph_name})

    def _write_relations(self, relations: list[dict]):
        cypher = """
        UNWIND $batch AS item
        MATCH (src:Entity {name: item.source, graph: $graph})
        MATCH (tgt:Entity {name: item.target, graph: $graph})
        MERGE (src)-[r:RELATES {type: item.relation_type}]->(tgt)
        SET r.evidence = item.evidence,
            r.chunk_ids = CASE WHEN item.chunk_id IN coalesce(r.chunk_ids, []) THEN r.chunk_ids
                               ELSE coalesce(r.chunk_ids, []) + item.chunk_id END
        """
        for batch in self._batches(relations):
            self.client.execute_write(cypher, {"batch": batch, "graph": self.graph_name})

    # ── 实体消歧后的合并操作 ──────────────────────────────

    def merge_aliases(self, canonical: str, aliases: list[str]):
        """将别名实体合并到标准实体（同一 graph 内）"""
        raise RuntimeError("自动合并已禁用：需要先实现来源及关系属性的安全迁移")
        cypher = """
        MATCH (canon:Entity {name: $canonical, graph: $graph})
        UNWIND $aliases AS alias
        MATCH (alias_node:Entity {name: alias, graph: $graph})
        OPTIONAL MATCH (alias_node)-[r_in]->(other)
        OPTIONAL MATCH (other2)-[r_out]->(alias_node)
        FOREACH (_ IN CASE WHEN r_in IS NOT NULL THEN [1] ELSE [] END |
            MERGE (canon)-[:RELATES {type: type(r_in)}]->(other)
        )
        FOREACH (_ IN CASE WHEN r_out IS NOT NULL THEN [1] ELSE [] END |
            MERGE (other2)-[:RELATES {type: type(r_out)}]->(canon)
        )
        DETACH DELETE alias_node
        """
        self.client.execute_write(cypher, {"canonical": canonical, "aliases": aliases, "graph": self.graph_name})

    def create_is_a_relation(self, child: str, parent: str):
        """创建上下位关系（同一 graph 内）"""
        cypher = """
        MATCH (child:Entity {name: $child, graph: $graph})
        MATCH (parent:Entity {name: $parent, graph: $graph})
        MERGE (child)-[:IS_A]->(parent)
        """
        self.client.execute_write(cypher, {"child": child, "parent": parent, "graph": self.graph_name})

    # ── 辅助 ────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: str | Path) -> list[dict]:
        p = Path(path)
        if not p.exists():
            logger.warning(f"File not found: {p}")
            return []
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _batches(items: list, size: int = None) -> list[list]:
        size = size or GraphBuilder.BATCH_SIZE
        for i in range(0, len(items), size):
            yield items[i:i + size]
