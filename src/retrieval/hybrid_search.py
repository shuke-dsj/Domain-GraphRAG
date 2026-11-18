"""混合检索引擎 —— 向量召回起点 + 图遍历路径"""


from __future__ import annotations
import logging
from typing import Optional

from src.storage.neo4j_client import Neo4jClient
from src.storage.vector_index import VectorIndexManager
from src.retrieval.atomic_tools import AtomicTools

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """混合检索：向量定位入口实体 → 图遍历获取上下文子图 → 文本化输出"""

    def __init__(
        self,
        client: Neo4jClient,
        vector_index: VectorIndexManager,
        default_hop_depth: int = 2,
        top_k_entities: int = 5,
    ):
        self.client = client
        self.vector_index = vector_index
        self.tools = AtomicTools(client)
        self.default_hop_depth = default_hop_depth
        self.top_k_entities = top_k_entities
        self.last_mode = "uninitialized"

    def search(self, query: str, hop_depth: int = None) -> list[dict]:
        """执行混合检索，返回结构化的路径短句列表

        返回格式: [{"path_text": "砂岩 -- 属于 --> 有效储层", "entities": [...], "score": 0.85}]
        """
        depth = hop_depth or self.default_hop_depth

        # Step 1: 向量召回入口实体
        entry_entities = self._find_entry_entities(query)
        if not entry_entities:
            logger.info("No entry entities found, fallback to keyword search")
            entry_entities = self._keyword_entry_entities(query)
            self.last_mode = "keyword_fallback"

        if not entry_entities:
            return []

        # Step 2: 对每个入口实体做图遍历
        all_paths = []
        seen = set()

        for entity in entry_entities[:self.top_k_entities]:
            entity_name = entity["name"]
            entity_score = entity.get("score", 0)

            subgraph = self.tools.get_n_hop_neighbors(
                entity_name, depth=depth, limit=30
            )

            for row in subgraph:
                node_chain = row.get("node_chain", [])
                edge_types = row.get("edge_types", [])
                descriptions = row.get("descriptions", [])

                directions = row.get("directions", [])
                path_text = self._path_to_text(node_chain, edge_types, directions)
                # The same physical path can be discovered from either endpoint.
                forward_key = (tuple(node_chain), tuple(edge_types), tuple(directions))
                reverse_key = (tuple(reversed(node_chain)), tuple(reversed(edge_types)), tuple(not d for d in reversed(directions)))
                path_key = min(forward_key, reverse_key)

                if path_key not in seen:
                    seen.add(path_key)
                    all_paths.append({
                        "path_text": path_text,
                        "entities": node_chain,
                        "edge_types": edge_types,
                        "directions": directions,
                        "evidence_ids": row.get("evidence_ids", []),
                        "descriptions": descriptions,
                        "entry_entity": entity_name,
                        "vector_score": entity_score,
                    })

        logger.info(f"Hybrid search: {len(entry_entities)} entry entities → {len(all_paths)} paths")
        return all_paths

    def search_with_context(self, query: str, hop_depth: int = None) -> str:
        """检索并返回拼接后的上下文字符串（可直接传给 LLM answer）"""
        paths = self.search(query, hop_depth)
        if not paths:
            return ""

        lines = []
        for p in paths:
            descs = " | ".join(d for d in p.get("descriptions", []) if d)
            lines.append(f"- {p['path_text']}" + (f" ({descs})" if descs else ""))

        return "\n".join(lines)

    # ── 入口实体查找 ──────────────────────────────────────

    def _find_entry_entities(self, query: str) -> list[dict]:
        """通过向量索引找到与 query 最相关的入口实体"""
        graph_name = self.client.graph_name
        # 优先用 Neo4j 原生向量索引
        results = self.vector_index.search_similar_entities(
            query, top_k=self.top_k_entities, graph_name=graph_name
        )
        self.last_mode = "neo4j_vector"
        if len(results) < self.top_k_entities:
            # The native index is shared by graphs and filters only after top-k.
            # Recompute within the current graph to avoid missing its candidates.
            results = self.vector_index.search_similar_entities_fallback(
                query, top_k=self.top_k_entities, graph_name=graph_name
            )
            self.last_mode = "memory_vector"
        return results

    def _keyword_entry_entities(self, query: str) -> list[dict]:
        """向量检索失败时的关键词回退"""
        results = self.tools.search_entity_by_name(query.split()[0] if query else "", limit=5)
        return [{"name": r["name"], "type": r.get("type", ""), "score": 0.5} for r in results]

    # ── 路径文本化 ──────────────────────────────────────────

    @staticmethod
    def _path_to_text(node_chain: list[str], edge_types: list[str], directions: list[bool] = None) -> str:
        """将图路径转为自然语言短句"""
        if not node_chain:
            return ""

        if len(node_chain) == 1:
            return node_chain[0]

        parts = [node_chain[0]]
        for i, edge in enumerate(edge_types):
            if i + 1 < len(node_chain):
                forward = directions is None or i >= len(directions) or directions[i]
                parts.append(f" -- {edge} --> " if forward else f" <-- {edge} -- ")
                parts.append(node_chain[i + 1])

        return "".join(parts)
