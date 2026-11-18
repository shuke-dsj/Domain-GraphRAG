"""Neo4j Vector Index 管理 —— 为实体描述字段建立向量索引"""


from __future__ import annotations
import logging
import numpy as np
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from src.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Neo4j 5.x 向量索引默认维度（取决于 embedding 模型）
DEFAULT_EMBEDDING_DIM = 384   # paraphrase-multilingual-MiniLM-L12-v2 输出维度
INDEX_NAME = "entity_description_vector"


class VectorIndexManager:
    """管理 Neo4j 向量索引：创建、更新、查询

    嵌入模型延迟加载 —— 只在首次查询或 embed_and_update_entities 时下载。
    """

    def __init__(
        self,
        client: Neo4jClient,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    ):
        self.client = client
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dim = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self):
        if self._dim is None:
            getter = getattr(self.model, "get_embedding_dimension", None)
            self._dim = getter() if getter else self.model.get_sentence_embedding_dimension()
        return self._dim

    def create_index(self):
        """创建 Neo4j 向量索引（需要 Neo4j 5.x Enterprise）"""
        cypher = f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        try:
            self.client.run(cypher)
            logger.info(f"Vector index '{INDEX_NAME}' created (dim={self.dim})")
        except Exception as e:
            logger.warning(f"Failed to create vector index (may need Neo4j Enterprise): {e}")

    def embed_and_update_entities(self):
        """为所有没有 embedding 的实体计算并写入向量"""
        graph_name = self.client.graph_name
        rows = self.client.run(
            "MATCH (e:Entity {graph: $graph}) "
            "WHERE e.embedding IS NULL OR coalesce(e.embedding_model, '') <> $model "
            "RETURN e.name AS name, e.type AS type, e.description AS description",
            {"graph": graph_name, "model": self.model_name},
        )

        if not rows:
            logger.info("No entities need embedding update")
            return

        # description 为空时用 name，保证每个实体都有可搜索的向量
        texts = [(r["description"] or r["name"]) for r in rows]
        names = [r["name"] for r in rows]
        logger.info(f"Computing embeddings for {len(names)} entities...")
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        cypher = """
        UNWIND $batch AS item
        MATCH (e:Entity {name: item.name, graph: $graph})
        SET e.embedding = item.embedding, e.embedding_model = $model
        """
        batch = [
            {"name": name, "embedding": emb.tolist()}
            for name, emb in zip(names, embeddings)
        ]
        self.client.execute_write(cypher, {"batch": batch, "graph": graph_name, "model": self.model_name})
        logger.info(f"Updated embeddings for {len(batch)} entities in graph '{graph_name}'")

    def search_similar_entities(
        self,
        query_text: str,
        top_k: int = 10,
        min_score: float = 0.0,
        graph_name: str = "default",
    ) -> list[dict]:
        """向量相似度检索实体 —— 作为图谱遍历的起点"""
        query_embedding = self.model.encode(
            [query_text], normalize_embeddings=True
        )[0].tolist()

        cypher = f"""
        CALL db.index.vector.queryNodes('{INDEX_NAME}', $k, $embedding)
        YIELD node, score
        WHERE score >= $min_score AND node.graph = $graph
        RETURN node.name AS name, node.type AS type, node.description AS description, score
        ORDER BY score DESC
        """
        try:
            return self.client.run(
                cypher,
                {"embedding": query_embedding, "k": top_k, "min_score": min_score, "graph": graph_name},
            )
        except Exception as e:
            logger.warning(f"Vector search failed (index may not exist): {e}")
            return []

    def search_similar_entities_fallback(
        self,
        query_text: str,
        top_k: int = 10,
        graph_name: str = "default",
    ) -> list[dict]:
        """向量检索的回退方案：内存计算余弦相似度（无需 Neo4j 向量索引）"""
        query_embedding = self.model.encode(
            [query_text], normalize_embeddings=True
        )[0]

        rows = self.client.run(
            "MATCH (e:Entity {graph: $graph}) WHERE e.embedding IS NOT NULL "
            "RETURN e.name AS name, e.type AS type, e.description AS description, e.embedding AS embedding",
            {"graph": graph_name},
        )

        if not rows:
            return []

        scored = []
        for r in rows:
            emb = np.array(r["embedding"])
            score = float(np.dot(query_embedding, emb))  # cosine with normalized vectors
            scored.append({**r, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
