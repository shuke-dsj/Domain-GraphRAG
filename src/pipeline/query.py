"""查询管线 —— query → 向量召回 → 图遍历 → 重排序 → LLM 生成答案"""

import logging

from src.config.settings import Settings
from src.storage.neo4j_client import Neo4jClient
from src.storage.vector_index import VectorIndexManager
from src.retrieval.atomic_tools import AtomicTools
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class QueryPipeline:
    """编排完整的查询流程"""

    def __init__(self, settings: Settings, llm):
        self.settings = settings
        self.llm = llm

        neo4j_config = settings.neo4j
        self.client = Neo4jClient(
            uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            user=neo4j_config.get("user", "neo4j"),
            password=neo4j_config.get("password", ""),
            database=neo4j_config.get("database", "neo4j"),
            graph_name=neo4j_config.get("graph", "default"),
        )

        self.vector_index = VectorIndexManager(
            self.client,
            model_name=settings.embedding.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
            device=settings.embedding.get("device", "cpu"),
        )

        self.hybrid_search = HybridSearchEngine(
            self.client,
            self.vector_index,
            default_hop_depth=settings.retrieval.get("default_hop_depth", 2),
            top_k_entities=settings.retrieval.get("top_k_entities", 5),
        )

        self.reranker = Reranker(
            model_name=settings.reranker.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=settings.reranker.get("device", "cpu"),
        )

    def ask(self, query: str) -> dict:
        """执行一次问答

        返回: {"answer": str, "context": str, "paths": list, "entry_entities": list}
        """
        # Step 0: 确保实体嵌入已计算（首次查询时懒加载）
        self._ensure_embeddings()

        # Step 1: 混合检索
        paths = self.hybrid_search.search(query)

        if not paths:
            return {
                "answer": "未找到相关知识。请检查图谱是否已入库数据。",
                "context": "",
                "paths": [],
                "retrieval_status": "no_paths",
                "retrieval_mode": self.hybrid_search.last_mode,
            }

        # Step 2: 重排序
        top_n = self.settings.retrieval.get("top_n_paths", 10)
        min_score = self.settings.retrieval.get("reranker_threshold", 0.0)
        ranked_paths = self.reranker.rerank(query, paths, top_n=top_n, min_score=min_score)
        if not ranked_paths:
            return {"answer": "没有候选路径通过重排序，请调整问题或阈值。", "context": "", "paths": [], "sources": [], "retrieval_status": "filtered", "retrieval_mode": self.hybrid_search.last_mode, "reranker_mode": self.reranker.last_mode}

        # Step 3: 上下文拼接
        ids = list({chunk_id for path in ranked_paths for edge in path.get("evidence_ids", []) for chunk_id in edge})
        sources = self.client.run(
            "MATCH (c:Chunk {graph: $graph}) WHERE c.chunk_id IN $ids "
            "RETURN c.chunk_id AS chunk_id, c.source_doc AS source_doc, c.text AS text, c.position AS position",
            {"graph": self.client.graph_name, "ids": ids},
        ) if ids else []
        sources_by_id = {s["chunk_id"]: s for s in sources}
        for path in ranked_paths:
            edge_ids = path.get("evidence_ids", [])
            path["sources"] = [sources_by_id[x] for x in dict.fromkeys(x for edge in edge_ids for x in edge) if x in sources_by_id]
        # 仅使用每条边都能追溯到来源的路径。
        ranked_paths = [p for p in ranked_paths if p.get("evidence_ids") and all(any(x in sources_by_id for x in edge) for edge in p["evidence_ids"])]
        if not ranked_paths:
            return {"answer": "未找到有原文依据的相关知识，请检查文档入库或调整问题。", "context": "", "paths": [], "sources": [], "retrieval_status": "no_evidence", "retrieval_mode": self.hybrid_search.last_mode, "reranker_mode": self.reranker.last_mode}
        context = "\n".join(
            f"图谱路径 {i}: {path['path_text']}；对应原文 " + " ".join(f"[{source['chunk_id']}]" for source in path['sources'])
            for i, path in enumerate(ranked_paths, 1)
        )
        used_ids = {s["chunk_id"] for p in ranked_paths for s in p["sources"]}
        sources = [s for s in sources if s["chunk_id"] in used_ids]
        context += "\n\n原文来源：\n" + "\n".join(f"[{s['chunk_id']}] {s.get('source_doc') or '未命名文档'}：{s.get('text', '')}" for s in sources)

        # Step 4: LLM 生成答案
        answer = self.llm.answer(context, query)

        return {
            "answer": answer,
            "context": context,
            "paths": ranked_paths,
            "sources": sources,
            "reranker_mode": self.reranker.last_mode,
            "reranker_error": self.reranker.last_error,
            "retrieval_status": "ok",
            "retrieval_mode": self.hybrid_search.last_mode,
        }

    def _ensure_embeddings(self):
        """首次查询时懒加载嵌入模型并补算实体向量"""
        self.vector_index.embed_and_update_entities()

    def ask_simple(self, query: str) -> str:
        """简化接口：直接返回答案文本"""
        result = self.ask(query)
        return result["answer"]

    def close(self):
        self.client.close()
