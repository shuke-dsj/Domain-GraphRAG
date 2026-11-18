"""Cross-Encoder 重排序 —— 对图谱路径语义截断"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Reranker:
    """用 Cross-Encoder 模型对图谱检索路径进行语义重排序"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.last_mode = "uninitialized"
        self.last_error = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder: {e}")
                self.last_error = str(e)
                self._model = False  # sentinel
        return self._model if self._model is not False else None

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_n: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """对候选路径列表进行 Cross-Encoder 打分和排序

        每个 candidate 需包含 "path_text" 字段。
        返回附带 rerank_score 的排序结果。
        """
        if not candidates:
            self.last_mode = "empty"
            return []

        cross_encoder = self.model
        if cross_encoder is None:
            self.last_mode = "vector_fallback"
            logger.info("CrossEncoder not available, using vector_score as fallback")
            return self._fallback_rerank(candidates, top_n)

        texts = [c.get("path_text", "") for c in candidates]
        pairs = [(query, t) for t in texts]

        try:
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
        except Exception as e:
            self.last_error = str(e)
            self.last_mode = "vector_fallback"
            logger.warning("CrossEncoder inference failed: %s", e)
            return self._fallback_rerank(candidates, top_n)
        self.last_mode = "cross_encoder"

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # 阈值截断 + Top-N
        filtered = [c for c in candidates if min_score is None or c.get("rerank_score", 0) >= min_score]
        return filtered[:top_n]

    def _fallback_rerank(self, candidates: List[Dict], top_n: int) -> List[Dict]:
        """回退：使用向量分数排序"""
        for c in candidates:
            if "rerank_score" not in c:
                c["rerank_score"] = c.get("vector_score", 0)
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates[:top_n]

    def rerank_and_format(
        self,
        query: str,
        candidates: List[Dict],
        top_n: int = 10,
        min_score: float = 0.0,
    ) -> str:
        """重排序并输出格式化上下文"""
        ranked = self.rerank(query, candidates, top_n, min_score)

        return self.format_ranked(ranked)

    @staticmethod
    def format_ranked(ranked: List[Dict]) -> str:
        lines = []
        for i, item in enumerate(ranked, 1):
            score = item.get("rerank_score", 0)
            path = item.get("path_text", "")
            descs = " | ".join(d for d in item.get("descriptions", []) if d)
            lines.append(f"[{i}] (score={score:.3f}) {path}")
            if descs:
                lines.append(f"    描述: {descs}")

        return "\n".join(lines)
