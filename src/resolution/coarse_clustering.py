"""向量粗聚类 —— 低阈值余弦相似度圈定候选消歧簇"""


from __future__ import annotations
import numpy as np
from typing import List, Dict, Set
from sentence_transformers import SentenceTransformer


class CoarseClusterer:
    """用向量相似度将疑似同义/近义的实体聚为候选簇"""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = threshold

    def cluster(self, entities: list[dict]) -> list[list[dict]]:
        """将实体按相似度聚类，返回候选簇列表"""
        if len(entities) < 2:
            return [entities] if entities else []

        names = [e["name"] for e in entities]
        embeddings = self.model.encode(
            [f"{e['name']} {e.get('description', '')}" for e in entities],
            normalize_embeddings=True,
        )

        # 余弦相似度矩阵
        sim_matrix = np.dot(embeddings, embeddings.T)

        # 并查集合并相似实体
        n = len(entities)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > self.threshold:
                    union(i, j)

        # 收集簇
        clusters: Dict[int, List[dict]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(entities[i])

        return list(clusters.values())

    def find_candidates(
        self, new_entity: dict, existing_entities: list[dict]
    ) -> list[dict]:
        """为新实体找到可能冲突的已有实体"""
        if not existing_entities:
            return []

        new_emb = self.model.encode(
            [f"{new_entity['name']} {new_entity.get('description', '')}"],
            normalize_embeddings=True,
        )[0]

        candidates = []
        for entity in existing_entities:
            emb = self.model.encode(
                [f"{entity['name']} {entity.get('description', '')}"],
                normalize_embeddings=True,
            )[0]
            score = float(np.dot(new_emb, emb))
            if score > self.threshold:
                candidates.append({**entity, "similarity": score})

        return sorted(candidates, key=lambda x: x["similarity"], reverse=True)
