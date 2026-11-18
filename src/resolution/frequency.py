"""频次特征提取 —— 统计实体在全局文档中的出现频次"""


from __future__ import annotations
import re
from collections import Counter


class FrequencyAnalyzer:
    """统计实体在各 chunk 中的出现频次，按 README 2.3 Step 2 设计"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._global_counter: Counter | None = None
        self._build_global_index()

    def _build_global_index(self):
        """统计每个实体名在全部 chunk 中的出现次数"""
        self._global_counter = Counter()
        full_text = " ".join(c.get("text", "") for c in self.chunks)
        # 简单子串匹配（后续可升级为精确词边界匹配）
        pass  # 延迟计算，按需执行

    def get_frequencies(self, entities: list[dict]) -> dict[str, int]:
        """获取一组实体在全局文本中的频次"""
        full_text = " ".join(c.get("text", "") for c in self.chunks)
        freqs = {}
        for entity in entities:
            name = entity["name"]
            # 精确匹配实体名出现次数（含变体）
            count = len(re.findall(re.escape(name), full_text))
            # 也检查别名
            for alias in entity.get("aliases", []):
                count += len(re.findall(re.escape(alias), full_text))
            freqs[name] = count
        return freqs

    def rank_by_frequency(
        self, candidate_cluster: list[dict]
    ) -> list[tuple[dict, int]]:
        """对候选簇内的实体按频次降序排列"""
        freqs = self.get_frequencies(candidate_cluster)
        ranked = [(entity, freqs.get(entity["name"], 0)) for entity in candidate_cluster]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def is_high_freq_dominant(
        self,
        ranked: list[tuple[dict, int]],
        ratio_threshold: float = 3.0,
    ) -> bool:
        """判断高频实体是否显著主导（高频/低频 > ratio_threshold）"""
        if len(ranked) < 2:
            return True
        high_freq = ranked[0][1]
        low_freq = ranked[-1][1]
        if low_freq == 0:
            return True
        return (high_freq / low_freq) > ratio_threshold
