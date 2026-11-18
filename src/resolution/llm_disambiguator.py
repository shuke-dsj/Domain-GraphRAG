"""LLM 逻辑消歧 —— 将候选簇 + 频次输入 LLM，输出结构化判决"""


from __future__ import annotations
import json
import logging
from typing import Literal

from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# LLM 判决结果类型
DisambiguationDecision = Literal["synonym", "hypernym", "isolated"]


class LLMDisambiguator:
    """用 LLM 判定同义词/上下位/孤立新概念"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def decide(
        self,
        candidate_cluster: list[dict],
        frequencies: dict[str, int],
    ) -> list[dict]:
        """对候选簇进行消歧判决，返回结构化的判决结果列表

        每项结果包含:
          - entity: 实体名
          - decision: "synonym" | "hypernym" | "isolated"
          - canonical: 标准名（synonym 时），或 parent（hypernym 时）
          - reason: 判决理由
        """
        if len(candidate_cluster) < 2:
            return [
                {"entity": e["name"], "decision": "isolated", "canonical": None, "reason": "唯一实体"}
                for e in candidate_cluster
            ]

        prompt = self._build_disambiguation_prompt(candidate_cluster, frequencies)

        try:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            raw = raw.replace("```json", "").replace("```", "").strip()
            decisions = json.loads(raw)
            return decisions.get("decisions", [])
        except Exception as e:
            logger.warning(f"LLM disambiguation failed: {e}")
            return [{"entity": e["name"], "decision": "isolated", "canonical": None,
                     "reason": "消歧失败，保留原实体"} for e in candidate_cluster]

    def _build_disambiguation_prompt(
        self, cluster: list[dict], freqs: dict[str, int]
    ) -> str:
        entities_desc = []
        for e in cluster:
            name = e["name"]
            f = freqs.get(name, 0)
            t = e.get("type", "")
            d = e.get("description", "")
            aliases = e.get("aliases", [])
            entities_desc.append(
                f"  - 名称: {name}\n    类型: {t}\n    描述: {d}\n    全局频次: {f}\n    已有别名: {aliases}"
            )

        entities_block = "\n".join(entities_desc)

        # JSON 模板（普通字符串）
        json_template = """{
  "decisions": [
    {"entity": "实体名", "decision": "synonym|hypernym|isolated", "canonical": "标准名或上位名", "reason": "理由"}
  ]
}"""

        return f"""你是领域知识图谱的实体消歧专家。请判断以下候选实体簇中每对实体的关系。

## 背景
高频词通常是标准术语，低频词多为别名或变体。实体描述也有助于判断语义关系。

## 候选实体簇
{entities_block}

## 判决规则
1. **synonym（同义词）**: 两个实体指同一概念，仅命名不同（如"离心机轴承"与"离心机内部轴承"）
   → 将低频实体作为高频实体的别名合并
2. **hypernym（上下位概念）**: 一个实体是另一个的子类或子概念（如"高速离心机轴承"是"离心机轴承"的子类）
   → 指定 parent（上位概念）和 child（下位概念）
3. **isolated（孤立新概念）**: 实体是完全不同的概念，不应合并

## 输出格式（严格 JSON，不要思考过程，不要解释）

{json_template}

只输出 JSON："""

    def _fallback_decisions(
        self, cluster: list[dict], freqs: dict[str, int]
    ) -> list[dict]:
        """LLM 不可用时的回退规则：高频词作为标准术语"""
        ranked = sorted(cluster, key=lambda e: freqs.get(e["name"], 0), reverse=True)
        canonical = ranked[0]
        decisions = []
        for e in ranked:
            if e["name"] == canonical["name"]:
                decisions.append({"entity": e["name"], "decision": "isolated", "canonical": None, "reason": "最高频标准术语"})
            else:
                decisions.append({"entity": e["name"], "decision": "synonym", "canonical": canonical["name"], "reason": "低频变体合并到高频标准术语"})
        return decisions
