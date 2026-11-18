"""Schema 约束的 LLM 实体提取器 —— 注入领域白名单与 Few-shot Prompt"""


from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.llm.base import BaseLLM
from src.extraction.models import Entity, Relation, ExtractionResult
from src.extraction.boundary_defender import BoundaryDefender

logger = logging.getLogger(__name__)


class SchemaConstrainedExtractor:
    """用领域 Schema 约束 LLM，强制执行实体/关系类型白名单"""

    def __init__(
        self,
        llm: BaseLLM,
        boundary_defender: BoundaryDefender,
        entity_types: list[str],
        mandatory_relations: list[str],
        allow_free_relations: bool = True,
    ):
        self.llm = llm
        self.defender = boundary_defender
        self.entity_types = entity_types
        self.mandatory_relations = mandatory_relations
        self.allow_free_relations = allow_free_relations

    def extract(self, chunk_text: str, chunk_id: str = "") -> ExtractionResult:
        """对单个 chunk 执行 Schema 约束的实体关系提取"""

        marked_text = self.defender.mark(chunk_text)
        prompt = self._build_prompt(marked_text)

        try:
            raw = self.llm.extract_entities(prompt, chunk_id=chunk_id, chunk_preview=chunk_text)
            result = self._validate(raw, chunk_id, chunk_text)
            return result
        except Exception as e:
            logger.warning(f"提取失败 chunk={chunk_id}: {e}")
            raise  # 抛出错误，让上层决定是否重试

    def _build_prompt(self, text: str) -> str:
        """构造带 Schema 约束的 Few-shot 提示词"""

        entity_types_str = "、".join(self.entity_types)
        mandatory_rel_str = "、".join(self.mandatory_relations)
        free_hint = (
            f'如果需要，也可以提取上述之外的关系类型。'
            if self.allow_free_relations
            else "只允许使用上述关系类型。"
        )

        # JSON 模板（普通字符串，直接含花括号）
        json_template = """{
  "entities": [
    {"name": "实体名称", "type": "类型", "description": "≤15字"}
  ],
  "relations": [
    {"source": "源实体名", "target": "目标实体名", "relation_type": "关系类型"}
  ]
}"""

        return f"""你是领域知识图谱的实体关系提取专家。请严格按照以下规则从文本中提取信息。
# 最高规则
你只能返回json格式的输出，且必须完全遵守以下规则进行提取。任何不符合规则的输出都将被视为无效。
        
## 规则1：术语边界
文本中所有被【】包裹的词汇是不可拆分的专业术语，提取为实体时必须原样保留，不得增删文字。

## 规则2：实体类型约束
优先归类到以下类型：{entity_types_str}
如果以上类型都不合适，可以自主定义新的类型名（如"安全等级""检测方法"等），但不要设为无效类型。
假如这段话中没有有用实体信息，例如这是一段歌词/广告语/数学公式等，请返回一个空的实体列表，不要编造无关实体。

## 规则3：关系类型
优先使用：{mandatory_rel_str}
{free_hint}

## 规则4：提取原则
- 每个实体独立提取，不在提取阶段判断同义词或别名
- 同一事物的不同说法各自作为独立实体提取，后续统一消歧
- 实体名使用文本中出现的原文表述
- 描述控制在50字以内，只写核心描述，忠于原文，不要添加任何推测信息

## 规则5：输出格式
严格输出 JSON，不要思考过程，不要 Markdown 代码块，不要有任何解释文字，你只返回 JSON 格式即可：

{json_template}

## 待提取文本
{text}

请根据上述规则提取实体和关系，输出 JSON。
"""

    @staticmethod
    def _validate(raw: dict, chunk_id: str, chunk_text: str) -> ExtractionResult:
        if not isinstance(raw, dict) or not isinstance(raw.get("entities"), list) or not isinstance(raw.get("relations"), list):
            raise ValueError("抽取结果必须包含 entities 和 relations 列表")
        entities = []
        relations = []

        for e in raw.get("entities", []):
            try:
                entity = Entity(
                    name=e.get("name", ""),
                    type=e.get("type", ""),
                    description=e.get("description", ""),
                    aliases=e.get("aliases", []),
                    chunk_id=chunk_id,
                )
                entities.append(entity)
            except ValidationError as ve:
                logger.warning(f"实体校验失败: {e}, 错误: {ve}")

        for r in raw.get("relations", []):
            try:
                rel = Relation(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relation_type=r.get("relation_type", ""),
                    evidence=r.get("evidence", ""),
                    chunk_id=chunk_id,
                )
                relations.append(rel)
            except ValidationError as ve:
                logger.warning(f"关系校验失败: {r}, 错误: {ve}")

        return ExtractionResult(
            entities=entities,
            relations=relations,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
        )
