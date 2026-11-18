"""实体与关系的 Pydantic 数据模型 —— 严格校验 LLM 输出"""


from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """知识图谱实体"""

    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型，必须在领域白名单内")
    description: str = Field(default="", description="实体描述")
    aliases: list[str] = Field(default_factory=list, description="别名列表")
    chunk_id: str = Field(default="", description="来源文档块 ID")


class Relation(BaseModel):
    """实体间关系"""

    source: str = Field(..., description="源实体名称")
    target: str = Field(..., description="目标实体名称")
    relation_type: str = Field(..., description="关系类型")
    evidence: str = Field(default="", description="关系在原文中的证据")
    chunk_id: str = Field(default="", description="来源文档块 ID")


class ExtractionResult(BaseModel):
    """单次 LLM 抽取的结果"""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    chunk_id: str = Field(default="")
    chunk_text: str = Field(default="", description="原始文本留档")


class ChunkMetadata(BaseModel):
    """文档分块元数据"""

    chunk_id: str
    text: str
    token_count: int = 0
    source_doc: str = ""
    page_range: tuple[int, int] = Field(default=(0, 0))
    has_pic_marker: bool = False
