"""LLM 抽象基类 —— 定义所有 LLM 后端必须实现的接口"""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseLLM(ABC):
    """所有 LLM 后端的统一抽象接口"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
        response_format: dict = None,
    ) -> str:
        """通用对话接口"""
        ...

    @abstractmethod
    def extract_entities(self, text: str, chunk_id: str = "", chunk_preview: str = "") -> Dict:
        """从文本中提取实体和关系，返回结构化字典"""
        ...

    @abstractmethod
    def summarize(self, texts: List[str], query: str = None) -> str:
        """总结文本内容"""
        ...

    @abstractmethod
    def answer(self, context: str, query: str) -> str:
        """基于上下文生成最终回答"""
        ...
