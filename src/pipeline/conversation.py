"""多轮对话管理 —— 对话历史、指代消解、上下文跟踪"""

import re
from typing import List, Dict, Optional


class ConversationManager:
    """维护多轮对话上下文，支持指代消解"""

    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history
        self._current_entities: List[str] = []
        self._last_intent: Optional[str] = None

    def add_turn(self, user_query: str, answer: str, entities: List[str] = None):
        """记录一轮对话"""
        turn = {
            "user": user_query,
            "answer": answer,
            "entities": entities or [],
        }
        self.history.append(turn)

        if entities:
            self._current_entities = entities

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def resolve_pronouns(self, query: str) -> str:
        """将查询中的代词替换为最近提到的实体

        支持的指代：它、这个、那个、该、上述、提到、此
        """
        if not self._current_entities:
            return query

        pronouns = ["它", "这个", "那个", "该", "上述", "此", "其"]
        has_pronoun = any(p in query for p in pronouns)

        if not has_pronoun:
            # 如果查询中没有显式实体，且包含"呢""怎么样"等疑问词
            if not self._has_explicit_entity(query) and ("呢" in query or "怎么样" in query):
                return f"{self._current_entities[0]}{query}"
            return query

        # 用最近实体替换代词
        resolved = query
        for pronoun in pronouns:
            if pronoun in resolved:
                resolved = resolved.replace(pronoun, self._current_entities[0])
        return resolved

    def _has_explicit_entity(self, query: str) -> bool:
        """检查查询中是否有显式的实体提及（非代词）"""
        pronouns = {"它", "这个", "那个", "该", "上述", "此", "其", "呢", "怎么样", "什么", "如何"}
        words = set(query.replace("？", "").replace("?", "").split())
        # 简单判断：查询中是否有非代词的实词
        content_words = [w for w in words if w not in pronouns and len(w) >= 2]
        return len(content_words) > 0

    @property
    def recent_entities(self) -> List[str]:
        return self._current_entities

    @property
    def recent_context(self) -> str:
        """最近几轮对话的摘要，供 LLM 参考"""
        if not self.history:
            return ""
        recent = self.history[-3:]
        lines = []
        for t in recent:
            lines.append(f"用户: {t['user']}\n助手: {t['answer'][:200]}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()
        self._current_entities = []
        self._last_intent = None
