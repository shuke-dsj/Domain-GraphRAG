"""AC 自动机实体边界防御 —— 文本送入 LLM 前标记专业术语"""


from __future__ import annotations
import ahocorasick
from typing import Set


class BoundaryDefender:
    """用 AC 自动机扫描文本，将领域术语用 【】 包裹，防止 LLM 拆分"""

    MARKER_LEFT = "【"
    MARKER_RIGHT = "】"

    def __init__(self, terminology: list[str]):
        self._automaton = ahocorasick.Automaton()
        self._active_terms: Set[str] = set()
        self.load_terminology(terminology)

    def load_terminology(self, terms: list[str]):
        """加载领域术语白名单，构建 AC 自动机"""
        self._automaton = ahocorasick.Automaton()
        self._active_terms = set()
        for term in terms:
            t = term.strip()
            if t and t not in self._active_terms:
                self._active_terms.add(t)
                self._automaton.add_word(t, t)
        if self._active_terms:
            self._automaton.make_automaton()

    def mark(self, text: str) -> str:
        """扫描文本，将匹配到的术语用 【】 包裹

        规则：
        - 只标记第一次出现（避免过度标记）
        - 如果术语已被 【】 包裹，则跳过
        - 长术语优先，短术语在后（避免"离心机"遮盖"离心机轴承"）
        """
        if not self._active_terms:
            return text

        ends_with_marker: list[tuple[int, int, str]] = []

        for end_index, term in self._automaton.iter(text):
            start = end_index - len(term) + 1
            ends_with_marker.append((start, end_index, term))

        if not ends_with_marker:
            return text

        # 按位置排序，长术语优先
        ends_with_marker.sort(key=lambda x: (x[0], -len(x[2])))

        # 去重叠：短术语如果被长术语覆盖则跳过
        filtered: list[tuple[int, int, str]] = []
        for start, end, term in ends_with_marker:
            # 检查是否已有 【 在 start 前或 】在 end 后（已被标记）
            if self._already_marked(text, start, end):
                continue
            # 检查是否与已选中的重叠
            overlaps = False
            for fs, fe, _ in filtered:
                if not (end < fs or start > fe):
                    overlaps = True
                    break
            if not overlaps:
                filtered.append((start, end, term))

        # 从后往前替换（避免索引偏移）
        parts = []
        last_idx = 0
        for start, end, term in sorted(filtered, key=lambda x: x[0]):
            parts.append(text[last_idx:start])
            parts.append(f"{self.MARKER_LEFT}{term}{self.MARKER_RIGHT}")
            last_idx = end + 1
        parts.append(text[last_idx:])

        return "".join(parts)

    def _already_marked(self, text: str, start: int, end: int) -> bool:
        """检查 start 前是否已有 【 且 end 后是否已有 】"""
        before = text[max(0, start - 1):start]
        after = text[end + 1:end + 2]
        if before == self.MARKER_LEFT and after == self.MARKER_RIGHT:
            return True
        return False

    @property
    def term_count(self) -> int:
        return len(self._active_terms)
