"""智能文本分块器 —— 按段落/Markdown表格边界切割，支持断点续传"""


from __future__ import annotations
import re
import json
import hashlib
from pathlib import Path
from typing import Iterator

from src.extraction.models import ChunkMetadata


class SmartChunker:
    """按语义边界切割文本，保持强因果关系的段落不被打散"""

    def __init__(
        self,
        target_tokens: int = 1500,
        min_tokens: int = 512,
    ):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens

    # ── token 估算 ──────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """粗略估算 token 数：中文 ~1.5 字符/token，英文 ~4 字符/token"""
        chinese_chars = len(re.findall(r"[一-鿿]", text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    # ── 段落边界检测 ──────────────────────────────────────────

    @staticmethod
    def _split_segments(text: str) -> list[str]:
        """将文本拆分为语义段（段落、表格行、标题等）"""
        # 去掉 Markdown 图片引用（保留段落结构）
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # 优先找 Markdown 表格边界
        segments = re.split(r"(\n(?:\|.+\|\n)+)", text)
        result = []
        for seg in segments:
            if re.match(r"\n(?:\|.+\|\n)+", seg):
                result.append(seg)
            else:
                # 按双换行拆段落
                result.extend(seg.split("\n\n"))
        return [s.strip() for s in result if s.strip()]

    # ── 分块迭代 ─────────────────────────────────────────────

    def chunk_iter(
        self,
        text: str,
        source_doc: str = "",
        start_page: int = 1,
        start_chunk_idx: int = 0,
    ) -> Iterator[ChunkMetadata]:
        """将文本划分为指定 token 范围的数据块，返回迭代器"""
        segments = self._split_segments(text)
        buf = ""
        buf_tokens = 0
        page_offset = 0
        chunk_idx = start_chunk_idx

        for seg in segments:
            seg_tokens = self.estimate_tokens(seg)

            if buf_tokens + seg_tokens > self.target_tokens and buf_tokens >= self.min_tokens:
                yield self._make_chunk(buf, source_doc, start_page + page_offset, chunk_idx)
                chunk_idx += 1
                buf = seg
                buf_tokens = seg_tokens
            else:
                buf = f"{buf}\n\n{seg}" if buf else seg
                buf_tokens += seg_tokens

            # 检测页内换页符（可选）
            if "\f" in seg:
                page_offset += 1

        if buf.strip():
            yield self._make_chunk(buf, source_doc, start_page + page_offset, chunk_idx)

    def chunk(self, text: str, source_doc: str = "", start_page: int = 1) -> list[ChunkMetadata]:
        return list(self.chunk_iter(text, source_doc, start_page))

    def _make_chunk(self, text: str, source_doc: str, page: int, idx: int) -> ChunkMetadata:
        chunk_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        return ChunkMetadata(
            chunk_id=chunk_id,
            text=text,
            token_count=self.estimate_tokens(text),
            source_doc=source_doc,
            page_range=(page, page),
            has_pic_marker=bool(re.search(r"[<\[【]PIC[>\]】]", text)),
        )

    # ── 断点续传支持 ─────────────────────────────────────────

    def save_checkpoint(self, chunks: list[ChunkMetadata], path: str | Path):
        """保存分块结果，支持断点续传"""
        data = [c.model_dump() for c in chunks]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, path: str | Path) -> list[ChunkMetadata]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChunkMetadata(**item) for item in data]
