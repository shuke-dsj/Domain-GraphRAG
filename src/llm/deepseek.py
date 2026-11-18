"""DeepSeek V3 适配器 —— 通过 SiliconFlow API 调用"""

import datetime
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from typing import Dict, List

from src.llm.base import BaseLLM


class DeepSeekAdapter(BaseLLM):
    """SiliconFlow DeepSeek V3 API 适配器"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        default_model: str = "Pro/deepseek-ai/DeepSeek-V3",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self._deepseek_official = urlparse(base_url).hostname == "api.deepseek.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 32768,
        response_format: dict = None,
    ) -> str:
        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self._deepseek_official:
            payload["thinking"] = {"type": "disabled"}
        if response_format:
            payload["response_format"] = response_format

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"LLM API error [{response.status_code}]: {response.text}")

        data = response.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})
        if self._deepseek_official and msg.get("reasoning_content"):
            raise RuntimeError("模型返回了不应出现的思维链，请检查服务配置")

        content = msg.get("content", "")
        if not content:
            raise RuntimeError(
                f"LLM 返回空 content。finish_reason={choice.get('finish_reason')}, "
                f"message keys={list(msg.keys())}"
            )

        return content

    def extract_entities(self, text: str, chunk_id: str = "", chunk_preview: str = "") -> Dict:
        """从已格式化好的提示词文本中提取实体关系 —— text 已是完整的用户提示词"""
        messages = [
            {
                "role": "system",
                "content": "你是信息提取专家。只输出JSON，不要思考过程，不要解释，不要Markdown。",
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 8192,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self._deepseek_official:
            payload["thinking"] = {"type": "disabled"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120,
            )
            if response.status_code != 200:
                raise RuntimeError(f"LLM API error [{response.status_code}]: {response.text}")

            data = response.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            if self._deepseek_official and msg.get("reasoning_content"):
                raise RuntimeError("模型返回了不应出现的思维链，请检查服务配置")
            content = msg.get("content", "")

            if not content:
                raise RuntimeError(
                    f"LLM 返回空 content。finish_reason={choice.get('finish_reason')}"
                )

            result = self._parse_json_response(content)
            self._save_response_log(text, content, chunk_id, chunk_preview, success=True)
            return result
        except Exception as e:
            raw_response = content if 'content' in dir() else ""
            self._save_response_log(text, raw_response, chunk_id, chunk_preview, success=False)
            raise RuntimeError(f"实体提取请求失败: {e}")

    @staticmethod
    def _save_response_log(prompt: str, response: str, chunk_id: str,
                           chunk_preview: str, *, success: bool):
        """保存 LLM 输入和输出到 log 目录"""
        subdir = "runtime" if success else "error"
        log_dir = Path("log") / subdir
        log_dir.mkdir(parents=True, exist_ok=True)

        # 移除 Windows 文件名非法字符：< > : " / \ | ? *
        preview = re.sub(r'[<>:"/\\|?*]', '_', chunk_preview[:10])
        preview = preview.replace("\n", " ").strip() or "unknown"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chunk_{preview}_{chunk_id}_{ts}.log"
        filepath = log_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CHUNK_ID: {chunk_id}\n")
            f.write(f"TIMESTAMP: {ts}\n")
            f.write("=" * 60 + "\n\n")
            f.write(">>> INPUT (PROMPT):\n\n")
            f.write(prompt)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(">>> OUTPUT (RESPONSE):\n\n")
            f.write(response if response else "(empty)")

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """鲁棒 JSON 解析：处理 LLM 常见的格式问题"""
        text = response.strip()

        # 去除 markdown 代码块
        for fence in ("```json", "```"):
            text = text.replace(fence, "")

        # 尝试1：直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试2：修复尾逗号（LLM 常见错误： ] 前的逗号，} 前的逗号）
        text = re.sub(r",\s*\]", "]", text)
        text = re.sub(r",\s*\}", "}", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试3：如果 JSON 被截断（缺 }），尝试补全
        if text.count("{") > text.count("}"):
            text += "}" * (text.count("{") - text.count("}"))
        if text.count("[") > text.count("]"):
            text += "]" * (text.count("[") - text.count("]"))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        raise RuntimeError(
            f"JSON 解析失败。响应长度={len(response)}, "
            f"尾部200字符: ...{response[-200:] if response else '(empty)'}"
        )

    def summarize(self, texts: List[str], query: str = None) -> str:
        combined = "\n---\n".join(texts[:5])

        messages = [
            {
                "role": "user",
                "content": (
                    f"总结以下内容，提取关键信息和要点：\n\n{combined[:3000]}\n\n"
                    f"{'针对问题：' + query if query else '请简明扼要总结：'}"
                ),
            }
        ]

        return self.chat(messages, temperature=0.3)

    def answer(self, context: str, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是专业领域知识助手。只能依据提供的原文回答，事实后必须引用原文的完整块ID，格式为[块ID]；不要引用图谱路径编号，不要编造引用。证据不足时明确说明，禁止编造事实。原文没有提到某种关系，不代表该关系不存在；不能将未说明的内容断言为否定事实。",
            },
            {
                "role": "user",
                "content": (
                    f"根据以下知识内容回答问题：\n\n知识内容：\n{context[:12000]}\n\n"
                    f"用户问题：{query}\n\n要求：直接回答问题并引用原文块ID。只陈述原文明示或能可靠推得的内容；如果原文没有说明，明确指出而不要推测。\n\n回答："
                ),
            },
        ]

        return self.chat(messages, temperature=0.4)
