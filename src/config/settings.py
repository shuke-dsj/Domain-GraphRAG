"""配置加载器 —— 读取 YAML 配置文件 + .env + 环境变量覆盖敏感信息"""


from __future__ import annotations
import os
import warnings
import yaml
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
_dotenv_path = Path(__file__).resolve().parents[2] / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)
else:
    warnings.warn(
        f".env file not found at {_dotenv_path}. "
        f"Copy .env.example to .env and fill in your credentials."
    )


class Settings:
    """单例配置管理，优先环境变量覆盖"""

    _instance: "Settings | None" = None
    _data: dict[str, Any] = {}

    def __init__(self, config_path: str | Path = "config.yaml"):
        if Settings._instance is not None:
            return
        Settings._instance = self

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        # 环境变量覆盖敏感配置
        self._apply_env_overrides()

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> "Settings":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def _apply_env_overrides(self):
        # ── HuggingFace 配置（必须在任何模型加载前设置）──
        emb = self._data.get("embedding", {})
        if emb.get("offline_mode"):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        if emb.get("hf_endpoint"):
            os.environ["HF_ENDPOINT"] = emb["hf_endpoint"]

        # ── LLM ──
        llm = self._data.setdefault("llm", {})
        if os.getenv("LLM_API_KEY"):
            llm["api_key"] = os.getenv("LLM_API_KEY")
        if os.getenv("LLM_BASE_URL"):
            llm["base_url"] = os.getenv("LLM_BASE_URL")
        if os.getenv("LLM_DEFAULT_MODEL"):
            llm["default_model"] = os.getenv("LLM_DEFAULT_MODEL")

        # ── Neo4j ──
        neo4j = self._data.setdefault("neo4j", {})
        if os.getenv("NEO4J_URI"):
            neo4j["uri"] = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USER"):
            neo4j["user"] = os.getenv("NEO4J_USER")
        if os.getenv("NEO4J_PASSWORD"):
            neo4j["password"] = os.getenv("NEO4J_PASSWORD")
        if os.getenv("NEO4J_DATABASE"):
            neo4j["database"] = os.getenv("NEO4J_DATABASE")
        if os.getenv("GRAPHRAG_GRAPH"):
            neo4j["graph"] = os.getenv("GRAPHRAG_GRAPH")

        if os.getenv("GRAPHRAG_EXTRACTED_DIR"):
            self._data.setdefault("paths", {})["extracted_dir"] = os.getenv("GRAPHRAG_EXTRACTED_DIR")

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    @property
    def llm(self) -> dict:
        return self._data.get("llm", {})

    @property
    def neo4j(self) -> dict:
        return self._data.get("neo4j", {})

    @property
    def paths(self) -> dict:
        return self._data.get("paths", {})

    @property
    def domain(self) -> str:
        return self._data.get("domain", "general.yaml")

    @property
    def embedding(self) -> dict:
        return self._data.get("embedding", {})

    @property
    def extraction(self) -> dict:
        return self._data.get("extraction", {})

    @property
    def retrieval(self) -> dict:
        return self._data.get("retrieval", {})

    @property
    def reranker(self) -> dict:
        return self._data.get("reranker", {})
