"""Neo4j 客户端 —— 连接管理、会话池、参数化查询"""

import logging
from typing import Any, Dict, List

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j 图数据库客户端，封装连接管理和参数化查询

    社区版通过 graph_name 属性隔离不同知识库（企业版可直接建多个数据库）。
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        graph_name: str = "default",
    ):
        self.uri = uri
        self.user = user
        self.database = database
        self.graph_name = graph_name
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()

    def _verify_connection(self):
        try:
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j connected: {self.uri}, db={self.database}, graph={self.graph_name}")
        except Exception as e:
            self._driver.close()
            hint = ""
            err_str = str(e)
            if "Unauthorized" in err_str or "authentication" in err_str.lower():
                hint = (
                    "\nNeo4j 认证失败。请检查：\n"
                    "  1. 是否已从 .env.example 创建 .env 文件并填入真实密码？\n"
                    "  2. Neo4j 用户名/密码是否正确？（默认 neo4j/你的密码）\n"
                    "  3. Neo4j 服务是否已启动？"
                )
            logger.error(f"Neo4j connection failed: {e}{hint}")
            raise RuntimeError(f"Neo4j connection failed: {e}{hint}") from e

    @property
    def driver(self) -> Driver:
        return self._driver

    def session(self) -> Session:
        return self._driver.session(database=self.database)

    # ── 通用查询 ────────────────────────────────────────────

    def run(self, cypher: str, params: dict = None) -> List[Dict[str, Any]]:
        """执行参数化 Cypher 查询，返回记录列表"""
        with self.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def execute_write(self, cypher: str, params: dict = None):
        """执行写事务"""
        with self.session() as session:
            session.execute_write(lambda tx: tx.run(cypher, params or {}))

    def execute_read(self, cypher: str, params: dict = None) -> List[Dict[str, Any]]:
        """执行读事务"""
        with self.session() as session:
            return list(session.execute_read(lambda tx: list(tx.run(cypher, params or {}))))

    # ── Schema 初始化 ─────────────────────────────────────

    def init_schema(self):
        """创建约束和索引（社区版用复合唯一约束区分 graph）"""
        constraints = [
            "CREATE CONSTRAINT entity_name_graph IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.graph) IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_graph IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE (c.chunk_id, c.graph) IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_graph IF NOT EXISTS FOR (e:Entity) ON (e.graph)",
        ]

        for cypher in constraints + indexes:
            try:
                self.run(cypher)
            except Neo4jError as e:
                # 约束/索引已存在等非致命错误
                logger.debug(f"Schema init: {e}")

    def clear_graph(self):
        """清空当前 graph 下的所有节点和关系"""
        self.run(
            "MATCH (n {graph: $graph}) DETACH DELETE n",
            {"graph": self.graph_name},
        )

    def close(self):
        self._driver.close()
