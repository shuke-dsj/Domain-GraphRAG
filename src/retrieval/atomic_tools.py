"""原子化 Cypher 工具 —— Agent 传参调用，避免 Text-to-Cypher 风险"""

from typing import Dict, List, Optional

from src.storage.neo4j_client import Neo4jClient


class AtomicTools:
    """封装高频图检索动作为参数化函数，向 Agent 暴露安全接口"""

    def __init__(self, client: Neo4jClient):
        self.client = client
        self.graph = client.graph_name

    def get_n_hop_neighbors(
        self,
        entity_name: str,
        depth: int = 2,
        directed: bool = False,
        limit: int = 50,
    ) -> List[Dict]:
        """获取实体 N 跳范围内的邻居子图"""
        if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 3:
            raise ValueError("depth must be an integer between 1 and 3")
        limit = max(1, min(int(limit), 100))
        right_arrow = ">" if directed else ""
        cypher = f"""
        MATCH path = (start:Entity {{name: $name, graph: $graph}})-[:RELATES|IS_A *1..{depth}]-{right_arrow}(neighbor:Entity)
        WHERE all(n IN nodes(path) WHERE n.graph = $graph)
        WITH path LIMIT $limit
        RETURN [n in nodes(path) | n.name] AS node_chain,
               [r in relationships(path) | COALESCE(r.type, type(r))] AS edge_types,
               [i IN range(0, length(path)-1) | startNode(relationships(path)[i]) = nodes(path)[i]] AS directions,
               [r in relationships(path) | COALESCE(r.chunk_ids, [])] AS evidence_ids,
               [n in nodes(path) | COALESCE(n.description, '')] AS descriptions
        """
        return self.client.run(
            cypher, {"name": entity_name, "limit": limit, "graph": self.graph}
        )

    def find_shortest_path(
        self, entity_a: str, entity_b: str, max_depth: int = 4
    ) -> List[Dict]:
        """查找两个实体之间的最短路径"""
        cypher = f"""
        MATCH path = shortestPath(
            (a:Entity {{name: $a, graph: $graph}})-[:RELATES|IS_A *1..{max_depth}]-(b:Entity {{name: $b, graph: $graph}})
        )
        RETURN [n in nodes(path) | n.name] AS path_nodes,
               [r in relationships(path) | type(r)] AS edge_types,
               length(path) AS path_length
        """
        return self.client.run(cypher, {"a": entity_a, "b": entity_b, "graph": self.graph})

    def get_entity_context(self, entity_name: str) -> List[Dict]:
        """获取实体详细上下文：属性、关联实体、出现文档"""
        cypher = """
        MATCH (e:Entity {name: $name, graph: $graph})
        OPTIONAL MATCH (e)-[r:RELATES]-(other:Entity {graph: $graph})
        OPTIONAL MATCH (e)-[:APPEARS_IN]->(c:Chunk {graph: $graph})
        RETURN e.name AS name,
               e.type AS type,
               e.description AS description,
               e.aliases AS aliases,
               collect(DISTINCT other.name)[0..10] AS related_entities,
               collect(DISTINCT c.text)[0..3] AS source_snippets
        """
        return self.client.run(cypher, {"name": entity_name, "graph": self.graph})

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> List[Dict]:
        """按类型列出实体"""
        return self.client.run(
            "MATCH (e:Entity {type: $type, graph: $graph}) "
            "RETURN e.name AS name, e.description AS description LIMIT $limit",
            {"type": entity_type, "limit": limit, "graph": self.graph},
        )

    def check_relation_exists(
        self, source: str, target: str, relation_type: Optional[str] = None
    ) -> bool:
        """检查两个实体间是否存在特定关系"""
        cypher = """
        MATCH (a:Entity {name: $source, graph: $graph})-[r:RELATES]->(b:Entity {name: $target, graph: $graph})
        WHERE $relation_type IS NULL OR r.type = $relation_type
        RETURN count(r) > 0 AS exists
        """
        results = self.client.run(cypher, {"source": source, "target": target, "graph": self.graph, "relation_type": relation_type})
        return results[0]["exists"] if results else False

    def search_entity_by_name(self, keyword: str, limit: int = 10) -> List[Dict]:
        """按名称模糊搜索实体"""
        cypher = """
        MATCH (e:Entity)
        WHERE e.graph = $graph AND e.name CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return self.client.run(cypher, {"keyword": keyword, "limit": limit, "graph": self.graph})
