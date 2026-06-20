"""知识图谱查询引擎。

支持：
  1. 自然语言/关键词查询（邻域扩展）
  2. 影响传播查询（"如果改变 X，什么会被影响"）
  3. 态射链追踪（"从这个结构到那个结构经过了哪些态射"）
  4. LLM 上下文生成（将查询结果转为 prompt 友好的文本）
"""

from __future__ import annotations

from .graph import MathKnowledgeGraph, node_id


class GraphQueryEngine:
    """知识图谱查询引擎."""

    def __init__(self, kg: MathKnowledgeGraph):
        self.kg = kg

    # ── 基本查询 ──

    def query(self, seed: str, depth: int = 1, top_k: int = 20) -> dict:
        """关键词/自然语言查询."""
        return self.kg.query(seed, depth=depth, top_k=top_k)

    # ── 影响传播 ──

    def impact(self, param_name: str, max_depth: int = 3) -> dict:
        """参数改变的下游影响分析.

        从参数节点出发，沿着 controls/affects/constrains 边向前传播。
        """
        pid = node_id(param_name, "parameter")
        if pid not in self.kg._graph:
            return {"error": f"Parameter '{param_name}' not found", "impacted": []}

        visited: set[str] = set()
        impacted: list[dict] = []

        def _traverse(node: str, depth: int, path: list[str]):
            if depth > max_depth or node in visited:
                return
            visited.add(node)
            data = self.kg._graph.nodes[node]

            for _, dst, edge_data in self.kg._graph.out_edges(node, data=True):
                dst_data = self.kg._graph.nodes[dst]
                relation = edge_data.get("relation", "related_to")

                if relation in ("controls", "affects", "constrains", "loses", "maps_to"):
                    new_path = path + [f"{data.get('label', node)} --[{relation}]→ {dst_data.get('label', dst)}"]
                    impacted.append(
                        {
                            "source": data.get("label", node),
                            "relation": relation,
                            "target": dst_data.get("label", dst),
                            "target_type": dst_data.get("type", ""),
                            "path": " → ".join(new_path),
                            "depth": depth,
                        }
                    )
                    _traverse(dst, depth + 1, new_path)

        _traverse(pid, 0, [])

        return {
            "parameter": param_name,
            "impacted_count": len(impacted),
            "impacted": impacted,
        }

    # ── 根因分析 ──

    def root_cause(self, problem_desc: str) -> dict:
        """从问题描述反向追踪可能的根因.

        例如："SCF 不收敛" → 找到 convergence 不变量 → 找到控制它的参数 → 报告
        """
        # 查找相关的不变量
        inv_nodes = self.kg.find_nodes(problem_desc, entity_type="invariant")

        results = []
        for inv_id in inv_nodes:
            # 反向追踪：什么参数/近似约束了这个不变量
            predecessors = list(self.kg._graph.predecessors(inv_id))
            related = []
            for pred in predecessors:
                pred_data = self.kg._graph.nodes[pred]
                edge_data = self.kg._graph.get_edge_data(pred, inv_id) or {}
                related.append(
                    {
                        "node": pred_data.get("label", pred),
                        "type": pred_data.get("type", ""),
                        "relation": edge_data.get("relation", "related_to"),
                    }
                )
            results.append(
                {
                    "invariant": self.kg._graph.nodes[inv_id].get("label", inv_id),
                    "related_entities": related,
                }
            )

        return {"query": problem_desc, "root_causes": results}

    # ── 态射链查询 ──

    def trace_morphism_chain(self, from_label: str, to_label: str) -> dict:
        """追踪从源结构到目标结构的态射链."""
        from_id = node_id(from_label, "structure")
        to_id = node_id(to_label, "structure")

        if from_id not in self.kg._graph or to_id not in self.kg._graph:
            return {"error": "One or both structures not found"}

        # BFS on morphism links
        from collections import deque

        queue = deque([(from_id, [])])
        visited = {from_id}

        while queue:
            current, path = queue.popleft()
            if current == to_id:
                chain = []
                for step in path:
                    m_data = self.kg._graph.nodes[step["morphism"]]
                    chain.append(
                        {
                            "morphism": m_data.get("label", step["morphism"]),
                            "keeps": [
                                self.kg._graph.nodes[n].get("label", n)
                                for _, n, _ in self.kg._graph.out_edges(step["morphism"], data=True)
                                if self.kg._graph.edges[step["morphism"], n].get("relation") == "keeps"
                            ],
                            "loses": [
                                self.kg._graph.nodes[n].get("label", n)
                                for _, n, _ in self.kg._graph.out_edges(step["morphism"], data=True)
                                if self.kg._graph.edges[step["morphism"], n].get("relation") == "loses"
                            ],
                        }
                    )
                return {
                    "from": from_label,
                    "to": to_label,
                    "chain_length": len(chain),
                    "chain": chain,
                }

            for _, morph_id, edge_data in self.kg._graph.out_edges(current, data=True):
                if edge_data.get("relation") == "source_of":
                    for _, target_id, _ in self.kg._graph.out_edges(morph_id, data=True):
                        if target_id not in visited:
                            visited.add(target_id)
                            queue.append(
                                (
                                    target_id,
                                    path
                                    + [
                                        {
                                            "morphism": morph_id,
                                            "target": target_id,
                                        }
                                    ],
                                )
                            )

        return {"error": f"No morphism chain found from {from_label} to {to_label}"}

    # ── 等价性查询 ──

    def find_equivalents(self, param_name: str, target_engine: str) -> list[dict]:
        """查找某参数在目标引擎中的等价参数."""
        pid = node_id(param_name, "parameter")
        if pid not in self.kg._graph:
            return []

        equivalents = []
        for _, dst, edge_data in self.kg._graph.out_edges(pid, data=True):
            if edge_data.get("relation") == "equivalent_to":
                dst_data = self.kg._graph.nodes[dst]
                if target_engine.lower() in dst_data.get("label", "").lower():
                    equivalents.append(
                        {
                            "source_param": param_name,
                            "target_param": dst_data.get("label", dst),
                            "meaning": edge_data.get("meaning", ""),
                            "confidence": edge_data.get("confidence", 0.5),
                        }
                    )

        return equivalents

    # ── LLM 上下文生成 ──

    def to_llm_context(self, query_result: dict) -> str:
        """将查询结果转为 LLM prompt 友好的文本."""
        lines = ["# Knowledge Graph Query Result\n"]

        if "error" in query_result:
            lines.append(f"Error: {query_result['error']}")
            return "\n".join(lines)

        # 节点列表
        if "nodes" in query_result:
            nodes = query_result["nodes"]
            lines.append(f"## Nodes ({len(nodes)})\n")
            for n in nodes:
                label = n.get("label", n.get("id", "?"))
                ntype = n.get("type", "")
                lines.append(f"- [{ntype}] **{label}**")
                if n.get("canonical_form"):
                    lines.append(f"  - Form: {n['canonical_form']}")
                if n.get("expression"):
                    lines.append(f"  - Expression: {n['expression']}")

        # 边列表
        if "edges" in query_result:
            edges = query_result["edges"]
            lines.append(f"\n## Relations ({len(edges)})\n")
            for e in edges:
                src = e.get("source", "?")
                dst = e.get("target", "?")
                rel = e.get("relation", "→")
                lines.append(f"- {src} --[{rel}]→ {dst}")

        # 影响链
        if "impacted" in query_result:
            lines.append(f"\n## Impacted ({query_result.get('impacted_count', 0)})\n")
            for imp in query_result.get("impacted", []):
                lines.append(f"- {imp.get('target', '?')} ({imp.get('target_type', '')})")
                lines.append(f"  via: {imp.get('path', '')}")

        # 态射链
        if "chain" in query_result:
            lines.append(f"\n## Morphism Chain ({query_result.get('chain_length', 0)} steps)\n")
            for step in query_result.get("chain", []):
                lines.append(f"- **{step['morphism']}**")
                if step.get("keeps"):
                    lines.append(f"  keeps: {', '.join(step['keeps'])}")
                if step.get("loses"):
                    lines.append(f"  loses: {', '.join(step['loses'])}")

        return "\n".join(lines)
