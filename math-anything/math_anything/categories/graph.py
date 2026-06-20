"""范畴知识图谱 — 持久化层的图结构。

将范畴论结构（对象、态射、函子）存储为 JSON-LD 兼容的有向图。
继承并扩展已有的 graph.jsonl 格式。

节点标签规范：
  Structure:     "struct:NonlinearEigenvalueProblem"
  Morphism:      "morph:BornOppenheimer"
  Engine:        "engine:VASP"  (与已有格式兼容)
  Parameter:     "param:ENCUT"
  Invariant:     "inv:self_adjointness"
  PiGroup:       "pi:Re=ρUL/μ"

关系标签规范：
  instantiated_by:  Structure ← Engine
  has_morphism:     Structure --[morphism]→ Structure
  controls:         Parameter → Discretization
  constrains:       Invariant → Parameter
  equivalent_to:    Parameter → Parameter (cross-engine)
  keeps/loses:      Morphism → Invariant
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

# ── 节点/关系工具 ──

ENTITY_PREFIXES = {
    "structure": "struct",
    "morphism": "morph",
    "engine": "engine",
    "parameter": "param",
    "equation": "eq",
    "invariant": "inv",
    "pi_group": "pi",
    "approximation": "approx",
    "discretization": "disc",
}


def node_id(label: str, entity_type: str) -> str:
    prefix = ENTITY_PREFIXES.get(entity_type, entity_type.lower())
    return f"{prefix}:{label.strip()}"


def now_iso() -> str:
    return datetime.now().isoformat()


# ── 知识图谱类 ──


class MathKnowledgeGraph:
    """数学知识图谱：持久化 + 可查询.

    继承已有 graph.jsonl 的格式，提供结构化的节点和边管理。
    """

    FILENAME = "math_knowledge_graph.json"

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self._graph = nx.DiGraph()

        # 尝试加载已有数据
        self._load_existing_jsonl()

    def _load_existing_jsonl(self) -> None:
        """从已有的 graph.jsonl 加载数据."""
        jsonl_path = self.root / "ontology" / "graph.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
                try:
                    record = json.loads(line)
                    if record.get("op") == "create":
                        entity = record["entity"]
                        eid = entity["id"]
                        props = entity.get("properties", {})
                        self._graph.add_node(
                            eid,
                            label=props.get("title", props.get("name", eid)),
                            type=entity.get("type", "Unknown"),
                            source="graph.jsonl",
                            confidence=0.9,
                        )
                    elif record.get("op") == "relate":
                        self._graph.add_edge(
                            record["from"],
                            record["to"],
                            relation=record["rel"],
                            source="graph.jsonl",
                        )
                except (json.JSONDecodeError, KeyError):
                    continue

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._graph = nx.node_link_graph(data, directed=True, multigraph=False)

    def save(self) -> None:
        data = nx.node_link_data(self._graph)
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── 节点操作 ──

    def add_entity(
        self,
        label: str,
        entity_type: str,
        *,
        source: str = "auto",
        confidence: float = 0.5,
        **attrs: Any,
    ) -> str:
        eid = node_id(label, entity_type)
        ts = now_iso()

        if eid in self._graph:
            self._graph.nodes[eid]["mentions"] = self._graph.nodes[eid].get("mentions", 0) + 1
            self._graph.nodes[eid]["last_seen"] = ts
            old_conf = self._graph.nodes[eid].get("confidence", confidence)
            self._graph.nodes[eid]["confidence"] = min(0.99, old_conf + 0.05)
        else:
            self._graph.add_node(
                eid,
                label=label,
                type=entity_type,
                source=source,
                confidence=confidence,
                created_at=ts,
                last_seen=ts,
                mentions=1,
                **self._normalize_props(attrs),
            )
        return eid

    def add_relation(
        self,
        src_id: str,
        relation: str,
        dst_id: str,
        *,
        source: str = "auto",
        confidence: float = 0.5,
        **attrs: Any,
    ) -> None:
        if src_id not in self._graph or dst_id not in self._graph:
            return
        ts = now_iso()
        if self._graph.has_edge(src_id, dst_id):
            data = self._graph.edges[src_id, dst_id]
            data["mentions"] = data.get("mentions", 0) + 1
            data["last_seen"] = ts
            old_conf = data.get("confidence", confidence)
            data["confidence"] = min(0.99, old_conf + 0.05)
        else:
            self._graph.add_edge(
                src_id,
                dst_id,
                relation=relation,
                source=source,
                confidence=confidence,
                created_at=ts,
                last_seen=ts,
                mentions=1,
                **self._normalize_props(attrs),
            )

    def has_entity(self, label: str, entity_type: str) -> bool:
        return node_id(label, entity_type) in self._graph

    def get_entity(self, label: str, entity_type: str) -> dict | None:
        eid = node_id(label, entity_type)
        if eid in self._graph:
            return dict(self._graph.nodes[eid])
        return None

    # ── 查询操作 ──

    def find_nodes(self, seed: str, entity_type: str | None = None, top_k: int = 5) -> list[str]:
        """按标签/类型查找节点."""
        seed_lower = seed.lower()
        matches: list[tuple[tuple, str]] = []

        for nid, data in self._graph.nodes(data=True):
            label = data.get("label", "").lower()
            ntype = data.get("type", "")

            if entity_type and ntype != entity_type:
                continue

            if seed_lower in label:
                exact = label == seed_lower
                score = (int(exact), data.get("confidence", 0), data.get("mentions", 0))
                matches.append((score, nid))

        matches.sort(reverse=True)
        return [nid for _, nid in matches[:top_k]]

    def neighborhood(self, seed_nodes: list[str], depth: int = 1, top_k: int = 20) -> dict:
        """从种子节点扩展邻域."""
        visited: set[str] = set(seed_nodes)
        frontier = set(seed_nodes)

        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self._graph.successors(node))
                next_frontier.update(self._graph.predecessors(node))
            frontier = next_frontier - visited
            visited.update(frontier)

        if len(visited) > top_k:
            scored = [(self._graph.nodes[n].get("confidence", 0), n) for n in visited]
            scored.sort(reverse=True)
            visited = {n for _, n in scored[:top_k]}

        sub = self._graph.subgraph(visited)
        return {
            "nodes": [{"id": n, **self._graph.nodes[n]} for n in sub.nodes()],
            "edges": [{"source": u, "target": v, **d} for u, v, d in sub.edges(data=True)],
        }

    def query(self, text: str, depth: int = 1, top_k: int = 20) -> dict:
        """高层查询接口."""
        seeds = self.find_nodes(text)
        if not seeds:
            seeds = self.find_nodes(text, entity_type=None)
        if not seeds:
            return {"nodes": [], "edges": []}
        return self.neighborhood(seeds, depth=depth, top_k=top_k)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            t = data.get("type", "Unknown")
            counts[t] = counts.get(t, 0) + 1
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "node_types": counts,
        }

    def to_context_string(self, node_ids: set[str]) -> str:
        """将子图转为 LLM 友好的文本摘要."""
        lines: list[str] = []
        for eid in sorted(node_ids):
            data = self._graph.nodes[eid]
            label = data.get("label", eid)
            etype = data.get("type", "")
            lines.append(f"- [{etype}] {label}")
            for _, dst, edge_data in self._graph.out_edges(eid, data=True):
                dst_label = self._graph.nodes[dst].get("label", dst)
                rel = edge_data.get("relation", "→")
                lines.append(f"  {rel} → {dst_label}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_props(props: dict) -> dict:
        return {k: v.strip() if isinstance(v, str) else v for k, v in props.items() if v is not None}
