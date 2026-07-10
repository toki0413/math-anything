"""Loop detection engine over CategoryEngine morphism graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from .loop import Loop

if TYPE_CHECKING:
    from math_anything.categories.engine import CategoryEngine


@dataclass
class LoopEngine:
    """Detects loops in a CategoryEngine morphism graph."""

    category_engine: "CategoryEngine"

    def build_graph(self) -> nx.MultiDiGraph:
        """Build a directed multi-graph from registered morphism links."""
        graph = nx.MultiDiGraph()
        for link in self.category_engine.morphism_links:
            graph.add_edge(
                link.source_structure,
                link.target_structure,
                key=link.morphism.name,
                morphism=link.morphism,
            )
        return graph

    def find_loops(self) -> list[Loop]:
        """Return a cycle basis of the underlying undirected graph as Loop objects."""
        digraph = self.build_graph()
        if digraph.number_of_nodes() == 0:
            return []

        undirected = digraph.to_undirected()
        # minimum_cycle_basis does not support multigraphs, so compute on a
        # simple undirected view while keeping the original edges for lookups.
        simple = nx.Graph(undirected)
        try:
            cycles = nx.minimum_cycle_basis(simple)
        except nx.NetworkXNoCycle:
            return []

        loops: list[Loop] = []
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Order cycle nodes by walking the undirected graph.
            ordered = self._order_cycle(undirected, cycle)
            if ordered is None:
                continue

            edges: list[str] = []
            is_directed = True
            for u, v in zip(ordered, ordered[1:]):
                # Prefer directed edge key if it exists.
                key = None
                if digraph.has_edge(u, v):
                    key = next(iter(digraph[u][v]))
                elif digraph.has_edge(v, u):
                    key = next(iter(digraph[v][u]))
                    is_directed = False
                else:
                    key = next(iter(undirected[u][v]))
                    is_directed = False
                edges.append(key)

            loops.append(
                Loop(
                    nodes=tuple(ordered),
                    edges=tuple(edges),
                    is_directed=is_directed,
                    canonical_form=" -> ".join(ordered),
                )
            )
        return loops

    def _order_cycle(
        self, graph: nx.Graph, cycle_nodes: list[str]
    ) -> list[str] | None:
        """Return a closed walk ordering of nodes in an undirected cycle."""
        start = cycle_nodes[0]
        ordered = [start]
        visited = {start}
        current = start
        while len(visited) < len(cycle_nodes):
            neighbors = [n for n in graph.neighbors(current) if n in cycle_nodes and n not in visited]
            if not neighbors:
                return None
            nxt = neighbors[0]
            ordered.append(nxt)
            visited.add(nxt)
            current = nxt
        # Close the loop if start is a neighbor of the last node.
        if start not in list(graph.neighbors(current)):
            return None
        ordered.append(start)
        return ordered
