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

    def betti_numbers(self) -> dict[str, int]:
        """Return Betti numbers of the underlying undirected graph.

        β0 = number of connected components.
        β1 = number of independent cycles (first Betti number).
        """
        graph = self.build_graph().to_undirected()
        if graph.number_of_nodes() == 0:
            return {"beta0": 0, "beta1": 0}

        beta0 = nx.number_connected_components(graph)
        # β1 = E - V + C for each component summed.
        beta1 = graph.number_of_edges() - graph.number_of_nodes() + beta0
        return {"beta0": beta0, "beta1": beta1}

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
        except Exception:
            # NetworkX may raise on unusual graph configurations; treat as no loops.
            return []
        if not cycles:
            return []

        loops: list[Loop] = []
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Order cycle nodes by walking the undirected graph.
            ordered = self._order_cycle(undirected, cycle)
            if ordered is None:
                continue

            forward = self._edges_for_orientation(digraph, ordered)
            backward = self._edges_for_orientation(digraph, list(reversed(ordered)))
            if forward is not None:
                edges, is_directed = forward, True
                normalized = self._normalize_cycle(ordered)
            elif backward is not None:
                edges, is_directed = backward, True
                normalized = self._normalize_cycle(list(reversed(ordered)))
            else:
                edges, is_directed = self._edges_for_orientation(undirected, ordered), False
                normalized = self._normalize_cycle(ordered)

            loops.append(
                Loop(
                    nodes=tuple(normalized),
                    edges=tuple(edges),
                    is_directed=is_directed,
                    canonical_form=" -> ".join(normalized),
                )
            )
        return loops

    def _order_cycle(self, graph: nx.Graph, cycle_nodes: list[str]) -> list[str] | None:
        """Return a closed walk ordering of nodes in an undirected cycle."""
        start = sorted(cycle_nodes)[0]
        ordered = [start]
        visited = {start}
        current = start
        while len(visited) < len(cycle_nodes):
            neighbors = sorted(n for n in graph.neighbors(current) if n in cycle_nodes and n not in visited)
            if not neighbors:
                return None
            nxt = neighbors[0]
            ordered.append(nxt)
            visited.add(nxt)
            current = nxt
        # Close the loop if start is a neighbor of the last node.
        if start not in graph.neighbors(current):
            return None
        ordered.append(start)
        return ordered

    def _edges_for_orientation(self, graph: nx.MultiDiGraph | nx.MultiGraph, ordered: list[str]) -> list[str] | None:
        """Return edge keys if every consecutive pair has an edge in the given order."""
        edges: list[str] = []
        for u, v in zip(ordered, ordered[1:]):
            if not graph.has_edge(u, v):
                return None
            edges.append(sorted(graph[u][v])[0])
        return edges

    def _normalize_cycle(self, ordered: list[str]) -> list[str]:
        """Return a rotation/reversal-invariant form of a closed walk."""
        # Drop the repeated start at the end for normalization.
        nodes = ordered[:-1]
        rotations = [nodes[i:] + nodes[:i] for i in range(len(nodes))]
        reversals = [list(reversed(r)) for r in rotations]
        candidates = [tuple(r) for r in rotations + reversals]
        best = min(candidates)
        return list(best) + [best[0]]
