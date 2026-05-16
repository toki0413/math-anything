"""
引用网络分析 - 插件化实现

功能：
- 构建论文关联网络（合作+语义+期刊）
- PageRank中心性
- 社区发现
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict
import numpy as np

from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig
from core.interfaces import BaseAnalyzer


class CitationNetworkAnalyzer(BaseAnalyzer):
    """引用网络分析"""
    
    def name(self) -> str:
        return "advanced.citation_network"
    
    def analyze(self, papers: List[Paper]) -> AnalysisResult:
        # 构建邻接表
        adjacency = defaultdict(lambda: defaultdict(float))
        
        # 合作网络
        for p in papers:
            authors = p.authors[:3] if p.authors else []
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    adjacency[a1][a2] += 0.4
                    adjacency[a2][a1] += 0.4
        
        # 语义网络（关键词共现）
        for p in papers:
            kws = p.keywords or []
            for i, kw1 in enumerate(kws):
                for kw2 in kws[i+1:]:
                    adjacency[kw1][kw2] += 0.3
                    adjacency[kw2][kw1] += 0.3
        
        # PageRank
        pagerank = self._pagerank(adjacency)
        
        # 统计
        node_count = len(adjacency)
        edge_count = sum(len(v) for v in adjacency.values()) // 2
        
        # Top节点
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 识别社区（简化：基于度中心性分组）
        communities = self._detect_communities(adjacency)
        
        return AnalysisResult(
            module_name=self.name(),
            timestamp=datetime.now(),
            success=True,
            data={
                "network_stats": {
                    "nodes": node_count,
                    "edges": edge_count,
                    "density": round(2 * edge_count / (node_count * (node_count - 1)), 4) if node_count > 1 else 0,
                },
                "top_pagerank": [{"node": n, "score": round(s, 4)} for n, s in top_nodes],
                "communities": len(communities),
                "community_details": communities[:5],
            }
        )
    
    def _pagerank(self, adjacency: Dict, damping: float = 0.85, iterations: int = 50) -> Dict[str, float]:
        nodes = list(adjacency.keys())
        if not nodes:
            return {}
        
        n = len(nodes)
        scores = {node: 1.0 / n for node in nodes}
        
        for _ in range(iterations):
            new_scores = {}
            for node in nodes:
                rank = (1 - damping) / n
                for other_node in nodes:
                    if node in adjacency[other_node]:
                        out_degree = len(adjacency[other_node])
                        if out_degree > 0:
                            rank += damping * scores[other_node] / out_degree
                new_scores[node] = rank
            scores = new_scores
        
        return scores
    
    def _detect_communities(self, adjacency: Dict) -> List[Dict]:
        # 简化社区发现：基于度中心性
        degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        communities = []
        visited = set()
        
        for node, degree in sorted_nodes:
            if node in visited or degree < 2:
                continue
            
            community = [node]
            visited.add(node)
            
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    community.append(neighbor)
                    visited.add(neighbor)
            
            if len(community) >= 2:
                communities.append({
                    'hub': node,
                    'size': len(community),
                    'members': community[:5],
                })
        
        communities.sort(key=lambda x: x['size'], reverse=True)
        return communities


def get_analyzer(config: AnalysisConfig) -> BaseAnalyzer:
    return CitationNetworkAnalyzer(config)
