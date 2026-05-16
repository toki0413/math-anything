"""
主题模型分析 - 插件化实现

功能：
- 基于关键词共现的主题聚类
- 新兴主题检测
- 主题演化追踪
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig
from core.interfaces import BaseAnalyzer


class TopicModelingAnalyzer(BaseAnalyzer):
    """主题模型分析（基于关键词共现）"""
    
    def name(self) -> str:
        return "advanced.topic_modeling"
    
    def analyze(self, papers: List[Paper]) -> AnalysisResult:
        # 构建关键词共现矩阵
        cooccurrence = defaultdict(lambda: defaultdict(int))
        keyword_counts = Counter()
        
        for p in papers:
            kws = p.keywords if p.keywords else []
            keyword_counts.update(kws)
            for i, kw1 in enumerate(kws):
                for kw2 in kws[i+1:]:
                    cooccurrence[kw1][kw2] += 1
                    cooccurrence[kw2][kw1] += 1
        
        # 基于共现强度聚类
        clusters = self._cluster_by_cooccurrence(cooccurrence, keyword_counts)
        
        # 新兴主题检测
        emerging = self._detect_emerging(papers)
        
        # 主题演化
        evolution = self._track_evolution(papers)
        
        return AnalysisResult(
            module_name=self.name(),
            timestamp=datetime.now(),
            success=True,
            data={
                "keyword_counts": dict(keyword_counts.most_common(20)),
                "topics": clusters[:8],
                "emerging_topics": emerging,
                "topic_evolution": evolution,
            }
        )
    
    def _cluster_by_cooccurrence(self, cooccurrence, keyword_counts) -> List[Dict]:
        top_keywords = [kw for kw, _ in keyword_counts.most_common(30)]
        
        # 简单聚类：找共现最强的关键词组
        visited = set()
        clusters = []
        
        for kw in top_keywords:
            if kw in visited:
                continue
            
            cluster_members = [kw]
            visited.add(kw)
            
            # 找与kw共现最强的词
            if kw in cooccurrence:
                neighbors = sorted(cooccurrence[kw].items(), key=lambda x: x[1], reverse=True)
                for neighbor, count in neighbors[:5]:
                    if neighbor not in visited and neighbor in top_keywords and count >= 3:
                        cluster_members.append(neighbor)
                        visited.add(neighbor)
            
            if len(cluster_members) >= 2:
                clusters.append({
                    'core_keyword': kw,
                    'members': cluster_members,
                    'member_count': len(cluster_members),
                    'total_papers': keyword_counts.get(kw, 0),
                })
        
        clusters.sort(key=lambda x: x['total_papers'], reverse=True)
        return clusters
    
    def _detect_emerging(self, papers: List[Paper]) -> List[Dict]:
        recent = [p for p in papers if p.year >= 2024]
        older = [p for p in papers if p.year < 2024]
        
        recent_kw = Counter()
        older_kw = Counter()
        
        for p in recent:
            recent_kw.update(p.keywords or [])
        for p in older:
            older_kw.update(p.keywords or [])
        
        emerging = []
        for kw, count in recent_kw.items():
            old_count = older_kw.get(kw, 0)
            if old_count == 0 and count >= 3:
                emerging.append({
                    'keyword': kw,
                    'recent_count': count,
                    'older_count': old_count,
                    'status': '新出现',
                })
            elif old_count > 0 and count / max(len(recent), 1) > old_count / max(len(older), 1) * 2:
                emerging.append({
                    'keyword': kw,
                    'recent_count': count,
                    'older_count': old_count,
                    'status': '快速增长',
                })
        
        emerging.sort(key=lambda x: x['recent_count'], reverse=True)
        return emerging[:10]
    
    def _track_evolution(self, papers: List[Paper]) -> Dict:
        yearly_kw = defaultdict(Counter)
        for p in papers:
            if p.year > 0:
                yearly_kw[p.year].update(p.keywords or [])
        
        evolution = {}
        for year in sorted(yearly_kw.keys()):
            evolution[str(year)] = dict(yearly_kw[year].most_common(5))
        
        return evolution


def get_analyzer(config: AnalysisConfig) -> BaseAnalyzer:
    return TopicModelingAnalyzer(config)
