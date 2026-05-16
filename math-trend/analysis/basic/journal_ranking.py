"""
期刊数学驱动排名 - 插件化实现

功能：
- 5维指标评估期刊
- 基尼系数（影响力扩散）
- 时间衰减因子
- 跨学科度
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict
import numpy as np

from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig
from core.interfaces import BaseAnalyzer


class JournalRankingAnalyzer(BaseAnalyzer):
    """期刊数学驱动排名"""
    
    def name(self) -> str:
        return "basic.journal_ranking"
    
    def analyze(self, papers: List[Paper]) -> AnalysisResult:
        venue_data = defaultdict(lambda: {'papers': [], 'citations': []})
        
        for p in papers:
            if not p.venue:
                continue
            venue_data[p.venue]['papers'].append(p)
            venue_data[p.venue]['citations'].append(p.citations)
        
        # 过滤只发1篇的期刊
        venues = {v: d for v, d in venue_data.items() if len(d['papers']) >= 2}
        
        if not venues:
            return AnalysisResult(
                module_name=self.name(),
                timestamp=datetime.now(),
                success=True,
                data={"rankings": [], "note": "期刊数不足（需≥2篇）"}
            )
        
        rankings = []
        for venue, data in venues.items():
            n_papers = len(data['papers'])
            citations = data['citations']
            
            # 维度1: 规模（论文数）
            scale = n_papers
            
            # 维度2: 影响力（平均被引）
            avg_citation = np.mean(citations)
            
            # 维度3: 影响力扩散（基尼系数，越低越均匀）
            gini = self._gini_coefficient(citations)
            
            # 维度4: 时间活跃度
            years = [p.year for p in data['papers'] if p.year > 0]
            year_span = max(years) - min(years) if years else 0
            
            # 维度5: 高置信度论文占比
            high_ratio = sum(1 for p in data['papers'] if p.confidence == 'high') / n_papers
            
            # 综合得分
            score = (
                np.log1p(scale) * 10 +
                np.log1p(avg_citation) * 15 +
                (1 - gini) * 20 +
                min(year_span / 5, 1) * 15 +
                high_ratio * 40
            )
            
            rankings.append({
                'venue': venue,
                'papers': n_papers,
                'avg_citations': round(avg_citation, 1),
                'gini': round(gini, 3),
                'year_span': year_span,
                'high_confidence_ratio': round(high_ratio, 2),
                'score': round(score, 1),
            })
        
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # 分级
        for i, r in enumerate(rankings):
            percentile = (i + 1) / len(rankings) * 100
            if percentile <= 25:
                r['tier'] = 'T1'
            elif percentile <= 50:
                r['tier'] = 'T2'
            elif percentile <= 75:
                r['tier'] = 'T3'
            else:
                r['tier'] = 'T4'
        
        return AnalysisResult(
            module_name=self.name(),
            timestamp=datetime.now(),
            success=True,
            data={
                "total_venues": len(venues),
                "rankings": rankings[:20],
                "tier_distribution": dict(Counter(r['tier'] for r in rankings)),
            }
        )
    
    def _gini_coefficient(self, values: List[float]) -> float:
        if not values or len(values) < 2:
            return 0
        sorted_vals = sorted(values)
        total = sum(sorted_vals)
        if total == 0:
            return 0
        n = len(sorted_vals)
        return (2 * sum((i + 1) * v for i, v in enumerate(sorted_vals))) / (n * total) - (n + 1) / n


def get_analyzer(config: AnalysisConfig) -> BaseAnalyzer:
    return JournalRankingAnalyzer(config)
