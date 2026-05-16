"""
基础统计分析模块 - 插件化实现

功能：
- 论文规模统计
- 年度分布
- 期刊分布
- 引用分布
- 关键词频率
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List

from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig
from core.interfaces import BaseAnalyzer


class BasicStatisticsAnalyzer(BaseAnalyzer):
    """基础统计分析"""
    
    def name(self) -> str:
        return "basic.statistics"
    
    def analyze(self, papers: List[Paper]) -> AnalysisResult:
        if not papers:
            return AnalysisResult(
                module_name=self.name(),
                timestamp=datetime.now(),
                success=False,
                data={},
                error="无论文数据"
            )
        
        # 分层统计
        by_confidence = defaultdict(list)
        for p in papers:
            by_confidence[p.confidence].append(p)
        
        high = by_confidence.get('high', [])
        medium = by_confidence.get('medium', [])
        low = by_confidence.get('low', [])
        
        # 年度分布
        yearly = Counter(p.year for p in papers if p.year > 0)
        
        # 期刊分布
        venue_counts = Counter(p.venue for p in papers if p.venue)
        
        # 关键词统计
        keyword_counts = Counter()
        for p in papers:
            for kw in p.keywords:
                keyword_counts[kw] += 1
        
        # 高被引
        sorted_by_citation = sorted(papers, key=lambda x: x.citations, reverse=True)
        top_10 = sorted_by_citation[:10]
        
        result_data = {
            "scale": {
                "total": len(papers),
                "high_confidence": len(high),
                "medium_confidence": len(medium),
                "low_confidence": len(low),
                "year_range": f"{min(p.year for p in papers if p.year > 0)}-{max(p.year for p in papers if p.year > 0)}",
                "unique_venues": len(venue_counts),
                "total_citations": sum(p.citations for p in papers),
                "avg_citations": round(sum(p.citations for p in papers) / len(papers), 1),
            },
            "yearly_distribution": dict(sorted(yearly.items())),
            "top_venues": dict(venue_counts.most_common(10)),
            "top_keywords": dict(keyword_counts.most_common(20)),
            "top_10_cited": [
                {"title": p.title, "citations": p.citations, "year": p.year, "venue": p.venue}
                for p in top_10
            ],
        }
        
        return AnalysisResult(
            module_name=self.name(),
            timestamp=datetime.now(),
            success=True,
            data=result_data
        )


def get_analyzer(config: AnalysisConfig) -> BaseAnalyzer:
    return BasicStatisticsAnalyzer(config)
