"""
跨领域知识迁移检测 - 插件化实现

功能：
- 检测从其他领域迁移到目标领域的概念
- 识别知识传播延迟
- 分析语义桥接
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Optional

from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig
from core.interfaces import BaseAnalyzer


class CrossDomainAnalyzer(BaseAnalyzer):
    """跨领域知识迁移检测"""
    
    SOURCE_CONCEPTS = {
        'battery': ['lithium', 'Li-ion', 'battery', 'anode', 'cathode', 'electrolyte'],
        'supercapacitor': ['EDLC', 'electric double layer', 'supercapacitor', 'pseudocapacitance'],
        'nanotechnology': ['nanoparticle', 'nanocomposite', 'nanotube', 'nanostructure'],
        'sensor': ['strain sensor', 'pressure sensor', 'self-sensing', 'piezoresistive'],
        'thermal': ['thermal conductivity', 'heat storage', 'phase change', 'thermal energy'],
    }
    
    def name(self) -> str:
        return "basic.cross_domain"
    
    def analyze(self, papers: List[Paper]) -> AnalysisResult:
        transfers = self._detect_transfers(papers)
        timeline = self._get_transfer_timeline(transfers, papers)
        
        high_confidence = [t for t in transfers if t.get('confidence') == 'high']
        medium_confidence = [t for t in transfers if t.get('confidence') == 'medium']
        
        return AnalysisResult(
            module_name=self.name(),
            timestamp=datetime.now(),
            success=True,
            data={
                "total_transfers": len(transfers),
                "high_confidence_transfers": len(high_confidence),
                "medium_confidence_transfers": len(medium_confidence),
                "transfers": transfers[:10],
                "timeline": {str(k): v for k, v in timeline.items()},
                "source_distribution": dict(Counter(t['source_domain'] for t in transfers)),
            }
        )
    
    def _detect_transfers(self, papers: List[Paper]) -> List[Dict]:
        transfers = []
        
        for source_domain, keywords in self.SOURCE_CONCEPTS.items():
            matching_papers = []
            for p in papers:
                text = f"{p.title} {p.abstract}".lower()
                if any(kw.lower() in text for kw in keywords):
                    matching_papers.append(p)
            
            if not matching_papers:
                continue
            
            # 检测首次出现年份
            years = sorted(set(p.year for p in matching_papers if p.year > 0))
            if not years:
                continue
            
            first_year = years[0]
            peak_year = max(years, key=lambda y: sum(1 for p in matching_papers if p.year == y))
            paper_count = len(matching_papers)
            
            # 置信度评估
            confidence = 'low'
            if paper_count >= 5 and len(years) >= 3:
                confidence = 'high'
            elif paper_count >= 3:
                confidence = 'medium'
            
            transfers.append({
                'source_domain': source_domain,
                'paper_count': paper_count,
                'first_appearance': first_year,
                'peak_year': peak_year,
                'year_span': len(years),
                'confidence': confidence,
                'sample_keywords': keywords[:3],
            })
        
        transfers.sort(key=lambda x: x['paper_count'], reverse=True)
        return transfers
    
    def _get_transfer_timeline(self, transfers: List[Dict], papers: List[Paper]) -> Dict:
        timeline = defaultdict(list)
        for t in transfers:
            timeline[t['first_appearance']].append(t['source_domain'])
        return dict(sorted(timeline.items()))


def get_analyzer(config: AnalysisConfig) -> BaseAnalyzer:
    return CrossDomainAnalyzer(config)
