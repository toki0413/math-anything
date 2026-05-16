"""
跨领域对比分析 - Math-Trend核心卖点

功能：
- 目标领域 vs 对比领域的多维度对比
- 技术路线竞争分析
- 增长速度对比
- 研究成熟度对比

所有对比基于公开学术数据，不做无依据推测
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict
import numpy as np

from core.data_model import Paper, AnalysisResult
from core.config import AnalysisConfig
from core.interfaces import BaseAnalyzer


class CrossFieldComparisonAnalyzer(BaseAnalyzer):
    """跨领域对比分析"""
    
    COMPARISON_FIELDS = {
        'lithium_battery': {
            'name': '锂离子电池',
            'keywords': ['lithium-ion battery', 'Li-ion', 'lithium battery', 'LIB'],
        },
        'supercapacitor': {
            'name': '超级电容器',
            'keywords': ['supercapacitor', 'ultracapacitor', 'EDLC', 'electric double-layer'],
        },
        'solid_state_battery': {
            'name': '固态电池',
            'keywords': ['solid-state battery', 'solid electrolyte battery'],
        },
        'sodium_ion': {
            'name': '钠离子电池',
            'keywords': ['sodium-ion battery', 'Na-ion battery', 'sodium battery'],
        },
        'flow_battery': {
            'name': '液流电池',
            'keywords': ['flow battery', 'redox flow battery'],
        },
        'hydrogen_storage': {
            'name': '氢储能',
            'keywords': ['hydrogen storage', 'fuel cell', 'hydrogen energy'],
        },
    }
    
    def name(self) -> str:
        return "advanced.cross_field_comparison"
    
    def analyze(self, papers: List[Paper]) -> AnalysisResult:
        # 目标领域统计
        target_stats = self._field_stats(papers, "水泥基储能")
        
        # 对比领域统计
        comparisons = {}
        for field_key, field_info in self.COMPARISON_FIELDS.items():
            matching = self._find_matching_papers(papers, field_info['keywords'])
            if matching:
                comparisons[field_key] = {
                    'name': field_info['name'],
                    'stats': self._field_stats(matching, field_info['name']),
                }
        
        # 多维度对比
        comparison_table = self._build_comparison_table(target_stats, comparisons, papers)
        
        # 增长速度对比
        growth_comparison = self._compare_growth(papers, comparisons)
        
        return AnalysisResult(
            module_name=self.name(),
            timestamp=datetime.now(),
            success=True,
            data={
                "target_field": target_stats,
                "comparison_fields": {k: v['stats'] for k, v in comparisons.items()},
                "comparison_table": comparison_table,
                "growth_comparison": growth_comparison,
            }
        )
    
    def _find_matching_papers(self, papers: List[Paper], keywords: List[str]) -> List[Paper]:
        matching = []
        for p in papers:
            text = f"{p.title} {p.abstract}".lower()
            if any(kw.lower() in text for kw in keywords):
                matching.append(p)
        return matching
    
    def _field_stats(self, papers: List[Paper], name: str) -> Dict:
        if not papers:
            return {'name': name, 'paper_count': 0}
        
        years = [p.year for p in papers if p.year > 0]
        citations = [p.citations for p in papers]
        
        yearly = Counter(years)
        
        return {
            'name': name,
            'paper_count': len(papers),
            'year_range': f"{min(years)}-{max(years)}" if years else "N/A",
            'avg_citations': round(np.mean(citations), 1) if citations else 0,
            'max_citations': max(citations) if citations else 0,
            'yearly_distribution': dict(sorted(yearly.items())),
        }
    
    def _build_comparison_table(self, target: Dict, comparisons: Dict, all_papers: List[Paper]) -> List[Dict]:
        table = []
        
        # 目标领域行
        table.append({
            'field': target.get('name', '水泥基储能'),
            'papers': target.get('paper_count', 0),
            'avg_citations': target.get('avg_citations', 0),
            'max_citations': target.get('max_citations', 0),
            'is_target': True,
        })
        
        for key, comp in comparisons.items():
            stats = comp['stats']
            table.append({
                'field': stats.get('name', key),
                'papers': stats.get('paper_count', 0),
                'avg_citations': stats.get('avg_citations', 0),
                'max_citations': stats.get('max_citations', 0),
                'is_target': False,
            })
        
        table.sort(key=lambda x: x['papers'], reverse=True)
        return table
    
    def _compare_growth(self, papers: List[Paper], comparisons: Dict) -> Dict:
        # 目标领域增长
        target_yearly = Counter(p.year for p in papers if p.year > 0)
        target_growth = self._calc_growth(target_yearly)
        
        result = {
            'target': {
                'name': '水泥基储能',
                'recent_3yr_avg': target_growth.get('recent_avg', 0),
                'cagr': target_growth.get('cagr', 0),
            },
            'comparisons': {}
        }
        
        for key, comp in comparisons.items():
            comp_papers = self._find_matching_papers(papers, self.COMPARISON_FIELDS[key]['keywords'])
            comp_yearly = Counter(p.year for p in comp_papers if p.year > 0)
            comp_growth = self._calc_growth(comp_yearly)
            
            result['comparisons'][key] = {
                'name': comp['name'],
                'recent_3yr_avg': comp_growth.get('recent_avg', 0),
                'cagr': comp_growth.get('cagr', 0),
            }
        
        return result
    
    def _calc_growth(self, yearly: Dict) -> Dict:
        if not yearly:
            return {}
        
        years = sorted(yearly.keys())
        counts = [yearly[y] for y in years]
        
        recent_years = [y for y in years if y >= max(years) - 2]
        recent_avg = np.mean([yearly[y] for y in recent_years]) if recent_years else 0
        
        if len(years) >= 2 and counts[0] > 0:
            n_years = years[-1] - years[0]
            cagr = ((counts[-1] / counts[0]) ** (1 / max(n_years, 1)) - 1) * 100
        else:
            cagr = 0
        
        return {
            'recent_avg': round(recent_avg, 1),
            'cagr': round(cagr, 1),
        }


def get_analyzer(config: AnalysisConfig) -> BaseAnalyzer:
    return CrossFieldComparisonAnalyzer(config)
