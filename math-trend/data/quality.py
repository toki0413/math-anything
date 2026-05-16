"""
数据质量评估与置信度分层
"""

from collections import Counter
from typing import List, Dict
from datetime import datetime

from core.data_model import Paper
from core.config import AnalysisConfig


class QualityClassifier:
    """数据质量分类器"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.tier1_journals = set(config.tier1_journals)
    
    def classify(self, papers: List[Paper]) -> List[Paper]:
        """对论文进行置信度分层"""
        for paper in papers:
            paper.confidence = self._evaluate_confidence(paper)
            paper.confidence_reason = self._get_reason(paper)
        
        return papers
    
    def _evaluate_confidence(self, paper: Paper) -> str:
        score = 0
        
        # 期刊质量
        if paper.venue in self.tier1_journals:
            score += 3
        elif paper.venue:
            score += 1
        
        # 被引数
        if paper.citations >= 20:
            score += 3
        elif paper.citations >= 5:
            score += 2
        elif paper.citations >= 1:
            score += 1
        
        # 作者数（通常≥2表示合作研究）
        if len(paper.authors) >= 2:
            score += 1
        
        # 有DOI
        if paper.doi:
            score += 1
        
        # 有摘要
        if paper.abstract and len(paper.abstract) > 50:
            score += 1
        
        if score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
    
    def _get_reason(self, paper: Paper) -> str:
        reasons = []
        if paper.venue in self.tier1_journals:
            reasons.append("T1期刊")
        if paper.citations >= 20:
            reasons.append(f"高被引({paper.citations})")
        if paper.doi:
            reasons.append("有DOI")
        if not paper.abstract:
            reasons.append("无摘要")
        return "; ".join(reasons) if reasons else "综合评估"
