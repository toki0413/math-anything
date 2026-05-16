"""
统一数据加载器 - 为所有math-trend模块提供标准化的数据接口

功能:
1. 加载分类后的608篇数据（带置信度）
2. 从多源API获取真实数据
3. 统一数据格式
4. 置信度分层管理
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 数据路径
OUTPUT_DIR = Path(__file__).parent / "output" / "analysis_608"
CLASSIFIED_DATA_PATH = OUTPUT_DIR / "classified_papers.json"


@dataclass
class Paper:
    """标准化论文数据结构"""
    id: str
    title: str
    authors: List[str]
    year: int
    citations: int
    abstract: str
    venue: str
    doi: str
    source: str
    url: str
    confidence: str  # high/medium/low
    confidence_reason: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Paper":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year", 0),
            citations=data.get("citations", 0),
            abstract=data.get("abstract", ""),
            venue=data.get("venue", ""),
            doi=data.get("doi", ""),
            source=data.get("source", ""),
            url=data.get("url", ""),
            confidence=data.get("confidence", "medium"),
            confidence_reason=data.get("confidence_reason", "")
        )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "citations": self.citations,
            "abstract": self.abstract,
            "venue": self.venue,
            "doi": self.doi,
            "source": self.source,
            "url": self.url,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason
        }


class UnifiedDataLoader:
    """统一数据加载器"""
    
    def __init__(self):
        self.high_confidence_papers: List[Paper] = []
        self.medium_confidence_papers: List[Paper] = []
        self.low_confidence_papers: List[Paper] = []
        self.all_papers: List[Paper] = []
        self._loaded = False
    
    def load_classified_data(self, data_path: Optional[Path] = None) -> Dict:
        """加载已分类的数据"""
        path = data_path or CLASSIFIED_DATA_PATH
        
        if not path.exists():
            raise FileNotFoundError(f"分类数据文件不存在: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 加载各置信度层级
        self.high_confidence_papers = [
            Paper.from_dict(p) for p in data["papers"]["high_confidence"]
        ]
        self.medium_confidence_papers = [
            Paper.from_dict(p) for p in data["papers"]["medium_confidence"]
        ]
        self.low_confidence_papers = [
            Paper.from_dict(p) for p in data["papers"]["low_confidence"]
        ]
        
        self.all_papers = (
            self.high_confidence_papers + 
            self.medium_confidence_papers + 
            self.low_confidence_papers
        )
        
        self._loaded = True
        
        return {
            "high": self.high_confidence_papers,
            "medium": self.medium_confidence_papers,
            "low": self.low_confidence_papers,
            "all": self.all_papers,
            "core": self.high_confidence_papers + self.medium_confidence_papers
        }
    
    def get_papers_by_confidence(self, level: str) -> List[Paper]:
        """按置信度获取论文"""
        if not self._loaded:
            self.load_classified_data()
        
        level_map = {
            "high": self.high_confidence_papers,
            "medium": self.medium_confidence_papers,
            "low": self.low_confidence_papers,
            "all": self.all_papers,
            "core": self.high_confidence_papers + self.medium_confidence_papers
        }
        return level_map.get(level, [])
    
    def get_papers_by_year_range(self, year_from: int, year_to: int, 
                                  confidence: str = "all") -> List[Paper]:
        """按年份范围获取论文"""
        papers = self.get_papers_by_confidence(confidence)
        return [p for p in papers if year_from <= p.year <= year_to]
    
    def get_papers_by_keyword(self, keywords: List[str], 
                               confidence: str = "all",
                               match_all: bool = False) -> List[Paper]:
        """按关键词筛选论文"""
        papers = self.get_papers_by_confidence(confidence)
        results = []
        
        for p in papers:
            text = f"{p.title} {p.abstract}".lower()
            if match_all:
                if all(kw.lower() in text for kw in keywords):
                    results.append(p)
            else:
                if any(kw.lower() in text for kw in keywords):
                    results.append(p)
        
        return results
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self._loaded:
            self.load_classified_data()
        
        stats = {
            "total": len(self.all_papers),
            "high_confidence": len(self.high_confidence_papers),
            "medium_confidence": len(self.medium_confidence_papers),
            "low_confidence": len(self.low_confidence_papers),
            "year_range": {
                "min": min(p.year for p in self.all_papers if p.year > 0),
                "max": max(p.year for p in self.all_papers if p.year > 0)
            },
            "total_citations": sum(p.citations for p in self.all_papers),
            "avg_citations": sum(p.citations for p in self.all_papers) / len(self.all_papers),
            "source_distribution": self._get_source_distribution(),
            "year_distribution": self._get_year_distribution()
        }
        
        return stats
    
    def _get_source_distribution(self) -> Dict[str, int]:
        """数据源分布"""
        dist = defaultdict(int)
        for p in self.all_papers:
            dist[p.source] += 1
        return dict(dist)
    
    def _get_year_distribution(self) -> Dict[int, int]:
        """年份分布"""
        dist = defaultdict(int)
        for p in self.all_papers:
            if p.year > 0:
                dist[p.year] += 1
        return dict(sorted(dist.items()))
    
    def get_top_cited(self, n: int = 10, confidence: str = "all") -> List[Paper]:
        """获取高被引论文"""
        papers = self.get_papers_by_confidence(confidence)
        return sorted(papers, key=lambda p: p.citations, reverse=True)[:n]
    
    def get_top_authors(self, n: int = 10, confidence: str = "all") -> List[Tuple[str, int, int]]:
        """获取高产作者 (作者名, 论文数, 总被引)"""
        papers = self.get_papers_by_confidence(confidence)
        
        author_papers = defaultdict(list)
        author_citations = defaultdict(int)
        
        for p in papers:
            for author in p.authors[:5]:  # 只考虑前5作者
                if author:
                    author_papers[author].append(p)
                    author_citations[author] += p.citations
        
        sorted_authors = sorted(
            [(name, len(papers), author_citations[name])
             for name, papers in author_papers.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_authors[:n]


def load_cement_storage_data() -> UnifiedDataLoader:
    """便捷函数：加载水泥储能数据"""
    loader = UnifiedDataLoader()
    loader.load_classified_data()
    return loader


if __name__ == "__main__":
    # 测试数据加载器
    print("测试统一数据加载器...")
    
    loader = load_cement_storage_data()
    stats = loader.get_statistics()
    
    print(f"\n数据加载成功!")
    print(f"  总计: {stats['total']}篇")
    print(f"  高置信度: {stats['high_confidence']}篇")
    print(f"  中置信度: {stats['medium_confidence']}篇")
    print(f"  低置信度: {stats['low_confidence']}篇")
    print(f"  年份范围: {stats['year_range']['min']}-{stats['year_range']['max']}")
    print(f"  总被引: {stats['total_citations']}")
    print(f"  平均被引: {stats['avg_citations']:.1f}")
    
    print(f"\n数据源分布:")
    for source, count in stats['source_distribution'].items():
        print(f"  {source}: {count}篇")
    
    print(f"\nTop 5 高被引论文:")
    for i, p in enumerate(loader.get_top_cited(5), 1):
        print(f"  {i}. {p.title[:50]}... ({p.citations}被引)")
    
    print(f"\nTop 5 高产作者:")
    for i, (author, count, citations) in enumerate(loader.get_top_authors(5), 1):
        print(f"  {i}. {author}: {count}篇, {citations}被引")
