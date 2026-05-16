"""
Math-Trend 统一数据模型

定义核心数据结构，避免各模块重复定义
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


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
    keywords: List[str] = None  # 预处理后添加
    topics: List[str] = None   # 主题分析后添加
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.topics is None:
            self.topics = []
    
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
            confidence_reason=data.get("confidence_reason", ""),
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
            "confidence_reason": self.confidence_reason,
            "keywords": self.keywords,
            "topics": self.topics,
        }


@dataclass
class DataSource:
    """数据源信息"""
    name: str
    total_papers: int
    success_rate: float
    last_updated: datetime
    metadata: Dict = None


@dataclass
class AnalysisResult:
    """分析结果统一格式"""
    module_name: str
    timestamp: datetime
    success: bool
    data: Dict
    metadata: Dict = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalysisCache:
    """分析缓存"""
    papers: List[Paper]
    preprocessed_data: Dict
    analysis_results: Dict[str, AnalysisResult]
    timestamp: datetime
    config_hash: str  # 配置变更时失效