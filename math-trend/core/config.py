"""
Math-Trend 核心配置管理

用户只需修改此文件即可切换研究领域，无需改动分析代码
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AnalysisConfig:
    """
    研究分析配置
    
    所有可配置项集中管理
    """
    # 研究领域定义
    domain_name: str = "水泥基电化学储能"
    search_queries: List[str] = field(default_factory=lambda: [
        "structural supercapacitor cement concrete",
        "cement based energy storage electrochemical",
        "multifunctional concrete supercapacitor battery",
        "conductive cement composite electrode",
    ])
    
    # 排除词（过滤不相关结果）
    exclude_terms: List[str] = field(default_factory=lambda: [
        "lithium battery",        # 传统锂电
        "solar cell",             # 太阳能
        "fuel cell",              # 燃料电池
    ])
    
    # 时间范围
    year_from: int = 2018
    year_to: int = 2026
    
    # 数据源配置
    sources: Dict[str, Dict] = field(default_factory=lambda: {
        "openalex": {"max_results": 300, "enabled": True},
        "crossref": {"max_results": 200, "enabled": True},
        "semanticscholar": {"max_results": 200, "enabled": True},
    })
    
    # 置信度分层阈值
    confidence_thresholds: Dict[str, Dict] = field(default_factory=lambda: {
        "high": {"min_citations": 5, "top_journals": True, "experienced_author": True},
        "medium": {"min_citations": 1, "top_journals": True},
        "low": {"min_citations": 0},
    })
    
    # 已知高质量期刊（用于置信度评估）
    tier1_journals: List[str] = field(default_factory=lambda: [
        "Cement and Concrete Research",
        "Cement and Concrete Composites",
        "Nature Materials",
        "Nature Energy",
        "Energy & Environmental Science",
        "Advanced Materials",
        "Advanced Energy Materials",
        "ACS Nano",
        "Nano Letters",
        "Journal of Materials Chemistry A",
    ])
    
    # 分析模块选择
    enabled_modules: List[str] = field(default_factory=lambda: [
        # 基础分析
        "basic.statistics",
        "basic.cross_domain",
        "basic.journal_ranking",
        # 高级分析
        "advanced.topic_modeling",
        "advanced.citation_network",
        "advanced.time_series",
        "advanced.cross_field_comparison",
    ])
    
    # 输出配置
    output_dir: str = "output"
    report_format: str = "markdown"  # markdown / json
    report_template: str = "data_driven"  # data_driven / comprehensive
