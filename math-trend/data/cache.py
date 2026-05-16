"""
缓存管理

避免重复采集和计算
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

from core.data_model import Paper
from core.config import AnalysisConfig


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "output/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_config_hash(self, config: AnalysisConfig) -> str:
        """计算配置哈希，配置变更时缓存失效"""
        config_str = f"{config.domain_name}_{config.year_from}_{config.year_to}_{sorted(config.search_queries)}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def load_papers(self, config: AnalysisConfig) -> Optional[List[Paper]]:
        """从缓存加载论文"""
        config_hash = self.get_config_hash(config)
        cache_file = self.cache_dir / f"papers_{config_hash}.json"
        
        if not cache_file.exists():
            return None
        
        # 检查缓存是否过期（7天）
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=7):
            return None
        
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        papers = [Paper.from_dict(p) for p in data.get("papers", [])]
        print(f"  从缓存加载 {len(papers)} 篇论文（{mtime.strftime('%Y-%m-%d')}）")
        return papers
    
    def save_papers(self, papers: List[Paper], config: AnalysisConfig):
        """保存论文到缓存"""
        config_hash = self.get_config_hash(config)
        cache_file = self.cache_dir / f"papers_{config_hash}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "config_hash": config_hash,
            "domain": config.domain_name,
            "total_papers": len(papers),
            "papers": [p.to_dict() for p in papers],
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  缓存已保存: {cache_file}")
    
    def is_fresh(self, config: AnalysisConfig, max_age_days: int = 7) -> bool:
        """检查缓存是否新鲜"""
        config_hash = self.get_config_hash(config)
        cache_file = self.cache_dir / f"papers_{config_hash}.json"
        
        if not cache_file.exists():
            return False
        
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return (datetime.now() - mtime).days < max_age_days
