"""
数据采集模块

从OpenAlex/CrossRef/Semantic Scholar采集论文数据
"""

import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from core.data_model import Paper, DataSource
from core.config import AnalysisConfig


class DataCollector:
    """多源数据采集器"""
    
    def __init__(self, config: AnalysisConfig, cache_dir: str = "output/cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
    
    def collect(self) -> List[Paper]:
        """从所有启用的数据源采集"""
        all_papers = []
        seen_dois = set()
        
        sources = self.config.sources
        
        if sources.get("openalex", {}).get("enabled", False):
            papers = self._collect_openalex()
            for p in papers:
                if p.doi not in seen_dois:
                    all_papers.append(p)
                    seen_dois.add(p.doi)
        
        if sources.get("crossref", {}).get("enabled", False):
            papers = self._collect_crossref()
            for p in papers:
                if p.doi not in seen_dois:
                    all_papers.append(p)
                    seen_dois.add(p.doi)
        
        if sources.get("semanticscholar", {}).get("enabled", False):
            papers = self._collect_semantic_scholar()
            for p in papers:
                if p.doi not in seen_dois:
                    all_papers.append(p)
                    seen_dois.add(p.doi)
        
        # 过滤排除词
        all_papers = self._filter_excludes(all_papers)
        
        # 保存缓存
        self._save_cache(all_papers)
        
        return all_papers
    
    def _collect_openalex(self) -> List[Paper]:
        """从OpenAlex采集"""
        print("  [OpenAlex] 采集中...")
        papers = []
        max_results = self.config.sources.get("openalex", {}).get("max_results", 300)
        
        try:
            import requests
            if not self.session:
                self.session = requests.Session()
            
            for query in self.config.search_queries:
                url = "https://api.openalex.org/works"
                params = {
                    "search": query,
                    "per_page": min(50, max_results),
                    "sort": "relevance_score:desc",
                    "filter": f"from_publication_date:{self.config.year_from}-01-01"
                }
                
                try:
                    resp = self.session.get(url, params=params, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        for work in data.get("results", []):
                            paper = self._parse_openalex_work(work)
                            if paper:
                                papers.append(paper)
                        time.sleep(0.5)
                except Exception as e:
                    print(f"    请求失败: {e}")
            
            print(f"    采集 {len(papers)} 篇")
        except ImportError:
            print("    requests未安装，跳过")
        
        return papers
    
    def _collect_crossref(self) -> List[Paper]:
        """从CrossRef采集"""
        print("  [CrossRef] 采集中...")
        papers = []
        max_results = self.config.sources.get("crossref", {}).get("max_results", 200)
        
        try:
            import requests
            if not self.session:
                self.session = requests.Session()
            
            for query in self.config.search_queries:
                url = "https://api.crossref.org/works"
                params = {
                    "query": query,
                    "rows": min(50, max_results),
                    "filter": f"from-pub-date:{self.config.year_from}",
                    "sort": "relevance",
                    "order": "desc"
                }
                
                try:
                    resp = self.session.get(url, params=params, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("message", {}).get("items", []):
                            paper = self._parse_crossref_work(item)
                            if paper:
                                papers.append(paper)
                        time.sleep(1.0)
                except Exception as e:
                    print(f"    请求失败: {e}")
            
            print(f"    采集 {len(papers)} 篇")
        except ImportError:
            print("    requests未安装，跳过")
        
        return papers
    
    def _collect_semantic_scholar(self) -> List[Paper]:
        """从Semantic Scholar采集"""
        print("  [Semantic Scholar] 采集中...")
        papers = []
        max_results = self.config.sources.get("semanticscholar", {}).get("max_results", 200)
        
        try:
            import requests
            if not self.session:
                self.session = requests.Session()
            
            for query in self.config.search_queries:
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": query,
                    "limit": min(50, max_results),
                    "year": f"{self.config.year_from}-",
                    "fields": "title,authors,year,citationCount,abstract,venue,externalIds,url"
                }
                
                try:
                    resp = self.session.get(url, params=params, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("data", []):
                            paper = self._parse_s2_paper(item)
                            if paper:
                                papers.append(paper)
                        time.sleep(1.0)
                except Exception as e:
                    print(f"    请求失败: {e}")
            
            print(f"    采集 {len(papers)} 篇")
        except ImportError:
            print("    requests未安装，跳过")
        
        return papers
    
    def _parse_openalex_work(self, work: Dict) -> Optional[Paper]:
        try:
            doi = work.get("doi", "") or ""
            if doi.startswith("https://doi.org/"):
                doi = doi[16:]
            
            authors = []
            for a in work.get("authorships", []):
                name = a.get("author", {}).get("display_name", "")
                if name:
                    authors.append(name)
            
            year = work.get("publication_year", 0)
            venue = work.get("primary_location", {}).get("source", {}).get("display_name", "") or ""
            citations = work.get("cited_by_count", 0)
            title = work.get("title", "") or ""
            abstract_parts = work.get("abstract_inverted_index") or {}
            abstract = ""
            if abstract_parts:
                words = {}
                for word, positions in abstract_parts.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words[k] for k in sorted(words.keys()))
            
            return Paper(
                id=f"oa_{work.get('id', '')}",
                title=title,
                authors=authors[:5],
                year=year,
                citations=citations,
                abstract=abstract[:500],
                venue=venue,
                doi=doi,
                source="openalex",
                url=work.get("id", ""),
                confidence="medium",
                confidence_reason="待分类"
            )
        except Exception:
            return None
    
    def _parse_crossref_work(self, item: Dict) -> Optional[Paper]:
        try:
            doi = item.get("DOI", "")
            title = item.get("title", [""])[0]
            authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() 
                      for a in item.get("author", [])[:5]]
            year = item.get("published-print", {}).get("date-parts", [[0]])[0][0]
            venue = item.get("container-title", [""])[0] if item.get("container-title") else ""
            citations = item.get("is-referenced-by-count", 0)
            abstract = item.get("abstract", "")[:500]
            
            return Paper(
                id=f"cr_{doi}",
                title=title,
                authors=authors,
                year=year,
                citations=citations,
                abstract=abstract,
                venue=venue,
                doi=doi,
                source="crossref",
                url=f"https://doi.org/{doi}",
                confidence="medium",
                confidence_reason="待分类"
            )
        except Exception:
            return None
    
    def _parse_s2_paper(self, item: Dict) -> Optional[Paper]:
        try:
            ext_ids = item.get("externalIds", {})
            doi = ext_ids.get("DOI", "")
            title = item.get("title", "") or ""
            authors = [a.get("name", "") for a in item.get("authors", [])[:5] if a.get("name")]
            year = item.get("year", 0) or 0
            venue = item.get("venue", "") or ""
            citations = item.get("citationCount", 0) or 0
            abstract = item.get("abstract", "") or ""
            
            return Paper(
                id=f"s2_{item.get('paperId', '')}",
                title=title,
                authors=authors,
                year=year,
                citations=citations,
                abstract=abstract[:500],
                venue=venue,
                doi=doi,
                source="semanticscholar",
                url=item.get("url", ""),
                confidence="medium",
                confidence_reason="待分类"
            )
        except Exception:
            return None
    
    def _filter_excludes(self, papers: List[Paper]) -> List[Paper]:
        if not self.config.exclude_terms:
            return papers
        
        filtered = []
        for p in papers:
            text = f"{p.title} {p.abstract}".lower()
            excluded = False
            for term in self.config.exclude_terms:
                if term.lower() in text:
                    excluded = True
                    break
            if not excluded:
                filtered.append(p)
        
        print(f"  排除词过滤: {len(papers)} → {len(filtered)}篇")
        return filtered
    
    def _save_cache(self, papers: List[Paper]):
        cache_file = self.cache_dir / "collected_papers.json"
        data = {
            "timestamp": datetime.now().isoformat(),
            "domain": self.config.domain_name,
            "total_papers": len(papers),
            "papers": [p.to_dict() for p in papers],
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"  缓存已保存: {cache_file}")
