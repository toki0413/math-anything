"""
外部数据库爬虫模块

接入多个学术数据库获取期刊指标：
1. Scopus - CiteScore, SNIP, SJR
2. Web of Science - JCR, Impact Factor
3. DOAJ - 开放获取期刊目录
4. Google Scholar - h-index, 被引数
5. CrossRef - 期刊元数据
"""

import requests
import json
import time
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

OUTPUT_DIR = Path(__file__).parent / "output" / "external_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ExternalDataCrawler:
    """外部学术数据库爬虫"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.cache = {}
        
    def search_journal_scopus(self, journal_name: str) -> Dict:
        """
        从Scopus获取期刊指标
        数据来源: Scopus Sources页面
        """
        print(f"  搜索Scopus: {journal_name}")
        
        # 使用Scopus API或网页搜索
        # 这里使用模拟数据（实际应用需要API key）
        scopus_data = self._simulate_scopus_data(journal_name)
        
        return {
            "source": "Scopus",
            "cite_score": scopus_data.get("cite_score", 0),
            "snip": scopus_data.get("snip", 0),
            "sjr": scopus_data.get("sjr", 0),
            "h_index": scopus_data.get("h_index", 0),
            "quartile": scopus_data.get("quartile", "N/A"),
            "coverage": scopus_data.get("coverage", "N/A")
        }
    
    def search_journal_wos(self, journal_name: str) -> Dict:
        """
        从Web of Science获取JCR指标
        数据来源: JCR Reports
        """
        print(f"  搜索WoS: {journal_name}")
        
        # 使用JCR数据（实际应用需要订阅）
        wos_data = self._simulate_wos_data(journal_name)
        
        return {
            "source": "Web of Science",
            "jcr_if": wos_data.get("jcr_if", 0),
            "five_year_if": wos_data.get("five_year_if", 0),
            "immediacy_index": wos_data.get("immediacy_index", 0),
            "article_influence": wos_data.get("article_influence", 0),
            "quartile": wos_data.get("quartile", "N/A")
        }
    
    def search_journal_doaj(self, journal_name: str) -> Dict:
        """
        从DOAJ获取开放获取期刊信息
        数据来源: DOAJ API (免费)
        """
        print(f"  搜索DOAJ: {journal_name}")
        
        try:
            # DOAJ API
            url = "https://doaj.org/api/v3/search/journals"
            params = {
                "query": journal_name,
                "pageSize": 5
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    journal_data = results[0].get("bibjson", {})
                    return {
                        "source": "DOAJ",
                        "is_oa": True,
                        "apc": journal_data.get("apc", {}).get("average_price", 0),
                        "publisher": journal_data.get("publisher", {}).get("name", ""),
                        "license": journal_data.get("license", [{}])[0].get("type", ""),
                        "subjects": [s.get("term", "") for s in journal_data.get("subject", [])[:3]]
                    }
            
            return {"source": "DOAJ", "is_oa": False, "note": "Not found or API error"}
            
        except Exception as e:
            print(f"    DOAJ搜索失败: {e}")
            return {"source": "DOAJ", "is_oa": False, "error": str(e)}
    
    def search_journal_google_scholar(self, journal_name: str) -> Dict:
        """
        从Google Scholar获取h-index
        数据来源: Google Scholar Metrics
        """
        print(f"  搜索Google Scholar: {journal_name}")
        
        # Google Scholar需要特殊处理（反爬虫）
        # 这里使用模拟数据
        gs_data = self._simulate_google_scholar_data(journal_name)
        
        return {
            "source": "Google Scholar",
            "h5_index": gs_data.get("h5_index", 0),
            "h5_median": gs_data.get("h5_median", 0),
            "language": gs_data.get("language", "English")
        }
    
    def search_journal_crossref(self, journal_name: str) -> Dict:
        """
        从CrossRef获取期刊元数据
        数据来源: CrossRef API (免费)
        """
        print(f"  搜索CrossRef: {journal_name}")
        
        try:
            url = "https://api.crossref.org/journals"
            params = {
                "query": journal_name,
                "rows": 5
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])
                
                if items:
                    journal = items[0]
                    return {
                        "source": "CrossRef",
                        "issn": journal.get("ISSN", []),
                        "publisher": journal.get("publisher", ""),
                        "total_dois": journal.get("counts", {}).get("total-dois", 0),
                        "current_dois": journal.get("counts", {}).get("current-dois", 0),
                        "subjects": [s.get("name", "") for s in journal.get("subjects", [])[:3]]
                    }
            
            return {"source": "CrossRef", "note": "Not found"}
            
        except Exception as e:
            print(f"    CrossRef搜索失败: {e}")
            return {"source": "CrossRef", "error": str(e)}
    
    def crawl_journal_metrics(self, journal_name: str) -> Dict:
        """
        综合爬取多个数据库的期刊指标
        """
        print(f"\n综合爬取: {journal_name}")
        print("-" * 50)
        
        results = {
            "journal_name": journal_name,
            "crawl_time": datetime.now().isoformat(),
            "sources": {}
        }
        
        # 从多个源获取数据
        results["sources"]["scopus"] = self.search_journal_scopus(journal_name)
        time.sleep(0.5)  # 礼貌延迟
        
        results["sources"]["wos"] = self.search_journal_wos(journal_name)
        time.sleep(0.5)
        
        results["sources"]["doaj"] = self.search_journal_doaj(journal_name)
        time.sleep(0.5)
        
        results["sources"]["google_scholar"] = self.search_journal_google_scholar(journal_name)
        time.sleep(0.5)
        
        results["sources"]["crossref"] = self.search_journal_crossref(journal_name)
        
        # 计算综合指标
        results["composite"] = self._calculate_composite_metrics(results["sources"])
        
        return results
    
    def _calculate_composite_metrics(self, sources: Dict) -> Dict:
        """计算综合指标"""
        composite = {
            "impact_factor": 0,
            "h_index": 0,
            "quartile": "N/A",
            "is_oa": False,
            "confidence": "low"
        }
        
        # 收集所有IF数据
        ifs = []
        if sources.get("wos", {}).get("jcr_if", 0) > 0:
            ifs.append(sources["wos"]["jcr_if"])
        if sources.get("scopus", {}).get("cite_score", 0) > 0:
            ifs.append(sources["scopus"]["cite_score"])
        
        if ifs:
            composite["impact_factor"] = sum(ifs) / len(ifs)
            composite["confidence"] = "medium"
        
        # h-index
        hs = []
        if sources.get("scopus", {}).get("h_index", 0) > 0:
            hs.append(sources["scopus"]["h_index"])
        if sources.get("google_scholar", {}).get("h5_index", 0) > 0:
            hs.append(sources["google_scholar"]["h5_index"])
        
        if hs:
            composite["h_index"] = max(hs)
        
        # 开放获取
        composite["is_oa"] = sources.get("doaj", {}).get("is_oa", False)
        
        # 分区
        quartiles = []
        for source in ["wos", "scopus"]:
            q = sources.get(source, {}).get("quartile", "N/A")
            if q != "N/A":
                quartiles.append(q)
        
        if quartiles:
            composite["quartile"] = max(set(quartiles), key=quartiles.count)
            composite["confidence"] = "high"
        
        return composite
    
    def _simulate_scopus_data(self, journal_name: str) -> Dict:
        """模拟Scopus数据（实际应用中会调用真实API）"""
        # 基于真实知识的模拟数据
        simulated = {
            "Cement and Concrete Composites": {"cite_score": 10.2, "snip": 2.8, "sjr": 2.5, "h_index": 95, "quartile": "Q1"},
            "Cement and Concrete Research": {"cite_score": 12.5, "snip": 3.2, "sjr": 3.0, "h_index": 120, "quartile": "Q1"},
            "Journal of Energy Storage": {"cite_score": 9.0, "snip": 1.8, "sjr": 1.5, "h_index": 45, "quartile": "Q1"},
            "Journal of Power Sources": {"cite_score": 10.5, "snip": 2.5, "sjr": 2.2, "h_index": 280, "quartile": "Q1"},
            "Carbon": {"cite_score": 11.8, "snip": 2.2, "sjr": 2.8, "h_index": 180, "quartile": "Q1"},
            "Construction and Building Materials": {"cite_score": 8.2, "snip": 2.0, "sjr": 1.8, "h_index": 130, "quartile": "Q1"},
            "Composite Structures": {"cite_score": 7.0, "snip": 1.9, "sjr": 1.7, "h_index": 110, "quartile": "Q1"},
            "ACS Applied Materials & Interfaces": {"cite_score": 9.5, "snip": 1.5, "sjr": 2.0, "h_index": 200, "quartile": "Q1"},
            "Energy": {"cite_score": 8.8, "snip": 2.0, "sjr": 1.9, "h_index": 160, "quartile": "Q1"},
            "Materials Today": {"cite_score": 26.0, "snip": 4.5, "sjr": 5.5, "h_index": 150, "quartile": "Q1"}
        }
        
        # 模糊匹配
        for key, value in simulated.items():
            if key.lower() in journal_name.lower() or journal_name.lower() in key.lower():
                return value
        
        return {"cite_score": 0, "snip": 0, "sjr": 0, "h_index": 0, "quartile": "N/A"}
    
    def _simulate_wos_data(self, journal_name: str) -> Dict:
        """模拟Web of Science数据"""
        simulated = {
            "Cement and Concrete Composites": {"jcr_if": 9.5, "five_year_if": 9.8, "immediacy_index": 1.2, "article_influence": 1.5, "quartile": "Q1"},
            "Cement and Concrete Research": {"jcr_if": 11.0, "five_year_if": 11.5, "immediacy_index": 1.5, "article_influence": 1.8, "quartile": "Q1"},
            "Journal of Energy Storage": {"jcr_if": 8.5, "five_year_if": 8.8, "immediacy_index": 1.0, "article_influence": 1.2, "quartile": "Q1"},
            "Journal of Power Sources": {"jcr_if": 9.2, "five_year_if": 9.5, "immediacy_index": 1.3, "article_influence": 1.4, "quartile": "Q1"},
            "Carbon": {"jcr_if": 10.5, "five_year_if": 10.8, "immediacy_index": 1.4, "article_influence": 1.6, "quartile": "Q1"},
            "Construction and Building Materials": {"jcr_if": 7.4, "five_year_if": 7.8, "immediacy_index": 0.9, "article_influence": 1.1, "quartile": "Q1"},
            "Composite Structures": {"jcr_if": 6.0, "five_year_if": 6.3, "immediacy_index": 0.8, "article_influence": 0.9, "quartile": "Q1"},
            "ACS Applied Materials & Interfaces": {"jcr_if": 8.3, "five_year_if": 8.6, "immediacy_index": 1.1, "article_influence": 1.3, "quartile": "Q1"},
            "Energy": {"jcr_if": 8.0, "five_year_if": 8.3, "immediacy_index": 1.0, "article_influence": 1.2, "quartile": "Q1"},
            "Materials Today": {"jcr_if": 24.2, "five_year_if": 25.0, "immediacy_index": 3.5, "article_influence": 4.5, "quartile": "Q1"}
        }
        
        for key, value in simulated.items():
            if key.lower() in journal_name.lower() or journal_name.lower() in key.lower():
                return value
        
        return {"jcr_if": 0, "five_year_if": 0, "immediacy_index": 0, "article_influence": 0, "quartile": "N/A"}
    
    def _simulate_google_scholar_data(self, journal_name: str) -> Dict:
        """模拟Google Scholar数据"""
        simulated = {
            "Cement and Concrete Composites": {"h5_index": 85, "h5_median": 120, "language": "English"},
            "Cement and Concrete Research": {"h5_index": 95, "h5_median": 140, "language": "English"},
            "Journal of Energy Storage": {"h5_index": 55, "h5_median": 80, "language": "English"},
            "Journal of Power Sources": {"h5_index": 120, "h5_median": 180, "language": "English"},
            "Carbon": {"h5_index": 110, "h5_median": 160, "language": "English"},
            "Construction and Building Materials": {"h5_index": 90, "h5_median": 130, "language": "English"},
            "Composite Structures": {"h5_index": 75, "h5_median": 110, "language": "English"},
            "Materials Today": {"h5_index": 105, "h5_median": 150, "language": "English"}
        }
        
        for key, value in simulated.items():
            if key.lower() in journal_name.lower() or journal_name.lower() in key.lower():
                return value
        
        return {"h5_index": 0, "h5_median": 0, "language": "English"}
    
    def batch_crawl_journals(self, journal_names: List[str]) -> Dict[str, Dict]:
        """批量爬取多个期刊"""
        print(f"\n批量爬取 {len(journal_names)} 个期刊...")
        print("=" * 60)
        
        results = {}
        for i, journal in enumerate(journal_names, 1):
            print(f"\n[{i}/{len(journal_names)}]")
            result = self.crawl_journal_metrics(journal)
            results[journal] = result
            
            # 保存中间结果
            if i % 5 == 0:
                self._save_cache(results)
        
        # 保存最终结果
        self._save_cache(results, final=True)
        
        return results
    
    def _save_cache(self, data: Dict, final: bool = False):
        """保存缓存"""
        suffix = "_final" if final else ""
        cache_path = OUTPUT_DIR / f"journal_metrics_cache{suffix}.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if final:
            print(f"\n✅ 最终结果已保存: {cache_path}")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " 外部数据库期刊指标爬虫 ".center(76) + "█")
    print("█" * 80)
    
    crawler = ExternalDataCrawler()
    
    # 测试期刊列表
    test_journals = [
        "Cement and Concrete Composites",
        "Cement and Concrete Research",
        "Journal of Energy Storage",
        "Journal of Power Sources",
        "Carbon",
        "Construction and Building Materials",
        "Composite Structures"
    ]
    
    # 批量爬取
    results = crawler.batch_crawl_journals(test_journals)
    
    # 显示结果摘要
    print("\n" + "=" * 80)
    print("爬取结果摘要")
    print("=" * 80)
    
    print(f"\n{'期刊':<35} {'IF':<8} {'h-index':<8} {'Q区':<6} {'置信度':<8}")
    print("-" * 70)
    
    for journal, data in results.items():
        composite = data.get("composite", {})
        print(f"{journal[:33]:<35} {composite.get('impact_factor', 0):<8.1f} {composite.get('h_index', 0):<8} {composite.get('quartile', 'N/A'):<6} {composite.get('confidence', 'N/A'):<8}")
    
    print("\n" + "█" * 80)
    print("█" + " 爬取完成 ".center(76) + "█")
    print("█" * 80)
    
    print("\n📊 数据源覆盖:")
    print("  ✅ Scopus - CiteScore, SNIP, SJR")
    print("  ✅ Web of Science - JCR IF, 5-year IF")
    print("  ✅ DOAJ - 开放获取信息")
    print("  ✅ Google Scholar - h5-index")
    print("  ✅ CrossRef - 期刊元数据")
    
    print("\n💡 说明:")
    print("  • 当前使用模拟数据演示功能")
    print("  • 实际应用需要API key或订阅")
    print("  • DOAJ和CrossRef为真实API调用")
    print("  • 支持批量爬取和缓存机制")
    
    print()


if __name__ == "__main__":
    main()
