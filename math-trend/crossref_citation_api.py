"""
CrossRef引用API接入模块

功能：
1. 获取论文的真实引用关系
2. 构建有向引用网络
3. 分析引用模式和知识传播路径
"""

import requests
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from unified_data_loader import UnifiedDataLoader, Paper, load_cement_storage_data

OUTPUT_DIR = Path(__file__).parent / "output" / "crossref_citations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CrossRefCitationAPI:
    """CrossRef引用API客户端"""
    
    def __init__(self, email: str = "user@example.com"):
        self.base_url = "https://api.crossref.org"
        self.email = email
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'MathTrendBot/1.0 (mailto:{email})'
        })
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def get_work_citations(self, doi: str) -> Dict:
        """
        获取指定DOI的引用信息
        
        Args:
            doi: 论文DOI
        """
        if not doi:
            return {"error": "No DOI provided"}
        
        url = f"{self.base_url}/works/{doi}"
        
        try:
            response = self.session.get(url, timeout=10)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                work = data.get("message", {})
                
                return {
                    "doi": doi,
                    "title": work.get("title", [""])[0],
                    "citation_count": work.get("is-referenced-by-count", 0),
                    "reference_count": work.get("reference-count", 0),
                    "published": work.get("published-print", {}).get("date-parts", [[0]])[0][0],
                    "publisher": work.get("publisher", ""),
                    "type": work.get("type", ""),
                    "references": work.get("reference", []),
                    "subject": work.get("subject", []),
                    "container_title": work.get("container-title", [""])[0] if work.get("container-title") else "",
                    "authors": [f"{a.get('given', '')} {a.get('family', '')}" 
                               for a in work.get("author", [])[:5]]
                }
            else:
                return {"error": f"HTTP {response.status_code}", "doi": doi}
                
        except Exception as e:
            return {"error": str(e), "doi": doi}
    
    def get_citing_works(self, doi: str, rows: int = 20) -> List[Dict]:
        """
        获取引用指定DOI的论文列表
        
        Args:
            doi: 被引用论文的DOI
            rows: 返回结果数量
        """
        if not doi:
            return []
        
        url = f"{self.base_url}/works"
        params = {
            "filter": f"cites:{doi}",
            "rows": rows,
            "sort": "published",
            "order": "desc"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])
                
                citing_works = []
                for item in items:
                    citing_works.append({
                        "doi": item.get("DOI", ""),
                        "title": item.get("title", [""])[0] if item.get("title") else "",
                        "year": item.get("published-print", {}).get("date-parts", [[0]])[0][0],
                        "authors": [f"{a.get('given', '')} {a.get('family', '')}" 
                                   for a in item.get("author", [])[:3]],
                        "container_title": item.get("container-title", [""])[0] if item.get("container-title") else ""
                    })
                
                return citing_works
            else:
                return []
                
        except Exception as e:
            print(f"  获取引用失败: {e}")
            return []
    
    def build_real_citation_network(self, papers: List[Paper], 
                                    max_papers: int = 50) -> Dict[str, Dict]:
        """
        构建真实引用网络
        
        Args:
            papers: 论文列表
            max_papers: 最大处理论文数（API限制）
        """
        print(f"\n构建真实引用网络（最多{max_papers}篇）...")
        
        network = {
            "nodes": {},
            "edges": [],
            "stats": {
                "total_papers": 0,
                "with_doi": 0,
                "with_citations": 0,
                "total_citations": 0
            }
        }
        
        # 选择有DOI的论文
        papers_with_doi = [p for p in papers if p.doi][:max_papers]
        print(f"  处理 {len(papers_with_doi)} 篇有DOI的论文")
        
        for i, paper in enumerate(papers_with_doi):
            print(f"  [{i+1}/{len(papers_with_doi)}] {paper.title[:40]}...")
            
            # 获取引用信息
            citation_info = self.get_work_citations(paper.doi)
            
            if "error" not in citation_info:
                network["nodes"][paper.id] = {
                    "paper_id": paper.id,
                    "doi": paper.doi,
                    "title": citation_info.get("title", paper.title),
                    "year": paper.year,
                    "citation_count": citation_info.get("citation_count", 0),
                    "reference_count": citation_info.get("reference_count", 0),
                    "authors": citation_info.get("authors", paper.authors),
                    "subject": citation_info.get("subject", [])
                }
                
                network["stats"]["with_citations"] += 1
                network["stats"]["total_citations"] += citation_info.get("citation_count", 0)
                
                # 获取引用该论文的论文
                citing_works = self.get_citing_works(paper.doi, rows=10)
                
                for citing in citing_works:
                    network["edges"].append({
                        "from": citing.get("doi", ""),
                        "to": paper.doi,
                        "from_year": citing.get("year", 0),
                        "to_year": paper.year,
                        "type": "citation"
                    })
            
            network["stats"]["with_doi"] += 1
            
            # 每10篇暂停一下
            if (i + 1) % 10 == 0:
                print(f"    已处理 {i+1} 篇，暂停...")
                time.sleep(1)
        
        network["stats"]["total_papers"] = len(papers_with_doi)
        
        print(f"\n  ✅ 网络构建完成")
        print(f"  节点数: {len(network['nodes'])}")
        print(f"  边数: {len(network['edges'])}")
        print(f"  总被引: {network['stats']['total_citations']}")
        
        return network
    
    def analyze_citation_patterns(self, network: Dict) -> Dict:
        """分析引用模式"""
        print("\n分析引用模式...")
        
        patterns = {
            "temporal": defaultdict(int),
            "subject_flow": defaultdict(int),
            "high_impact_papers": [],
            "citation_velocity": []
        }
        
        # 时间模式
        for edge in network["edges"]:
            if edge["from_year"] > 0 and edge["to_year"] > 0:
                lag = edge["from_year"] - edge["to_year"]
                patterns["temporal"][lag] += 1
        
        # 高影响力论文
        for paper_id, info in network["nodes"].items():
            if info["citation_count"] > 10:
                patterns["high_impact_papers"].append({
                    "paper_id": paper_id,
                    "title": info["title"][:60],
                    "citations": info["citation_count"],
                    "year": info["year"]
                })
        
        # 按被引排序
        patterns["high_impact_papers"].sort(
            key=lambda x: x["citations"], 
            reverse=True
        )
        
        # 引用速度（年均被引）
        current_year = 2026
        for paper_id, info in network["nodes"].items():
            years_since = max(1, current_year - info["year"])
            velocity = info["citation_count"] / years_since
            patterns["citation_velocity"].append({
                "paper_id": paper_id,
                "velocity": velocity,
                "total": info["citation_count"]
            })
        
        patterns["citation_velocity"].sort(
            key=lambda x: x["velocity"], 
            reverse=True
        )
        
        return patterns
    
    def generate_citation_report(self, network: Dict, patterns: Dict) -> str:
        """生成引用分析报告"""
        print("\n生成引用网络分析报告...")
        
        report = f"""# CrossRef真实引用网络分析报告

**数据来源**: CrossRef API  
**分析方法**: 真实DOI引用关系  
**网络规模**: {len(network['nodes'])}节点, {len(network['edges'])}边  
**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## 1. 网络概览

### 1.1 基本统计

| 指标 | 数值 |
|-----|------|
| 总论文数 | {network['stats']['total_papers']} |
| 有DOI论文 | {network['stats']['with_doi']} |
| 成功获取引用 | {network['stats']['with_citations']} |
| 引用关系数 | {len(network['edges'])} |
| 总被引次数 | {network['stats']['total_citations']} |
| 平均每篇被引 | {network['stats']['total_citations'] / max(network['stats']['with_citations'], 1):.1f} |

### 1.2 与模拟网络对比

| 特征 | 模拟网络 | 真实引用网络 |
|-----|---------|------------|
| 关系类型 | 合作+语义+期刊 | 真实引用 |
| 方向性 | 无向 | 有向 |
| 时间信息 | 无 | 有 |
| 权威性 | 低 | 高 |
| 数据完整 | 完整 | 部分（仅DOI） |

---

## 2. 高影响力论文

### 2.1 Top 10 高被引论文

| 排名 | 论文 | 被引数 | 年份 |
|-----|------|-------|------|
"""
        
        for i, paper in enumerate(patterns["high_impact_papers"][:10], 1):
            report += f"| {i} | {paper['title'][:50]}... | {paper['citations']} | {paper['year']} |\n"
        
        report += f"""
### 2.2 引用速度Top 5

| 排名 | 论文 | 年均被引 | 总被引 |
|-----|------|---------|-------|
"""
        
        for i, paper in enumerate(patterns["citation_velocity"][:5], 1):
            info = network["nodes"].get(paper["paper_id"], {})
            title = info.get("title", "")[:40]
            report += f"| {i} | {title}... | {paper['velocity']:.1f} | {paper['total']} |\n"
        
        report += f"""
---

## 3. 引用时间模式

### 3.1 引用延迟分布

| 延迟（年） | 引用次数 | 占比 |
|-----------|---------|------|
"""
        
        total_citations = sum(patterns["temporal"].values())
        for lag in sorted(patterns["temporal"].keys()):
            count = patterns["temporal"][lag]
            pct = count / total_citations * 100 if total_citations > 0 else 0
            report += f"| {lag} | {count} | {pct:.1f}% |\n"
        
        avg_lag = np.average(list(patterns["temporal"].keys()), 
                            weights=list(patterns["temporal"].values())) if patterns["temporal"] else 0
        
        report += f"""
**平均引用延迟**: {avg_lag:.1f}年

### 3.2 时间模式解读

- 快速引用（0-1年）: 领域热点，即时关注
- 标准引用（2-3年）: 常规知识传播
- 延迟引用（4年+）: 经典工作，长期影响

---

## 4. 知识传播分析

### 4.1 传播路径

基于真实引用关系：

```
原始论文 → 直接引用 → 间接引用 → 领域基础
```

### 4.2 传播特征

- **引用半衰期**: 论文被引数量减半的时间
- **引用峰值**: 被引数量达到最高的年份
- **引用持续性**: 长期被引 vs 短期爆发

---

## 5. 方法说明

### 5.1 数据来源

**CrossRef API**:
- 免费学术API
- 覆盖1.5亿+学术作品
- 包含DOI注册信息
- 提供引用计数（非完整列表）

### 5.2 数据局限

1. **DOI覆盖率**: 并非所有论文都有DOI
2. **引用计数**: 可能不完整（依赖注册）
3. **API限制**: 请求频率限制
4. **时间延迟**: 最新引用可能未收录

### 5.3 与模拟网络的区别

| 方面 | 模拟网络 | 真实引用网络 |
|-----|---------|------------|
| 构建方式 | 基于属性推断 | 基于真实引用 |
| 准确性 | 估计 | 真实（但可能不完整） |
| 覆盖度 | 100% | 依赖DOI覆盖率 |
| 方向性 | 无向 | 有向 |
| 权重 | 统一 | 可基于引用次数 |

---

## 6. 应用建议

### 6.1 研究应用

1. **影响力评估**: 基于真实被引数
2. **知识追踪**: 追踪引用链
3. **领域映射**: 基于引用关系映射知识结构
4. **预测**: 基于引用模式预测未来影响

### 6.2 改进建议

1. **提高DOI覆盖率**: 补充缺失DOI
2. **完整引用列表**: 获取所有引用论文
3. **引用上下文**: 分析引用动机（支持/对比）
4. **动态更新**: 定期更新引用数据

---

*本报告基于CrossRef真实引用数据生成*  
*数据可能不完整，仅供参考*
"""
        
        return report
    
    def save_network(self, network: Dict):
        """保存网络数据"""
        network_path = OUTPUT_DIR / "citation_network.json"
        with open(network_path, 'w', encoding='utf-8') as f:
            json.dump(network, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 网络数据已保存: {network_path}")
    
    def save_report(self, report_content: str):
        """保存报告"""
        report_path = OUTPUT_DIR / "crossref_citation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✅ 引用报告已保存: {report_path}")
        return report_path


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " CrossRef真实引用网络分析 ".center(76) + "█")
    print("█" * 80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = load_cement_storage_data()
    print(f"  加载 {len(loader.high_confidence_papers)} 篇高置信度论文")
    
    # 2. 创建API客户端
    print("\n2. 创建CrossRef API客户端...")
    api = CrossRefCitationAPI()
    
    # 3. 构建真实引用网络
    print("\n3. 构建真实引用网络...")
    network = api.build_real_citation_network(
        loader.high_confidence_papers,
        max_papers=30  # 限制数量避免API限制
    )
    
    # 4. 分析引用模式
    print("\n4. 分析引用模式...")
    patterns = api.analyze_citation_patterns(network)
    
    # 5. 保存网络
    print("\n5. 保存网络数据...")
    api.save_network(network)
    
    # 6. 生成报告
    print("\n6. 生成引用分析报告...")
    report = api.generate_citation_report(network, patterns)
    report_path = api.save_report(report)
    
    # 7. 总结
    print("\n" + "█" * 80)
    print("█" + " 分析完成 ".center(76) + "█")
    print("█" * 80)
    
    print(f"\n📊 核心结果:")
    print(f"  处理论文: {network['stats']['total_papers']}篇")
    print(f"  成功获取: {network['stats']['with_citations']}篇")
    print(f"  引用关系: {len(network['edges'])}条")
    print(f"  总被引: {network['stats']['total_citations']}")
    
    print(f"\n🏆 Top 3 高被引论文:")
    for i, paper in enumerate(patterns["high_impact_papers"][:3], 1):
        print(f"  {i}. {paper['title'][:50]}... ({paper['citations']}被引)")
    
    print(f"\n📄 报告文件: {report_path}")
    print(f"💾 网络数据: {OUTPUT_DIR / 'citation_network.json'}")
    
    print()


if __name__ == "__main__":
    main()
